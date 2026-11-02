"""Live results from the journal and watchlists, for any stretch of days."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

MOVES = ("entry", "exit", "cancel", "reject", "dry_signal", "confirm", "drop", "halt")


def _rows(root: Path) -> list[dict]:
    path = root / "journal" / "trades.csv"
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def closed_trades(root: Path) -> list[dict]:
    """Each exit paired with its entry: day, strategy, regime, $ P&L, % of the trade and R."""
    rows = _rows(root)
    regime, open_, out = {}, {}, []
    for r in rows:
        day = r["time"][:10]
        if r["event"] == "regime":
            regime[day] = r["note"].split(":", 1)[0].strip()
        elif r["event"] == "entry":
            open_[(r["symbol"], r["strategy"])] = r
        elif r["event"] == "exit":
            e = open_.pop((r["symbol"], r["strategy"]), None)
            qty, pnl = _num(r["qty"]) or 0, _num(r["pnl"]) or 0
            entry = _num(e["price"]) if e else None
            stop = _num(e["stop"]) if e else _num(r["stop"])
            risk = abs(entry - stop) * qty if entry and stop and qty else 0
            out.append({
                "day": day, "symbol": r["symbol"], "strategy": r["strategy"], "regime": regime.get(day, "?"),
                "pnl": pnl, "pct_of_trade": round(pnl / (entry * qty) * 100, 2) if entry and qty else None,
                "r": round(pnl / risk, 2) if risk else None,
            })
        elif r["event"] == "cancel":
            open_.pop((r["symbol"], r["strategy"]), None)
    return out


def days(root: Path) -> dict[str, dict]:
    """What happened each day the bot ran: when it started trading, market type, picks, what fired,
    what was blocked and why."""
    out: dict[str, dict] = {}
    for r in _rows(root):
        d = out.setdefault(r["time"][:10], {"started": r["time"][11:16], "regime": "?", "picks": 0, **{e: 0 for e in MOVES},
                                            "pnl": 0.0, "blocked_because": Counter()})
        if r["event"] == "regime":
            d["regime"] = r["note"].split(":", 1)[0].strip()
        elif r["event"] in MOVES:
            d[r["event"]] += 1
        if r["event"] == "exit":
            d["pnl"] += _num(r["pnl"]) or 0
        if r["event"] in ("reject", "drop"):
            d["blocked_because"][re.sub(r"[-+]?\d[\d.,]*", "#", r["note"])[:80]] += 1
    for day, d in out.items():
        wl = root / "watchlists" / f"{day}.json"
        if wl.exists():
            d["picks"] = len(json.loads(wl.read_text()).get("candidates", []))
    return out


def summarize(trades: list[dict], equity: float) -> dict:
    """Win rate, P&L as % of the portfolio, average % of the trade, expectancy in R, profit factor."""
    if not trades:
        return {"trades": 0}
    pnl = [t["pnl"] for t in trades]
    rs = [t["r"] for t in trades if t["r"] is not None]
    pct = [t["pct_of_trade"] for t in trades if t["pct_of_trade"] is not None]
    gains, losses = sum(p for p in pnl if p > 0), -sum(p for p in pnl if p < 0)
    return {
        "trades": len(trades),
        "win_rate_pct": round(100 * sum(p > 0 for p in pnl) / len(pnl), 1),
        "pnl_usd": round(sum(pnl), 2),
        "pnl_pct_of_portfolio": round(100 * sum(pnl) / equity, 3),
        "avg_pct_of_trade": round(sum(pct) / len(pct), 2) if pct else None,
        "expectancy_r": round(sum(rs) / len(rs), 2) if rs else None,
        "profit_factor": round(gains / losses, 2) if losses else None,
        "best_r": max(rs) if rs else None,
        "worst_r": min(rs) if rs else None,
    }


def breakdown(trades: list[dict], equity: float) -> dict:
    out = {"overall": summarize(trades, equity)}
    for key in ("strategy", "regime"):
        groups: dict[str, list] = {}
        for t in trades:
            groups.setdefault(t[key], []).append(t)
        out[f"by_{key}"] = {k: summarize(v, equity) for k, v in sorted(groups.items())}
    return out


def window(trades: list[dict], day_info: dict, start: date, end: date, equity: float) -> dict:
    """Results between two dates, plus a day-by-day (or, for long stretches, week-by-week) log."""
    s, e = start.isoformat(), end.isoformat()
    t = [x for x in trades if s <= x["day"] <= e]
    log = {d: v for d, v in sorted(day_info.items()) if s <= d <= e}
    blocked = Counter()
    for v in log.values():
        blocked.update(v["blocked_because"])
    days_ = [{"day": d, **{k: (round(x, 2) if k == "pnl" else x) for k, x in v.items() if k != "blocked_because"}}
             for d, v in log.items()]
    if len(days_) > 70:   # quarters and years: weekly rows keep the prompt readable
        weeks: dict[str, dict] = {}
        for row in days_:
            wk = date.fromisoformat(row["day"]).strftime("%G-W%V")
            w = weeks.setdefault(wk, {"week": wk, "days": 0, "picks": 0, "entry": 0, "pnl": 0.0, "regimes": Counter()})
            w["days"] += 1
            w["picks"] += row["picks"]
            w["entry"] += row["entry"]
            w["pnl"] = round(w["pnl"] + row["pnl"], 2)
            w["regimes"][row["regime"]] += 1
        days_ = [{**w, "regimes": dict(w["regimes"])} for w in weeks.values()]
    return {
        "from": s, "to": e, "days_run": len(log),
        "days_with_trades": sum(1 for v in log.values() if v["entry"]),
        "market_types": dict(Counter(v["regime"] for v in log.values())),
        **breakdown(t, equity),
        "top_reasons_setups_were_blocked": dict(blocked.most_common(12)),
        "log": days_,
    }


def drawdown_pct(trades: list[dict], since: str, equity: float) -> float:
    """Largest peak-to-trough drop in realized P&L since a date, as % of the portfolio (negative)."""
    peak = run = worst = 0.0
    for t in trades:
        if t["day"] >= since:
            run += t["pnl"]
            peak = max(peak, run)
            worst = min(worst, run - peak)
    return 100 * worst / equity
