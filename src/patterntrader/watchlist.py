"""Daily watchlist files: the scanner writes them, you approve, the trader reads them."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from patterntrader.config import ET, ROOT
from patterntrader.models import SetupContext

DIR = ROOT / "watchlists"


def path_for(day: date) -> Path:
    return DIR / f"{day.isoformat()}.json"


def save(day: date, regime: str, notes: list[str], picks: list[SetupContext], others: list[SetupContext]) -> Path:
    DIR.mkdir(exist_ok=True)
    payload = {
        "day": day.isoformat(),
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "market_regime": regime,
        "market_notes": notes,
        "candidates": [{**c.to_dict(), "status": "pending"} for c in picks],
        "also_scanned": [c.to_dict() for c in others[:15]],
    }
    p = path_for(day)
    p.write_text(json.dumps(payload, indent=2, default=str))
    p.with_suffix(".md").write_text(render(payload))
    return p


def load(day: date) -> dict:
    p = path_for(day)
    if not p.exists():
        raise FileNotFoundError(f"no watchlist for {day}; run `patterntrader scan` first")
    return json.loads(p.read_text())


def set_status(day: date, symbols: list[str], status: str) -> list[str]:
    data = load(day)
    changed = []
    for c in data["candidates"]:
        if not symbols or c["symbol"] in symbols:
            c["status"] = status
            changed.append(c["symbol"])
    path_for(day).write_text(json.dumps(data, indent=2, default=str))
    path_for(day).with_suffix(".md").write_text(render(data))
    return changed


def approved(day: date) -> list[SetupContext]:
    data = load(day)
    return [
        SetupContext.from_dict({k: v for k, v in c.items() if k != "status"})
        for c in data["candidates"]
        if c["status"] == "approved"
    ]


def render(data: dict) -> str:
    lines = [
        f"# Watchlist {data['day']}",
        "",
        f"**Market regime:** `{data['market_regime']}`: {'; '.join(data['market_notes'])}",
        f"_Generated {data['generated_at']}_",
        "",
    ]
    if data["market_regime"] == "chaos":
        lines += ["> Market is in **chaos** mode. No trades today. Sitting out is a position.", ""]
    cands = data["candidates"]
    if not cands:
        lines += ["No setups met the bar today. That's the discipline working.", ""]
    else:
        lines += [
            "| # | Symbol | Score | Status | Gap | RVOL | ATR% | Stock regime | News | Strategies |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for i, c in enumerate(cands, 1):
            rvol = f"{c['rvol']:.1f}" if c.get("rvol") is not None else "at open"
            news = {True: "yes", False: "no", None: "?"}[c.get("has_catalyst")]
            lines.append(
                f"| {i} | **{c['symbol']}** | {c['score']:.0f} | {c['status']} | {c['gap_pct']:+.1f}% | {rvol} "
                f"| {c['atr_pct']:.1f} | {c['stock_regime']} | {news} | {', '.join(c['eligible'])} |"
            )
        lines.append("")
        for c in cands:
            lines.append(f"### {c['symbol']}")
            for n in c.get("notes", []):
                lines.append(f"- {n}")
            for h in c.get("headlines", [])[:3]:
                lines.append(f"- 📰 {h}")
            if c.get("rejected"):
                lines.append("- Not allowed today: " + "; ".join(f"{k} ({v})" for k, v in c["rejected"].items()))
            lines.append("")
    others = [o for o in data.get("also_scanned", []) if o["symbol"] not in {c["symbol"] for c in cands}][:8]
    if others:
        lines += ["## Scanned but passed on", ""]
        for o in others:
            why = "; ".join(f"{k}: {v}" for k, v in list(o.get("rejected", {}).items())[:3]) or f"score {o['score']:.0f}"
            lines.append(f"- **{o['symbol']}** (gap {o['gap_pct']:+.1f}%, {o['stock_regime']}): {why}")
    lines += [
        "",
        "Approve with `patterntrader approve SYMBOL ...` (or `--all`). Only approved symbols trade.",
    ]
    return "\n".join(lines) + "\n"
