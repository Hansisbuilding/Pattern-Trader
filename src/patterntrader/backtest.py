"""Event-driven intraday backtester.

Each simulated day replays the live process: judge the market regime from SPY, pick the
watchlist with the scanner's rules using only data known by 09:35, then step through 1-minute
bars running the same strategy and risk code as the live trader.

Fill assumptions are deliberately conservative: market entries at the NEXT bar's open plus
slippage, stops filled at the stop or worse on a gap, and if a bar touches both stop and
target the stop is assumed to have hit first.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from patterntrader.config import ET, ROOT, parse_time
from patterntrader.context import add_outside_data, build_context, evaluate, opening_volumes
from patterntrader.data import premarket_session, regular_session
from patterntrader.indicators import session_vwap
from patterntrader.models import SetupContext, Signal
from patterntrader.regime import market_regime
from patterntrader.risk import RiskManager
from patterntrader.sentiment import setup_size_factor
from patterntrader.scanner import coarse_select, pick_watchlist, prior_daily
from patterntrader.strategies import build_strategies

log = logging.getLogger(__name__)


@dataclass
class Trade:
    day: date
    symbol: str
    strategy: str
    side: str
    entry_time: datetime
    entry: float
    exit_time: datetime
    exit: float
    stop: float
    target: float
    qty: int
    pnl: float
    trade_return_pct: float   # P&L as % of the capital put into this position
    portfolio_pct: float      # P&L as % of account equity at the time of the trade
    equity_after: float
    r_multiple: float
    exit_reason: str
    market_regime: str
    stock_regime: str
    reason: str
    size_factor: float = 1.0


def simulate_day(
    candidates: list[tuple[SetupContext, pd.DataFrame]], strategies: dict, cfg, equity: float,
    size_factor: float = 1.0,
) -> tuple[list[Trade], float]:
    """candidates: (context, regular-session 1-minute bars with 'vwap') for one day."""
    slip = cfg.backtest.slippage_bps / 10_000
    flatten = parse_time(cfg.session.flatten_at)
    risk = RiskManager(cfg, equity, size_factor)
    index = {ctx.symbol: {t: i for i, t in enumerate(b.index)} for ctx, b in candidates}
    positions: dict[str, dict] = {}
    pending: dict[str, Signal] = {}
    fired: set[tuple[str, str]] = set()
    trades: list[Trade] = []
    times = sorted(set().union(*(set(b.index) for _, b in candidates))) if candidates else []

    def close(ctx, pos, t, px, why):
        nonlocal equity
        sig, d = pos["sig"], pos["sig"].direction
        pnl = (px - pos["entry"]) * pos["qty"] * d
        equity_before = equity
        equity += pnl
        risk.on_exit(ctx.symbol, pnl)
        risk.update_equity(equity)
        trades.append(
            Trade(
                ctx.day, ctx.symbol, sig.strategy, sig.side, pos["time"], round(pos["entry"], 4), t,
                round(px, 4), sig.stop, sig.target, pos["qty"], round(pnl, 2),
                round(pnl / (pos["entry"] * pos["qty"]) * 100, 3),
                round(pnl / equity_before * 100, 4),
                round(equity, 2),
                round(pnl / (pos["qty"] * abs(pos["entry"] - sig.stop)), 3), why,
                ctx.market_regime, ctx.stock_regime, sig.reason, risk.size_factor,
            )
        )
        del positions[ctx.symbol]

    for t in times:
        for ctx, bars in candidates:
            sym = ctx.symbol
            i = index[sym].get(t)
            if i is None:
                continue
            o, h, l, c = (bars[k].iat[i] for k in ("open", "high", "low", "close"))

            if sym in pending:  # fill last bar's signal at this bar's open
                sig = pending.pop(sym)
                d = sig.direction
                fill = o * (1 + d * slip)
                open_notional = sum(p["entry"] * p["qty"] for p in positions.values())
                qty = risk.size(
                    replace(sig, entry=fill), equity,
                    open_notional=open_notional, avg_dollar_volume=ctx.avg_dollar_volume,
                    setup_factor=setup_size_factor(ctx, cfg),
                )
                if (fill - sig.stop) * d <= 0 or (sig.target - fill) * d <= 0 or qty == 0:
                    risk.on_cancel(sym)
                else:
                    positions[sym] = {"sig": sig, "entry": fill, "qty": qty, "time": t}

            if sym in positions:
                pos = positions[sym]
                sig, d = pos["sig"], pos["sig"].direction
                if t.time() >= flatten or risk.flatten_now:
                    close(ctx, pos, t, c * (1 - d * slip), "flatten" if t.time() >= flatten else "daily_loss")
                elif (l <= sig.stop) if d == 1 else (h >= sig.stop):
                    px = min(o, sig.stop) if d == 1 else max(o, sig.stop)
                    close(ctx, pos, t, px * (1 - d * slip), "stop")
                elif (h > sig.target) if d == 1 else (l < sig.target):
                    px = max(o, sig.target) if d == 1 else min(o, sig.target)
                    close(ctx, pos, t, px, "target")

            if sym in positions or sym in pending or t.time() >= flatten or risk.halted:
                continue
            hist = bars.iloc[: i + 1]
            for name in ctx.eligible:
                if (sym, name) in fired:
                    continue
                sig = strategies[name].signal(ctx, hist)
                if sig is None:
                    continue
                fired.add((sym, name))  # one shot per setup: no chasing a second entry
                ok, _ = risk.approve(sig, t)
                if ok:
                    pending[sym] = sig
                    risk.on_entry(sig)
                    break

    for ctx, bars in candidates:  # anything still open closes at the last bar
        if ctx.symbol in positions:
            pos = positions[ctx.symbol]
            close(ctx, pos, bars.index[-1], bars["close"].iat[-1], "flatten")
    return trades, equity


def build_day_candidates(day, rows, daily, minutes, regime, cfg, strategies, md=None, ext=None):
    """Recreate the 09:35 watchlist for `day` using only information available then."""
    rv = cfg.session.rvol_minutes
    contexts = []
    for r in rows:
        sym = r["symbol"]
        m = minutes.get(sym, pd.DataFrame())
        if m.empty:
            continue
        vols = opening_volumes(m, rv)
        if day not in vols:
            continue
        past = [v for d, v in sorted(vols.items()) if d < day][-10:]
        headlines = None
        if md is not None and cfg.backtest.use_news and abs(r["gap_pct"]) >= cfg.scan.min_abs_gap_pct:
            open_dt = datetime.combine(day, time(9, 30), ET)
            headlines = md.headlines(sym, open_dt - timedelta(hours=cfg.scan.news_lookback_hours), open_dt)
        ctx = build_context(
            sym, day, prior_daily(daily[sym], day), r["price"], regime, cfg,
            open_price=r["price"], today_open_volume=vols[day], past_open_volumes=past,
            today_premarket=premarket_session(m, day), headlines=headlines, shortable=True,
        )
        if ctx:
            if ext is not None:
                add_outside_data(ctx, ext, cfg.scan.news_lookback_hours)
            contexts.append(evaluate(ctx, list(strategies.values()), cfg))
    contexts.sort(key=lambda c: -c.score)
    out = []
    for ctx in pick_watchlist(contexts, cfg):
        bars = regular_session(minutes[ctx.symbol], day).copy()
        if len(bars) < 30:
            continue
        bars["vwap"] = session_vwap(bars)
        out.append((ctx, bars))
    return out


def outside_data(cfg):
    """External data source, or None if SEC_USER_AGENT isn't configured."""
    from patterntrader.external import External, _user_agent

    try:
        _user_agent()
    except RuntimeError as exc:
        log.warning("%s: skipping SEC filings, VIX and macro calendar", exc)
        return None
    ext = External(cfg.data.cache_dir)
    ext.earnings_calls_left = 0  # read cached earnings only; scripts/fetch_earnings.py spends the daily quota
    return ext


def run_backtest(cfg, md, symbols: list[str], start: date, end: date) -> dict:
    strategies = {s.name: s for s in build_strategies(cfg)}
    spy = cfg.regime.market_symbol
    feed = cfg.data.backtest_feed
    log.info("loading daily bars for %d symbols", len(symbols))
    daily = md.daily(sorted(set(symbols) | {spy}), start - timedelta(days=400), end, feed)
    if spy not in daily:
        raise RuntimeError("no SPY data returned: check API keys and data permissions")
    spy_d = daily[spy]
    days = [d for d in sorted(set(spy_d.index.date)) if start <= d <= end]
    equity = float(cfg.backtest.starting_equity)
    trades: list[Trade] = []
    days_by_regime: dict[str, int] = {}
    day_regimes: dict[date, str] = {}
    ext = outside_data(cfg)
    sized_days: dict[float, int] = {}

    for n, day in enumerate(days, 1):
        spy_prior, spy_today = prior_daily(spy_d, day), spy_d[spy_d.index.date == day]
        spy_gap = (spy_today["open"].iloc[0] / spy_prior["close"].iloc[-1] - 1) * 100
        regime, _ = market_regime(spy_prior, spy_gap, cfg.regime)
        days_by_regime[regime] = days_by_regime.get(regime, 0) + 1
        day_regimes[day] = regime
        if regime == "chaos" and not cfg.regime.get("trade_chaos", False):
            continue
        rows = []
        for sym in symbols:
            d = daily.get(sym)
            if d is None:
                continue
            prior, today = prior_daily(d, day), d[d.index.date == day]
            if len(prior) < 60 or today.empty:
                continue
            open_px, prev_close = float(today["open"].iloc[0]), float(prior["close"].iloc[-1])
            rows.append({
                "symbol": sym, "price": open_px, "prev_close": prev_close,
                "gap_pct": (open_px / prev_close - 1) * 100,
                "avg_dollar_volume": float((prior["close"] * prior["volume"]).iloc[-20:].mean()),
            })
        rows = coarse_select(rows, cfg)
        lookback_start = [d for d in sorted(set(spy_prior.index.date))][-12]
        minutes = md.minutes([r["symbol"] for r in rows], lookback_start, day, feed)
        cands = build_day_candidates(day, rows, daily, minutes, regime, cfg, strategies, md, ext)
        factor = 1.0
        if ext is not None and cfg.get("sentiment", {}).get("enabled"):
            from patterntrader.sentiment import day_size_factor, market_mood

            mood = market_mood(ext, spy_d, day)["mood"]
            factor, _ = day_size_factor(mood, ext.vix_context(day).get("vix_term"), ext.events_on(day), cfg)
            sized_days[factor] = sized_days.get(factor, 0) + 1
        day_trades, equity = simulate_day(cands, strategies, cfg, equity, factor)
        trades += day_trades
        if n % 20 == 0 or day == days[-1]:
            log.info("%s  day %d/%d  trades %d  equity %.0f", day, n, len(days), len(trades), equity)

    spy_close = pd.Series(spy_d["close"].values, index=spy_d.index.date)
    summary = summarize(trades, cfg, start, end, days_by_regime, day_regimes, spy_close)
    summary["sentiment"] = {"enabled": bool(cfg.get("sentiment", {}).get("enabled")) and ext is not None,
                            "days_by_size_factor": {str(k): v for k, v in sorted(sized_days.items())}}
    return summary


def _stats(df: pd.DataFrame, start_eq: float) -> dict:
    """Aggregate stats. *_pct fields: trade = % of position capital, portfolio = % of account."""
    if df.empty:
        return {
            "trades": 0, "win_rate": 0.0, "expectancy_r": 0.0, "profit_factor": 0.0, "total_pnl": 0.0,
            "portfolio_pct": 0.0, "avg_trade_pct": 0.0, "best_trade_pct": 0.0, "worst_trade_pct": 0.0,
        }
    wins, losses = df[df.pnl > 0], df[df.pnl <= 0]
    gross_loss = -losses.pnl.sum()
    return {
        "trades": int(len(df)),
        "win_rate": round(float(len(wins) / len(df)), 3),
        "expectancy_r": round(float(df.r_multiple.mean()), 3),
        "avg_win_r": round(float(wins.r_multiple.mean()), 3) if len(wins) else 0.0,
        "avg_loss_r": round(float(losses.r_multiple.mean()), 3) if len(losses) else 0.0,
        "profit_factor": round(float(wins.pnl.sum() / gross_loss), 3) if gross_loss > 0 else 99.0,
        "total_pnl": round(float(df.pnl.sum()), 2),
        "portfolio_pct": round(float(df.pnl.sum() / start_eq * 100), 3),
        "avg_trade_pct": round(float(df.trade_return_pct.mean()), 3),
        "avg_win_trade_pct": round(float(wins.trade_return_pct.mean()), 3) if len(wins) else 0.0,
        "avg_loss_trade_pct": round(float(losses.trade_return_pct.mean()), 3) if len(losses) else 0.0,
        "best_trade_pct": round(float(df.trade_return_pct.max()), 3),
        "worst_trade_pct": round(float(df.trade_return_pct.min()), 3),
    }


def market_label(spy_return_pct: float) -> str:
    if spy_return_pct >= 15:
        return "Strong bull"
    if spy_return_pct >= 5:
        return "Bull"
    if spy_return_pct > -5:
        return "Flat / choppy"
    if spy_return_pct > -15:
        return "Bear"
    return "Strong bear"


def year_windows(start: date, end: date) -> list[tuple[date, date]]:
    out, cur = [], start
    while cur <= end:
        nxt = cur.replace(year=cur.year + 1)
        out.append((cur, min(nxt - timedelta(days=1), end)))
        cur = nxt
    return out


def by_year(df, start, end, start_eq, day_regimes, spy_close) -> list[dict]:
    years = []
    equity = start_eq
    for y0, y1 in year_windows(start, end):
        closes = spy_close[(spy_close.index >= y0) & (spy_close.index <= y1)]
        prior = spy_close[spy_close.index < y0]
        base = prior.iloc[-1] if len(prior) else closes.iloc[0]
        spy_ret = float(closes.iloc[-1] / base - 1) * 100 if len(closes) else 0.0
        regimes: dict[str, int] = {}
        for d, r in day_regimes.items():
            if y0 <= d <= y1:
                regimes[r] = regimes.get(r, 0) + 1
        g = df[(df.day >= y0) & (df.day <= y1)] if not df.empty else df
        year = {
            "period": [y0.isoformat(), y1.isoformat()],
            "spy_return_pct": round(spy_ret, 2),
            "market": market_label(spy_ret),
            "regime_days": dict(sorted(regimes.items(), key=lambda kv: -kv[1])),
            "start_equity": round(equity, 2),
            "overall": _stats(g, equity),
            "strategies": {n: _stats(sg, equity) for n, sg in g.groupby("strategy")} if len(g) else {},
            "strategy_regime": {},
        }
        if len(g):
            for (n, r), sg in g.groupby(["strategy", "market_regime"]):
                year["strategy_regime"].setdefault(n, {})[r] = _stats(sg, equity)
        equity += float(g.pnl.sum()) if len(g) else 0.0
        years.append(year)
    return years


def summarize(
    trades: list[Trade], cfg, start: date, end: date, days_by_regime: dict,
    day_regimes: dict | None = None, spy_close: pd.Series | None = None,
) -> dict:
    df = pd.DataFrame([asdict(t) for t in trades])
    start_eq = float(cfg.backtest.starting_equity)
    summary = {
        "period": [start.isoformat(), end.isoformat()],
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "starting_equity": start_eq,
        "days_by_market_regime": days_by_regime,
        "overall": _stats(df, start_eq),
        "strategies": {},
        "by_market_regime": {},
        "by_day": [],
        "trades": [],
        "by_year": [],
        "strategy_regime": {},
    }
    if not df.empty:
        curve = start_eq + df.pnl.cumsum()
        peak = curve.cummax().clip(lower=start_eq)
        o = summary["overall"]
        o["ending_equity"] = round(float(curve.iloc[-1]), 2)
        o["max_drawdown_pct"] = round(float(((curve - peak) / peak).min() * 100), 2)
        o["return_pct"] = round(float(df.pnl.sum() / start_eq * 100), 3)
        mid = start + (end - start) / 2
        for name, g in df.groupby("strategy"):
            st = _stats(g, start_eq)
            # Robustness check: did the edge hold in both halves of the test period?
            st["expectancy_r_first_half"] = _stats(g[g.day <= mid], start_eq)["expectancy_r"]
            st["expectancy_r_second_half"] = _stats(g[g.day > mid], start_eq)["expectancy_r"]
            summary["strategies"][name] = st
        for regime, g in df.groupby("market_regime"):
            summary["by_market_regime"][regime] = _stats(g, start_eq)
        for (name, regime), g in df.groupby(["strategy", "market_regime"]):
            summary["strategy_regime"].setdefault(name, {})[regime] = _stats(g, start_eq)
    if day_regimes is not None and spy_close is not None:
        summary["by_year"] = by_year(df, start, end, start_eq, day_regimes, spy_close)
    if not df.empty:
        for day, g in df.groupby("day"):
            day_start = float(g.equity_after.iloc[0] - g.pnl.iloc[0])
            summary["by_day"].append({
                "day": day.isoformat(), "trades": int(len(g)), "pnl": round(float(g.pnl.sum()), 2),
                "portfolio_pct": round(float(g.pnl.sum() / day_start * 100), 3),
                "equity_after": round(float(g.equity_after.iloc[-1]), 2),
            })
        summary["trades"] = [
            {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()}
            for r in df[["day", "symbol", "strategy", "side", "entry_time", "entry", "exit", "qty", "pnl",
                         "trade_return_pct", "portfolio_pct", "r_multiple", "exit_reason"]].to_dict("records")
        ]
    summary["trades_df"] = df
    return summary


def write_results(summary: dict, save_stats: bool = True, tag: str = "") -> Path:
    df = summary.pop("trades_df")
    out = ROOT / "results" / f"backtest_{summary['period'][0]}_{summary['period'][1]}{'_' + tag if tag else ''}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "trades.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "summary.md").write_text(render_summary(summary))
    if save_stats:
        (ROOT / "results" / "strategy_stats.json").write_text(json.dumps(summary, indent=2, default=str))
    return out


def _money(x: float) -> str:
    return f"-${-x:,.2f}" if x < 0 else f"+${x:,.2f}"


def render_summary(s: dict) -> str:
    o = s["overall"]
    lines = [
        f"# Backtest {s['period'][0]} → {s['period'][1]}",
        "",
        f"Starting equity **${s['starting_equity']:,.0f}** → ending **${o.get('ending_equity', s['starting_equity']):,.2f}**",
        "",
        f"- **Total P&L:** {_money(o['total_pnl'])} = **{o['portfolio_pct']:+.2f}% of portfolio**",
        f"- **Trades:** {o['trades']} · win rate {o['win_rate']:.0%} · expectancy {o['expectancy_r']:+.2f}R "
        f"· profit factor {o['profit_factor']:.2f}",
        f"- **Per trade (% of position):** average {o['avg_trade_pct']:+.2f}% · avg win "
        f"{o.get('avg_win_trade_pct', 0):+.2f}% · avg loss {o.get('avg_loss_trade_pct', 0):+.2f}% "
        f"· best {o['best_trade_pct']:+.2f}% · worst {o['worst_trade_pct']:+.2f}%",
        f"- **Max drawdown:** {o.get('max_drawdown_pct', 0):.2f}% of portfolio",
        "",
        "Market days by regime: " + ", ".join(f"{k} {v}" for k, v in s["days_by_market_regime"].items()),
        "",
    ]
    sent = s.get("sentiment") or {}
    if sent.get("enabled"):
        sized = sent.get("days_by_size_factor", {})
        lines += ["Sentiment **on**. Days by position size: " + ", ".join(
            f"{'sat out' if float(k) == 0 else 'normal' if float(k) == 1 else f'{float(k):.0%} size'} {v}"
            for k, v in sized.items()), ""]
    else:
        lines += ["Sentiment **off** (baseline rules only).", ""]
    lines += [
        "## By strategy",
        "",
        "| Strategy | Trades | Win % | Expectancy | Avg trade % | P&L | % of portfolio | 1st half | 2nd half |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, st in sorted(s["strategies"].items()):
        lines.append(
            f"| {name} | {st['trades']} | {st['win_rate']:.0%} | {st['expectancy_r']:+.2f}R "
            f"| {st['avg_trade_pct']:+.2f}% | {_money(st['total_pnl'])} | {st['portfolio_pct']:+.2f}% "
            f"| {st['expectancy_r_first_half']:+.2f}R | {st['expectancy_r_second_half']:+.2f}R |"
        )
    lines += [
        "", "## By market regime", "",
        "| Regime | Trades | Win % | Expectancy | Avg trade % | P&L | % of portfolio |",
        "|---|---|---|---|---|---|---|",
    ]
    for regime, st in sorted(s["by_market_regime"].items()):
        lines.append(
            f"| {regime} | {st['trades']} | {st['win_rate']:.0%} | {st['expectancy_r']:+.2f}R "
            f"| {st['avg_trade_pct']:+.2f}% | {_money(st['total_pnl'])} | {st['portfolio_pct']:+.2f}% |"
        )
    regimes = ["trend_up", "range", "mixed", "trend_down"]
    if s.get("strategy_regime"):
        lines += [
            "", "## Strategy × market regime (all years)", "",
            "Each cell: trades · expectancy · P&L (% of starting portfolio)", "",
            "| Strategy | " + " | ".join(regimes) + " |", "|---|" + "---|" * len(regimes),
        ]
        for name, row in sorted(s["strategy_regime"].items()):
            cells = [
                f"{c['trades']} · {c['expectancy_r']:+.2f}R · {_money(c['total_pnl'])} ({c['portfolio_pct']:+.2f}%)"
                if (c := row.get(r)) else "–"
                for r in regimes
            ]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for y in s.get("by_year", []):
        o = y["overall"]
        days = ", ".join(f"{k} {v}" for k, v in y["regime_days"].items())
        lines += [
            "", f"## {y['period'][0]} → {y['period'][1]}: {y['market']} (SPY {y['spy_return_pct']:+.1f}%)", "",
            f"Market days: {days}", "",
            f"Bot: {o['trades']} trades · win {o['win_rate']:.0%} · {o['expectancy_r']:+.2f}R · "
            f"{_money(o['total_pnl'])} = **{o['portfolio_pct']:+.2f}%** of ${y['start_equity']:,.0f}", "",
        ]
        if y["strategies"]:
            lines += [
                "| Strategy | Trades | Win % | Expectancy | Avg trade % | P&L | % of portfolio | By regime |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for name, st in sorted(y["strategies"].items()):
                per = "; ".join(
                    f"{r} {c['trades']}×{c['expectancy_r']:+.2f}R"
                    for r, c in y["strategy_regime"].get(name, {}).items()
                )
                lines.append(
                    f"| {name} | {st['trades']} | {st['win_rate']:.0%} | {st['expectancy_r']:+.2f}R "
                    f"| {st['avg_trade_pct']:+.2f}% | {_money(st['total_pnl'])} | {st['portfolio_pct']:+.2f}% | {per} |"
                )
    if s["by_day"]:
        lines += ["", "## By day", "", "| Day | Trades | P&L | % of portfolio | Equity after |", "|---|---|---|---|---|"]
        for d in s["by_day"]:
            lines.append(
                f"| {d['day']} | {d['trades']} | {_money(d['pnl'])} | {d['portfolio_pct']:+.2f}% | ${d['equity_after']:,.2f} |"
            )
    if s["trades"]:
        lines += [
            "", "## Every trade", "",
            "| Day | Symbol | Strategy | Side | Entry | Exit | Qty | P&L | Trade % | Portfolio % | R | Exit |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for t in s["trades"]:
            lines.append(
                f"| {t['day']} | {t['symbol']} | {t['strategy']} | {t['side']} | {t['entry']:.2f} | {t['exit']:.2f} "
                f"| {t['qty']} | {_money(t['pnl'])} | {t['trade_return_pct']:+.2f}% | {t['portfolio_pct']:+.3f}% "
                f"| {t['r_multiple']:+.2f} | {t['exit_reason']} |"
            )
    return "\n".join(lines) + "\n"
