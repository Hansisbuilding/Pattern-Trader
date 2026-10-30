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
from patterntrader.context import build_context, evaluate, opening_volumes
from patterntrader.data import premarket_session, regular_session
from patterntrader.indicators import session_vwap
from patterntrader.models import SetupContext, Signal
from patterntrader.regime import market_regime
from patterntrader.risk import RiskManager
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
    r_multiple: float
    exit_reason: str
    market_regime: str
    stock_regime: str
    reason: str


def simulate_day(
    candidates: list[tuple[SetupContext, pd.DataFrame]], strategies: dict, cfg, equity: float
) -> tuple[list[Trade], float]:
    """candidates: (context, regular-session 1-minute bars with 'vwap') for one day."""
    slip = cfg.backtest.slippage_bps / 10_000
    flatten = parse_time(cfg.session.flatten_at)
    risk = RiskManager(cfg, equity)
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
        equity += pnl
        risk.on_exit(ctx.symbol, pnl)
        risk.update_equity(equity)
        trades.append(
            Trade(
                ctx.day, ctx.symbol, sig.strategy, sig.side, pos["time"], round(pos["entry"], 4), t,
                round(px, 4), sig.stop, sig.target, pos["qty"], round(pnl, 2),
                round(pnl / (pos["qty"] * abs(pos["entry"] - sig.stop)), 3), why,
                ctx.market_regime, ctx.stock_regime, sig.reason,
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
                qty = risk.size(replace(sig, entry=fill), equity)
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


def build_day_candidates(day, rows, daily, minutes, regime, cfg, strategies, md=None):
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

    for n, day in enumerate(days, 1):
        spy_prior, spy_today = prior_daily(spy_d, day), spy_d[spy_d.index.date == day]
        spy_gap = (spy_today["open"].iloc[0] / spy_prior["close"].iloc[-1] - 1) * 100
        regime, _ = market_regime(spy_prior, spy_gap, cfg.regime)
        days_by_regime[regime] = days_by_regime.get(regime, 0) + 1
        if regime == "chaos":
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
        cands = build_day_candidates(day, rows, daily, minutes, regime, cfg, strategies, md)
        day_trades, equity = simulate_day(cands, strategies, cfg, equity)
        trades += day_trades
        if n % 20 == 0 or day == days[-1]:
            log.info("%s  day %d/%d  trades %d  equity %.0f", day, n, len(days), len(trades), equity)

    return summarize(trades, cfg, start, end, days_by_regime)


def _stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0, "win_rate": 0.0, "expectancy_r": 0.0, "profit_factor": 0.0, "total_pnl": 0.0}
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
    }


def summarize(trades: list[Trade], cfg, start: date, end: date, days_by_regime: dict) -> dict:
    df = pd.DataFrame([asdict(t) for t in trades])
    start_eq = float(cfg.backtest.starting_equity)
    summary = {
        "period": [start.isoformat(), end.isoformat()],
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "days_by_market_regime": days_by_regime,
        "overall": _stats(df),
        "strategies": {},
        "by_market_regime": {},
    }
    if not df.empty:
        curve = start_eq + df.pnl.cumsum()
        peak = curve.cummax().clip(lower=start_eq)
        summary["overall"]["max_drawdown_pct"] = round(float(((curve - peak) / peak).min() * 100), 2)
        summary["overall"]["return_pct"] = round(float(df.pnl.sum() / start_eq * 100), 2)
        mid = start + (end - start) / 2
        for name, g in df.groupby("strategy"):
            st = _stats(g)
            # Robustness check: did the edge hold in both halves of the test period?
            st["expectancy_r_first_half"] = _stats(g[g.day <= mid])["expectancy_r"]
            st["expectancy_r_second_half"] = _stats(g[g.day > mid])["expectancy_r"]
            summary["strategies"][name] = st
        for regime, g in df.groupby("market_regime"):
            summary["by_market_regime"][regime] = _stats(g)
    summary["trades_df"] = df
    return summary


def write_results(summary: dict, save_stats: bool = True) -> Path:
    df = summary.pop("trades_df")
    out = ROOT / "results" / f"backtest_{summary['period'][0]}_{summary['period'][1]}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "trades.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(render_summary(summary))
    if save_stats:
        (ROOT / "results" / "strategy_stats.json").write_text(json.dumps(summary, indent=2))
    return out


def render_summary(s: dict) -> str:
    o = s["overall"]
    lines = [
        f"# Backtest {s['period'][0]} → {s['period'][1]}",
        "",
        f"Trades **{o['trades']}** · win rate **{o['win_rate']:.0%}** · expectancy **{o['expectancy_r']:+.2f}R** "
        f"· profit factor **{o['profit_factor']:.2f}** · return **{o.get('return_pct', 0):+.2f}%** "
        f"· max drawdown **{o.get('max_drawdown_pct', 0):.2f}%**",
        "",
        "Market days by regime: " + ", ".join(f"{k} {v}" for k, v in s["days_by_market_regime"].items()),
        "",
        "## By strategy",
        "",
        "| Strategy | Trades | Win % | Expectancy | 1st half | 2nd half | Profit factor | P&L |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, st in sorted(s["strategies"].items()):
        lines.append(
            f"| {name} | {st['trades']} | {st['win_rate']:.0%} | {st['expectancy_r']:+.2f}R "
            f"| {st['expectancy_r_first_half']:+.2f}R | {st['expectancy_r_second_half']:+.2f}R "
            f"| {st['profit_factor']:.2f} | ${st['total_pnl']:,.0f} |"
        )
    lines += ["", "## By market regime", "", "| Regime | Trades | Win % | Expectancy | P&L |", "|---|---|---|---|---|"]
    for regime, st in sorted(s["by_market_regime"].items()):
        lines.append(
            f"| {regime} | {st['trades']} | {st['win_rate']:.0%} | {st['expectancy_r']:+.2f}R | ${st['total_pnl']:,.0f} |"
        )
    return "\n".join(lines) + "\n"
