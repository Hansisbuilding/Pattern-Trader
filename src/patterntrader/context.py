"""Builds a SetupContext for a symbol-day, decides which strategies fit, and scores it.

The same code path is used by the live scanner, the live trader and the backtester,
so a backtest measures exactly the selection rules that will trade.
"""

from __future__ import annotations

import math
from datetime import date, time

import pandas as pd

from patterntrader.indicators import atr
from patterntrader.models import SetupContext
from patterntrader.regime import TRENDS, classify_trend


def opening_rvol(today_volume: float | None, past_volumes: list[float]) -> float | None:
    """Volume in today's opening window vs. the average of the same window on prior days."""
    past = [v for v in past_volumes if v > 0]
    if today_volume is None or len(past) < 3:
        return None
    return float(today_volume / (sum(past) / len(past)))


def opening_volumes(minute_bars: pd.DataFrame, minutes: int) -> dict[date, float]:
    """Sum of volume over the first N regular-session minutes, per day (from 1-minute bars)."""
    if minute_bars.empty:
        return {}
    t = minute_bars.index.time
    window = minute_bars[(t >= time(9, 30)) & (t < time(9, 30 + minutes))]
    counts = window.groupby(window.index.date)["volume"].count()
    sums = window.groupby(window.index.date)["volume"].sum()
    return {d: float(sums[d]) for d in sums.index if counts[d] >= minutes - 1}


def build_context(
    symbol: str,
    day: date,
    daily_prior: pd.DataFrame,
    price: float,
    market_regime: str,
    cfg,
    *,
    open_price: float | None = None,
    today_open_volume: float | None = None,
    past_open_volumes: list[float] | None = None,
    today_premarket: pd.DataFrame | None = None,
    headlines: list[str] | None = None,
    shortable: bool = False,
) -> SetupContext | None:
    """daily_prior must contain only sessions BEFORE `day` (no lookahead).

    open_price / today_open_volume are known once the session has opened; before that the
    gap is measured from `price` (latest premarket trade) and RVOL stays unknown.
    """
    if len(daily_prior) < 60:
        return None
    prev = daily_prior.iloc[-1]
    daily_atr = float(atr(daily_prior).iloc[-1])
    avg_volume = float(daily_prior["volume"].iloc[-20:].mean())
    avg_dollar = float((daily_prior["volume"] * daily_prior["close"]).iloc[-20:].mean())
    stock_regime, strength = classify_trend(daily_prior, cfg.regime.adx_trend_min)

    ref = open_price if open_price else price
    ctx = SetupContext(
        symbol=symbol,
        day=day,
        price=float(price),
        prev_close=float(prev["close"]),
        prev_high=float(prev["high"]),
        prev_low=float(prev["low"]),
        daily_atr=daily_atr,
        atr_pct=daily_atr / float(prev["close"]) * 100,
        avg_volume=avg_volume,
        avg_dollar_volume=avg_dollar,
        gap_pct=(ref / float(prev["close"]) - 1) * 100,
        adx=strength,
        stock_regime=stock_regime,
        market_regime=market_regime,
        rvol=opening_rvol(today_open_volume, past_open_volumes or []),
        has_catalyst=None if headlines is None else bool(headlines),
        headlines=(headlines or [])[:5],
        shortable=shortable,
    )
    if today_premarket is not None and len(today_premarket):
        ctx.premarket_high = float(today_premarket["high"].max())
        ctx.premarket_low = float(today_premarket["low"].min())
    return ctx


def evaluate(ctx: SetupContext, strategies: list, cfg, edge=None) -> SetupContext:
    """Fill in which strategies are allowed for this situation, and a 0-100 score."""
    ctx.eligible, ctx.rejected = [], {}
    if ctx.market_regime == "chaos":
        ctx.rejected = {s.name: "market regime is chaos: sit out" for s in strategies}
        ctx.score = 0.0
        return ctx
    for strat in strategies:
        ok, why = strat.check(ctx)
        if ok and edge is not None:
            ok, why = edge.allows(strat.name)
        if ok:
            ctx.eligible.append(strat.name)
            if why:
                ctx.notes.append(f"{strat.name}: {why}")
        else:
            ctx.rejected[strat.name] = why
    ctx.score = score(ctx, {s.name: s for s in strategies})
    return ctx


def score(ctx: SetupContext, by_name: dict) -> float:
    """How clean is this setup? Weighted toward situation clarity and participation."""
    if not ctx.eligible:
        return 0.0
    kinds = {by_name[n].kind for n in ctx.eligible if n in by_name}
    s = 0.0

    # 1. Situation clarity (25): stock and market in agreement.
    if ctx.stock_regime == ctx.market_regime:
        s += 25
    elif ctx.stock_regime in TRENDS + ("range",):
        s += 15
    else:
        s += 5

    # 2. Participation (25): momentum wants high RVOL, reversion wants normal volume.
    if ctx.rvol is None:
        s += 10
    elif "momentum" in kinds:
        s += min(ctx.rvol / 4, 1) * 25
    else:
        s += 25 if 0.7 <= ctx.rvol <= 2.0 else 10

    # 3. Liquidity (15): $20M/day scores 0, $1B/day scores full.
    if ctx.avg_dollar_volume > 0:
        s += max(0.0, min((math.log10(ctx.avg_dollar_volume) - 7.3) / 1.7, 1.0)) * 15

    # 4. Volatility sweet spot (15): enough range to pay for risk, not so much it's a lottery.
    if 1.5 <= ctx.atr_pct <= 6:
        s += 15
    elif 1.0 <= ctx.atr_pct <= 9:
        s += 7

    # 5. Catalyst fit (10): news fuels momentum; its absence helps fades.
    if ctx.has_catalyst is not None:
        if ("momentum" in kinds and ctx.has_catalyst) or ("reversion" in kinds and not ctx.has_catalyst):
            s += 10

    # 6. More than one valid playbook (10).
    s += min(len(ctx.eligible), 2) * 5
    return round(s, 1)
