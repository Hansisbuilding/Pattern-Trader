"""Regime classification: is the market / stock trending, ranging, or too chaotic to trade?"""

from __future__ import annotations

import pandas as pd

from patterntrader.indicators import adx, atr, ema, sma

TRENDS = ("trend_up", "trend_down")


def classify_trend(daily: pd.DataFrame, adx_min: float = 20) -> tuple[str, float]:
    """Classify daily bars (strictly before the trading day) into a regime.

    trend_up / trend_down: price, 20 EMA and 50 SMA stacked in order AND ADX >= adx_min.
    range: ADX below adx_min (no directional pressure).
    mixed: ADX is elevated but the averages disagree, i.e. a trend that is turning.
    """
    if len(daily) < 60:
        return "mixed", float("nan")
    close = daily["close"]
    last, e20, s50 = close.iloc[-1], ema(close, 20).iloc[-1], sma(close, 50).iloc[-1]
    strength = float(adx(daily).iloc[-1])
    if strength < adx_min:
        return "range", strength
    if last > e20 > s50:
        return "trend_up", strength
    if last < e20 < s50:
        return "trend_down", strength
    return "mixed", strength


def market_regime(spy_daily: pd.DataFrame, spy_gap_pct: float, cfg) -> tuple[str, list[str]]:
    """Market-wide regime from SPY. 'chaos' means no new trades at all today."""
    notes: list[str] = []
    if abs(spy_gap_pct) >= cfg.chaos_gap_pct:
        notes.append(f"SPY gapped {spy_gap_pct:+.2f}% (>= {cfg.chaos_gap_pct}%)")
        return "chaos", notes
    if len(spy_daily) >= 60:
        atr_pct = (atr(spy_daily) / spy_daily["close"]).dropna().iloc[-250:]
        pctile = float((atr_pct <= atr_pct.iloc[-1]).mean())
        if pctile >= cfg.chaos_atr_pctile:
            notes.append(f"SPY volatility at {pctile:.0%} percentile of the last year")
            return "chaos", notes
    regime, strength = classify_trend(spy_daily, cfg.adx_trend_min)
    notes.append(f"SPY {regime} (ADX {strength:.0f})")
    return regime, notes
