"""Technical indicators on OHLCV DataFrames (columns: open, high, low, close, volume)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift()
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder(true_range(df), n)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = wilder(delta.clip(lower=0), n)
    down = wilder((-delta).clip(lower=0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = up / down
    out = 100 - 100 / (1 + rs)
    return out.where(down != 0, 100.0)


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(s, n)
    sd = s.rolling(n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = wilder(true_range(df), n)
    plus_di = 100 * wilder(plus_dm, n) / tr
    minus_di = 100 * wilder(minus_dm, n) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return wilder(dx.fillna(0), n)


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP for a single trading session's bars."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, np.nan)
    cum_vol = vol.cumsum().ffill()
    return ((typical * vol).cumsum().ffill() / cum_vol).bfill()


def resample_complete(bars: pd.DataFrame, minutes: int, session_open: pd.Timestamp) -> pd.DataFrame:
    """Resample 1-minute session bars to N-minute bars, keeping only completed bars."""
    if bars.empty:
        return bars
    agg = bars.resample(f"{minutes}min", label="left", closed="left", origin=session_open).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    agg = agg.dropna(subset=["close"])
    last_minute = bars.index[-1]
    if len(agg) and agg.index[-1] + pd.Timedelta(minutes=minutes - 1) > last_minute:
        agg = agg.iloc[:-1]
    return agg
