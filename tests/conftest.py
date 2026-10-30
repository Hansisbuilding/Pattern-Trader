from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from patterntrader.config import ET, load_config
from patterntrader.indicators import session_vwap
from patterntrader.models import SetupContext

DAY = date(2026, 3, 2)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Tests never call the SEC, Cboe, Fed or BLS."""
    from patterntrader import backtest

    monkeypatch.setattr(backtest, "outside_data", lambda cfg: None)


@pytest.fixture
def cfg():
    c = load_config()
    for params in c.strategies.values():  # test every playbook, whatever config.yaml switches off
        params["enabled"] = True
    return c


def make_bars(closes, volumes=None, start="09:30", day=DAY, wick=0.05, first_open=None, vwap=None):
    """1-minute bars: open = previous close, high/low = body +/- wick."""
    volumes = volumes or [1000] * len(closes)
    hh, mm = map(int, start.split(":"))
    t0 = datetime(day.year, day.month, day.day, hh, mm, tzinfo=ET)
    rows, prev = [], first_open if first_open is not None else closes[0]
    for i, (c, v) in enumerate(zip(closes, volumes)):
        o = prev
        rows.append({"open": o, "high": max(o, c) + wick, "low": min(o, c) - wick, "close": c, "volume": v})
        prev = c
    df = pd.DataFrame(rows, index=pd.DatetimeIndex([t0 + timedelta(minutes=i) for i in range(len(rows))]))
    df["vwap"] = session_vwap(df) if vwap is None else vwap
    return df


def make_ctx(**kw):
    base = dict(
        symbol="TEST", day=DAY, price=100.0, prev_close=100.0, prev_high=101.0, prev_low=99.0,
        daily_atr=4.0, atr_pct=4.0, avg_volume=5e6, avg_dollar_volume=5e8, gap_pct=3.0, adx=25.0,
        stock_regime="trend_up", market_regime="trend_up", rvol=3.0, has_catalyst=True,
    )
    base.update(kw)
    return SetupContext(**base)


def make_daily(closes, start=date(2025, 1, 1), spread=1.0):
    idx = pd.DatetimeIndex([pd.Timestamp(start, tz=ET) + pd.Timedelta(days=i) for i in range(len(closes))])
    s = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": s.shift().fillna(s.iloc[0]), "high": s + spread, "low": s - spread, "close": s, "volume": 1e6}
    )
