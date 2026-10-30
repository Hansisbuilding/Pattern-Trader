"""Each playbook fires on its textbook pattern and stays quiet when the situation is wrong."""

import pytest
from conftest import make_bars, make_ctx

from patterntrader.strategies import REGISTRY


def strat(cfg, name):
    return REGISTRY[name](cfg.strategies[name], cfg)


def orb_bars(breakout_volume=5000):
    closes = [100.05, 100.95] * 8  # 16-bar range 100.00-101.00
    closes += [100.5, 100.6, 100.4, 100.7]
    closes += [101.5]  # 09:50 break
    vols = [1000] * (len(closes) - 1) + [breakout_volume]
    return make_bars(closes, vols)


def test_orb_long_breakout(cfg):
    s = strat(cfg, "orb").signal(make_ctx(), orb_bars())
    assert s and s.side == "long" and s.entry == 101.5
    assert s.stop == pytest.approx(99.99) and s.reward_risk >= 2.0 - 0.02


def test_orb_needs_volume(cfg):
    assert strat(cfg, "orb").signal(make_ctx(), orb_bars(breakout_volume=1000)) is None


def test_orb_not_in_down_market(cfg):
    assert strat(cfg, "orb").signal(make_ctx(market_regime="trend_down"), orb_bars()) is None


def test_orb_stop_too_wide_rejected(cfg):
    # Daily ATR of 2 allows at most a $1.00 stop; range low and midpoint are both farther.
    assert strat(cfg, "orb").signal(make_ctx(daily_atr=1.0), orb_bars()) is None


def test_gap_and_go(cfg):
    closes = [106.0, 106.2, 106.1, 106.3, 106.2, 106.4, 106.3, 106.8]
    vols = [3000] * 7 + [9000]
    ctx = make_ctx(gap_pct=6.0, premarket_high=106.5)
    s = strat(cfg, "gap_and_go").signal(ctx, make_bars(closes, vols, first_open=106.0))
    assert s and s.side == "long" and "new high" in s.reason


def test_gap_and_go_waits_for_premarket_high(cfg):
    closes = [106.0, 106.2, 106.1, 106.3, 106.2, 106.4, 106.3, 106.45]
    vols = [3000] * 7 + [9000]
    ctx = make_ctx(gap_pct=6.0, premarket_high=107.0)
    assert strat(cfg, "gap_and_go").signal(ctx, make_bars(closes, vols)) is None


def test_vwap_pullback(cfg):
    closes = [101.6 + 0.05 * i for i in range(25)]  # trending above VWAP
    closes += [101.6, 102.9]  # dip to VWAP, then reclaim above the dip bar's high
    bars = make_bars(closes, wick=0.05, vwap=101.5)
    bars.iloc[-2, bars.columns.get_loc("low")] = 101.5
    s = strat(cfg, "vwap_pullback").signal(make_ctx(), bars)
    assert s and s.side == "long"


def test_vwap_pullback_needs_trending_stock(cfg):
    closes = [101.6 + 0.05 * i for i in range(25)] + [101.6, 102.9]
    bars = make_bars(closes, vwap=101.5)
    assert strat(cfg, "vwap_pullback").signal(make_ctx(stock_regime="range"), bars) is None


def test_bull_flag(cfg):
    closes = [100.0] * 20                                       # base
    closes += [100.25 * 1 + 0.25 * i for i in range(8)]         # pole to ~102
    closes += [101.9, 101.85, 101.8, 101.7, 101.65]             # quiet flag
    closes += [102.3]                                           # breakout
    vols = [1000] * 20 + [4000] * 8 + [800] * 5 + [2500]
    s = strat(cfg, "bull_flag").signal(make_ctx(), make_bars(closes, vols, vwap=100.5))
    assert s and s.side == "long" and "flag" in s.reason


def test_mean_reversion(cfg):
    closes = [100.0 + (0.02 if i % 2 else -0.02) for i in range(80)]
    closes += [99.8, 99.4, 99.0, 98.7, 98.5]   # stretch below the band
    closes += [98.6, 98.8, 98.9, 99.0, 99.0]   # turn back up
    bars = make_bars(closes, wick=0.02)
    ctx = make_ctx(stock_regime="range", market_regime="range", gap_pct=0.2, rvol=1.0, has_catalyst=False)
    s = strat(cfg, "mean_reversion").signal(ctx, bars)
    assert s and s.side == "long" and s.target > s.entry


def test_gap_fade_long(cfg):
    closes = [95.6, 96.4] * 8 + [96.0, 96.2, 96.7]
    bars = make_bars(closes, first_open=96.0, vwap=96.2)
    ctx = make_ctx(gap_pct=-4.0, prev_close=100.0, has_catalyst=False, market_regime="range", stock_regime="range")
    s = strat(cfg, "gap_fade").signal(ctx, bars)
    assert s and s.side == "long" and s.target == pytest.approx(99.0)


def test_shorts_disabled_by_default(cfg):
    closes = [100.05, 100.95] * 8 + [100.5, 100.4, 100.6, 100.3, 99.5]
    vols = [1000] * (len(closes) - 1) + [5000]
    ctx = make_ctx(market_regime="range", stock_regime="range", gap_pct=-3.0, shortable=True)
    assert strat(cfg, "orb").signal(ctx, make_bars(closes, vols)) is None
    cfg.risk["allow_short"] = True
    s = strat(cfg, "orb").signal(ctx, make_bars(closes, vols))
    assert s and s.side == "short" and s.stop > s.entry > s.target
    cfg.risk["short_market_regimes"] = ["trend_down"]  # shorts limited to falling markets
    assert strat(cfg, "orb").signal(ctx, make_bars(closes, vols)) is None


def test_shorts_need_a_gap_down_with_news(cfg):
    cfg.risk["allow_short"] = True
    o = strat(cfg, "orb")
    down = make_ctx(market_regime="trend_down", stock_regime="trend_down", gap_pct=-3.0, shortable=True)
    assert o.side_allowed(down, "short") and not o.side_allowed(down, "long")
    up_stock = make_ctx(market_regime="trend_down", stock_regime="trend_up", gap_pct=-3.0, shortable=True)
    assert not o.side_allowed(up_stock, "short")  # never short a stock in a daily uptrend
    small = make_ctx(market_regime="trend_down", stock_regime="trend_down", gap_pct=-0.5, shortable=True)
    assert not o.side_allowed(small, "short")
