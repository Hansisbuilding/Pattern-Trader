"""Indicators, regimes, risk rules, scoring and the paper-only guard."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from conftest import DAY, make_bars, make_ctx, make_daily

from patterntrader.config import ET, LiveKeyError, assert_paper_key
from patterntrader.context import evaluate, opening_rvol, opening_volumes
from patterntrader.indicators import atr, resample_complete, rsi, session_vwap
from patterntrader.models import Signal
from patterntrader.regime import classify_trend, market_regime
from patterntrader.risk import RiskManager
from patterntrader.strategies import build_strategies


def sig(entry=50.0, stop=49.5, target=51.0, symbol="TEST", side="long", t="10:00"):
    hh, mm = map(int, t.split(":"))
    return Signal("orb", symbol, side, entry, stop, target, datetime(2026, 3, 2, hh, mm, tzinfo=ET), "test")


# -- guard ---------------------------------------------------------------------
def test_live_keys_are_refused():
    with pytest.raises(LiveKeyError):
        assert_paper_key("AKLIVEKEY123")
    assert_paper_key("PKPAPERKEY123")


# -- indicators ----------------------------------------------------------------
def test_rsi_extremes():
    up = pd.Series(np.arange(1, 40, dtype=float))
    assert rsi(up, 14).iloc[-1] == pytest.approx(100)
    assert rsi(up[::-1].reset_index(drop=True), 14).iloc[-1] < 1


def test_atr_constant_range():
    d = make_daily([100.0] * 50, spread=1.0)
    assert atr(d).iloc[-1] == pytest.approx(2.0, rel=1e-3)


def test_vwap_equal_volume_is_mean_typical_price():
    b = make_bars([100, 102], wick=0)
    typical = ((b.high + b.low + b.close) / 3).mean()
    assert session_vwap(b).iloc[-1] == pytest.approx(typical)


def test_resample_drops_incomplete_bar():
    b = make_bars([100 + i * 0.1 for i in range(12)])  # 09:30-09:41
    r = resample_complete(b, 5, b.index[0])
    assert len(r) == 2 and r.index[-1].minute == 35


def test_opening_rvol():
    assert opening_rvol(3000, [1000, 1000, 1000]) == pytest.approx(3.0)
    assert opening_rvol(3000, [1000]) is None  # not enough history
    b = make_bars([100] * 10, volumes=[100] * 10)
    assert opening_volumes(b, 5) == {DAY: 500.0}


# -- regime --------------------------------------------------------------------
def test_regimes():
    up = make_daily([100 + i for i in range(120)])
    assert classify_trend(up)[0] == "trend_up"
    down = make_daily([300 - i for i in range(120)])
    assert classify_trend(down)[0] == "trend_down"
    rng = np.random.default_rng(1)
    flat = make_daily(list(100 + rng.normal(0, 0.3, 120)), spread=1.0)
    assert classify_trend(flat)[0] == "range"


def test_market_chaos_on_big_gap(cfg):
    up = make_daily([100 + i for i in range(120)])
    assert market_regime(up, -2.0, cfg.regime)[0] == "chaos"
    assert market_regime(up, 0.2, cfg.regime)[0] == "trend_up"


# -- risk ----------------------------------------------------------------------
def test_sizing_risk_and_cap(cfg):
    rm = RiskManager(cfg, 100_000)
    # 0.5% of 100k = $500 risk / $0.50 per share = 1000 shares, capped by 20% of equity / $50 = 400
    assert rm.size(sig(), 100_000) == 400
    assert rm.size(sig(entry=10, stop=9, target=12), 100_000) == 500  # 500 shares = $5k, under cap


def test_risk_rules(cfg):
    rm = RiskManager(cfg, 100_000)
    assert rm.approve(sig(t="09:31"), sig(t="09:31").time)[1] == "outside entry hours"
    s = sig()
    assert rm.approve(s, s.time)[0]
    rm.on_entry(s)
    assert not rm.approve(s, s.time)[0]  # already in symbol
    rm.on_exit("TEST", -500)
    assert "no re-entry" in rm.approve(s, s.time)[1]
    s2 = sig(symbol="OTHER")
    rm.on_entry(s2)
    rm.on_exit("OTHER", -500)
    assert rm.halted and "in a row" in rm.halted


def test_daily_loss_limit_flattens(cfg):
    rm = RiskManager(cfg, 100_000)
    rm.update_equity(98_500)
    assert not rm.halted
    rm.update_equity(97_900)
    assert rm.flatten_now and "daily loss" in rm.halted


def test_noise_stop_rejected(cfg):
    rm = RiskManager(cfg, 100_000)
    tight = sig(entry=100, stop=99.95, target=100.2)
    assert "noise" in rm.approve(tight, tight.time)[1]


# -- situation gating ----------------------------------------------------------
def test_chaos_market_blocks_everything(cfg):
    ctx = evaluate(make_ctx(market_regime="chaos"), build_strategies(cfg), cfg)
    assert ctx.eligible == [] and ctx.score == 0


def test_trend_day_vs_range_day_playbooks(cfg):
    strategies = build_strategies(cfg)
    trend = evaluate(make_ctx(gap_pct=5.0), strategies, cfg)
    assert {"orb", "gap_and_go", "vwap_pullback"} <= set(trend.eligible)
    assert "mean_reversion" not in trend.eligible  # never fade a trending stock

    quiet = evaluate(
        make_ctx(stock_regime="range", market_regime="range", gap_pct=0.5, rvol=1.0, has_catalyst=False),
        strategies, cfg,
    )
    assert "mean_reversion" in quiet.eligible
    assert "vwap_pullback" not in quiet.eligible and "gap_and_go" not in quiet.eligible


def test_no_longs_in_down_market_when_shorts_off(cfg):
    ctx = evaluate(make_ctx(market_regime="trend_down", stock_regime="trend_down"), build_strategies(cfg), cfg)
    assert ctx.eligible == []


def test_gap_strategies_respect_news(cfg):
    strategies = build_strategies(cfg)
    news = evaluate(make_ctx(gap_pct=6, has_catalyst=True, market_regime="range", stock_regime="range"), strategies, cfg)
    assert "gap_and_go" in news.eligible and "gap_fade" not in news.eligible
    quiet = evaluate(make_ctx(gap_pct=-5, has_catalyst=False, market_regime="range", stock_regime="range", rvol=2.0), strategies, cfg)
    assert "gap_fade" in quiet.eligible and "gap_and_go" not in quiet.eligible
    unknown = evaluate(make_ctx(gap_pct=6, has_catalyst=None), strategies, cfg)
    assert "gap_and_go" not in unknown.eligible


def test_edge_gate_blocks_unproven(cfg, tmp_path, monkeypatch):
    from patterntrader import edge as edge_mod

    monkeypatch.setattr(edge_mod, "ROOT", tmp_path)
    gate = edge_mod.EdgeGate(cfg)
    assert gate.allows("orb") == (False, "no backtest results yet (edge gate)")
    ctx = evaluate(make_ctx(), build_strategies(cfg), cfg, gate)
    assert ctx.eligible == []
