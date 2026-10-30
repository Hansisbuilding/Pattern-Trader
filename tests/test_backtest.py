"""Backtest fill logic must be conservative and use the same risk rules as live."""

from datetime import time

import pytest
from conftest import make_bars, make_ctx

from patterntrader.backtest import simulate_day, summarize
from patterntrader.strategies.base import Strategy


class OneShot(Strategy):
    """Fires a long at a fixed bar with fixed stop/target."""

    name = "dummy"

    def __init__(self, cfg, at, stop, target):
        super().__init__({"window": ["09:30", "15:59"], "max_stop_atr": 10, "min_rr": 1.0}, cfg)
        self.at, self.stop_px, self.target_px = at, stop, target

    def signal(self, ctx, bars):
        ts = bars.index[-1]
        if ts.time() == self.at:
            c = bars["close"].iloc[-1]
            return self.emit(ctx, "long", c, self.stop_px, self.target_px, ts, "test")
        return None


def run(cfg, closes, stop, target, at=time(9, 40), **bar_kw):
    bars = make_bars(closes, **bar_kw)
    ctx = make_ctx(eligible=["dummy"])
    strategies = {"dummy": OneShot(cfg, at, stop, target)}
    return simulate_day([(ctx, bars)], strategies, cfg, 100_000)


def test_entry_at_next_open_with_slippage(cfg):
    closes = [100.0] * 10 + [100.0, 101.0] + [101.0] * 5 + [103.5]
    trades, _ = run(cfg, closes, stop=99.0, target=103.0)
    t = trades[0]
    assert t.entry_time.time() == time(9, 41)
    assert t.entry == pytest.approx(100.0 * 1.0003)
    assert t.exit_reason == "target" and t.exit == pytest.approx(103.0)


def test_stop_assumed_first_when_bar_hits_both(cfg):
    closes = [100.0] * 11 + [100.0]
    bars = make_bars(closes)
    bars.iloc[11, bars.columns.get_loc("high")] = 104
    bars.iloc[11, bars.columns.get_loc("low")] = 98
    ctx = make_ctx(eligible=["dummy"])
    trades, equity = simulate_day([(ctx, bars)], {"dummy": OneShot(cfg, time(9, 40), 99.0, 103.0)}, cfg, 100_000)
    assert trades[0].exit_reason == "stop" and trades[0].pnl < 0 and equity < 100_000
    # Slightly worse than -1R because slippage applies on both the entry and the stop.
    assert trades[0].r_multiple == pytest.approx(-1.03, abs=0.01)


def test_stop_gap_fills_worse(cfg):
    bars = make_bars([100.0] * 14)
    for col, px in (("open", 97.0), ("high", 97.1), ("low", 96.9), ("close", 97.0)):
        bars.iloc[12, bars.columns.get_loc(col)] = px
    ctx = make_ctx(eligible=["dummy"])
    trades, _ = simulate_day([(ctx, bars)], {"dummy": OneShot(cfg, time(9, 40), 99.0, 103.0)}, cfg, 100_000)
    assert trades[0].exit_reason == "stop"
    assert trades[0].exit == pytest.approx(97.0 * (1 - 0.0003))  # filled at the gap, not at 99


def test_flatten_at_end_of_day(cfg):
    closes = [100.0] * 400
    trades, _ = run(cfg, closes, stop=95.0, target=110.0)
    assert trades[0].exit_reason == "flatten"
    assert trades[0].exit_time.time() == time(15, 50)


def test_summary_stats(cfg):
    closes = [100.0] * 10 + [100.0, 101.0] + [101.0] * 5 + [103.5]
    trades, _ = run(cfg, closes, stop=99.0, target=103.0)
    s = summarize(trades, cfg, trades[0].day, trades[0].day, {"trend_up": 1})
    assert s["overall"]["trades"] == 1 and s["overall"]["win_rate"] == 1.0
    assert s["strategies"]["dummy"]["expectancy_r"] > 2.5
