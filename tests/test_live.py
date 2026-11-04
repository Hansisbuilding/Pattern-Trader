"""Morning-to-afternoon hand-off: the resumed run must see the same positions and risk counters."""

from datetime import datetime, timedelta

from conftest import DAY, make_ctx

from patterntrader import live
from patterntrader.config import ET
from patterntrader.models import Signal
from patterntrader.risk import RiskManager


class FakeJournal:
    def __init__(self):
        self.rows = []

    def record(self, event, *a, **kw):
        self.rows.append(event)


def bare_trader(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(live, "ROOT", tmp_path)
    t = live.Trader.__new__(live.Trader)  # skip the broker connection
    t.cfg, t.day, t.journal = cfg, DAY, FakeJournal()
    t.tracked, t.fired = {}, set()
    return t


def test_handoff_round_trip(cfg, tmp_path, monkeypatch):
    morning = bare_trader(cfg, tmp_path, monkeypatch)
    sig = Signal("bull_flag", "TEST", "long", 50.0, 49.0, 60.0, datetime(2026, 3, 2, 10, 5, tzinfo=ET), "test")
    risk = RiskManager(cfg, 100_000)
    risk.on_entry(sig)
    morning.tracked = {"TEST": {"order_id": "abc", "sig": sig, "qty": 100}}
    morning.fired = {("TEST", "bull_flag")}
    morning.save_state([make_ctx()], risk)

    afternoon = bare_trader(cfg, tmp_path, monkeypatch)
    active, risk2 = afternoon.load_state()
    assert [c.symbol for c in active] == ["TEST"]
    assert afternoon.tracked["TEST"]["order_id"] == "abc"
    assert ("TEST", "bull_flag") in afternoon.fired
    assert risk2.trades_today == risk.trades_today == 1
    assert afternoon.journal.rows == ["resume"]


def test_resume_without_state_is_none(cfg, tmp_path, monkeypatch):
    assert bare_trader(cfg, tmp_path, monkeypatch).load_state() is None


def test_orders_only_from_github(cfg, monkeypatch):
    import argparse

    import pytest

    from patterntrader import cli

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    cfg["live"] = {"github_only": True}
    args = argparse.Namespace(dry_run=False, stop_at=None, resume=False)
    with pytest.raises(SystemExit, match="only places orders from GitHub"):
        cli.cmd_trade(cfg, args)


# -- entries: capped limit orders instead of an IEX spread check ---------------------------
class FakeBroker:
    def __init__(self):
        self.orders, self.cancelled = [], []

    def submit_bracket(self, sig, qty, day, limit_price):
        self.orders.append((sig.symbol, qty, limit_price, sig.stop, sig.target))
        return f"o{len(self.orders)}"

    def positions(self):
        return {}

    def cancel(self, order_id):
        self.cancelled.append(order_id)

    def order(self, order_id):
        from types import SimpleNamespace

        return SimpleNamespace(filled_qty=0, status="accepted")


def trader_with_broker(cfg, tmp_path, monkeypatch):
    from patterntrader.strategies import build_strategies

    t = bare_trader(cfg, tmp_path, monkeypatch)
    t.broker, t.dry_run = FakeBroker(), False
    t.strategies = {s.name: s for s in build_strategies(cfg)}
    return t


def test_entry_is_a_limit_capped_at_max_spread(cfg, tmp_path, monkeypatch):
    t = trader_with_broker(cfg, tmp_path, monkeypatch)
    sig = Signal("gap_and_go", "TEST", "long", 100.0, 98.5, 115.0, datetime(2026, 3, 2, 9, 40, tzinfo=ET), "test")
    assert t.entry_limit(sig) == 100.15  # max_spread_pct 0.15 above the signal price
    risk = RiskManager(cfg, 1_000_000)
    assert t.try_enter(sig, make_ctx(), risk, sig.time, 1_000_000, 4_000_000)
    sym, qty, limit, stop, target = t.broker.orders[0]
    assert (sym, limit, stop, target) == ("TEST", 100.15, 98.5, 115.0)
    assert qty == 3030  # 0.5% of $1M = $5,000 risk / $1.65 per share at the limit price
    assert t.journal.rows == ["entry"] and t.tracked["TEST"]["placed"] == sig.time


def test_unfilled_entry_is_cancelled_after_timeout(cfg, tmp_path, monkeypatch):
    t = trader_with_broker(cfg, tmp_path, monkeypatch)
    sig = Signal("gap_and_go", "TEST", "long", 100.0, 98.5, 115.0, datetime(2026, 3, 2, 9, 40, tzinfo=ET), "test")
    risk = RiskManager(cfg, 1_000_000)
    t.tracked = {"TEST": {"order_id": "o1", "sig": sig, "qty": 10, "placed": live.now_et() - timedelta(minutes=1)}}
    t.sync(risk)
    assert t.broker.cancelled == []  # 1 minute: still waiting
    t.tracked["TEST"]["placed"] = live.now_et() - timedelta(minutes=3)
    t.sync(risk)
    assert t.broker.cancelled == ["o1"]
