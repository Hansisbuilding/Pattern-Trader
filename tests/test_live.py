"""Morning-to-afternoon hand-off: the resumed run must see the same positions and risk counters."""

from datetime import datetime

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
