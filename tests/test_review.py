"""The reviewer: live stats, the limits on what it may change, and a full review with a stand-in Claude."""

import csv
from datetime import date

import pytest

from patterntrader.config import load_config
from patterntrader.review import agent, knobs, stats

FIELDS = ["time", "event", "symbol", "strategy", "side", "qty", "price", "stop", "target", "pnl", "note"]


@pytest.fixture
def base():
    return load_config()   # as written: orb off, the playbook regimes


def write_journal(root, rows):
    (root / "journal").mkdir(exist_ok=True)
    with open(root / "journal" / "trades.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def trade(day, sym, strat, entry, stop, exit_, qty=100, regime="range"):
    pnl = (exit_ - entry) * qty
    return [
        {"time": f"{day}T09:36:00-04:00", "event": "regime", "note": f"{regime}: SPY {regime}"},
        {"time": f"{day}T09:40:00-04:00", "event": "entry", "symbol": sym, "strategy": strat,
         "qty": qty, "price": entry, "stop": stop},
        {"time": f"{day}T15:50:00-04:00", "event": "exit", "symbol": sym, "strategy": strat,
         "qty": qty, "price": exit_, "stop": stop, "pnl": pnl},
    ]


def test_trades_pair_up_with_r_and_percentages(tmp_path):
    write_journal(tmp_path, trade("2026-11-02", "AAA", "gap_and_go", 50, 49, 52)
                  + trade("2026-11-03", "BBB", "bull_flag", 20, 19.5, 19.5, regime="trend_up"))
    t = stats.closed_trades(tmp_path)
    assert [(x["r"], x["pct_of_trade"], x["regime"]) for x in t] == [(2.0, 4.0, "range"), (-1.0, -2.5, "trend_up")]
    s = stats.summarize(t, 1_000_000)
    assert s["trades"] == 2 and s["win_rate_pct"] == 50 and s["expectancy_r"] == 0.5
    assert s["pnl_pct_of_portfolio"] == pytest.approx(0.015)
    assert stats.drawdown_pct(t, "2026-11-01", 1_000_000) == pytest.approx(-0.005)


def test_limits(base):
    spec = knobs.spec_for(base, "risk.max_trades_per_day")
    assert knobs.check_value(spec, 4, 5)[0]
    assert "at most 1" in knobs.check_value(spec, 4, 6)[1]          # one step at a time
    assert knobs.check_value(knobs.spec_for(base, "risk.max_daily_loss_pct"), 2.0, 2.5)[1].startswith("must be between")
    assert knobs.spec_for(base, "risk.allow_short") is None          # off limits
    assert knobs.spec_for(base, "live.github_only") is None
    assert knobs.spec_for(base, "strategies.mean_reversion.enabled") is None   # not in its list
    regimes = knobs.spec_for(base, "strategies.bull_flag.market_regimes")
    assert knobs.check_value(regimes, ["trend_up", "mixed"], ["range", "trend_up", "mixed"])[2] == ["trend_up", "range", "mixed"]
    assert not knobs.check_value(regimes, ["trend_up"], ["range", "mixed"])[0]      # two moves at once
    assert not knobs.check_value(regimes, ["trend_up"], ["trend_down"])[0]          # never downtrends
    win = knobs.spec_for(base, "strategies.gap_and_go.window")
    assert knobs.check_value(win, ["09:35", "10:30"], ["09:35", "11:00"])[0]
    assert not knobs.check_value(win, ["09:35", "10:30"], ["09:35", "11:30"])[0]
    assert not knobs.check_value(win, ["09:35", "10:30"], ["09:30", "10:30"])[0]   # before 09:35


def test_overrides_outside_limits_are_ignored(base, tmp_path):
    knobs.save_overrides(base, {"risk.max_trades_per_day": 6, "risk.risk_per_trade_pct": 5.0,
                                "risk.allow_short": True}, tmp_path)
    cfg = knobs.apply_overrides(base, tmp_path)
    assert cfg.risk.max_trades_per_day == 6
    assert cfg.risk.risk_per_trade_pct == base.risk.risk_per_trade_pct   # 5% is over the 0.75% ceiling
    assert cfg.risk.allow_short is False
    assert base.risk.max_trades_per_day == 4                              # config.yaml untouched


def test_due_periods():
    assert agent.due_periods(date(2026, 11, 3), {}) == ["daily"]                       # a Tuesday
    assert agent.due_periods(date(2026, 11, 6), {}) == ["daily", "weekly"]             # Friday
    assert agent.due_periods(date(2026, 12, 31), {}) == ["daily", "monthly", "quarterly", "yearly"]
    assert agent.due_periods(date(2027, 1, 29), {}) == ["daily", "weekly", "monthly"]
    # a missed weekly review (holiday Friday) is caught up the next run
    assert "weekly" in agent.due_periods(date(2026, 11, 10), {"weekly": "2026-10-30"})


def fake_claude(answer):
    calls = []

    def transport(argv, stdin, timeout, cwd):
        calls.append((argv, stdin))
        return {"structured_output": answer}

    return transport, calls


ANSWER = {
    "summary": "Two trades, one win.", "working": ["gap and go"], "not_working": ["bull flag"],
    "changes": [
        {"knob": "risk.max_trades_per_day", "value": 3, "reason": "fewer, better", "expect": "judge after 20"},
        {"knob": "risk.risk_per_trade_pct", "value": 0.75, "reason": "bigger", "expect": "x"},   # over the limit count
        {"knob": "risk.allow_short", "value": True, "reason": "try shorts", "expect": "x"},
    ],
    "undo": [], "lessons": "Lesson one.", "watch_next": ["bull flag in mixed markets"],
}


def test_full_review_applies_only_what_the_limits_allow(base, tmp_path):
    (tmp_path / "config.yaml").write_text("rules")
    write_journal(tmp_path, trade("2026-11-02", "AAA", "gap_and_go", 50, 49, 52))
    transport, calls = fake_claude(ANSWER)
    [report] = agent.run(base, date(2026, 11, 2), 1_000_000, transport=transport, root=tmp_path)

    argv, prompt = calls[0]
    assert argv[argv.index("--tools") + 1] == "" and argv[argv.index("--model") + 1] == "sonnet"
    assert "DAILY REVIEW" in prompt and "risk.max_trades_per_day" in prompt
    assert knobs.load_overrides(base, tmp_path) == {"risk.max_trades_per_day": 3}
    log = agent.State(base, tmp_path).changes()
    assert [r["action"] for r in log] == ["change", "rejected", "rejected"]
    assert "only 1 change" in log[1]["reason"] and "not a setting" in log[2]["reason"]
    assert (tmp_path / "reviewer" / "lessons.md").read_text() == "Lesson one."
    assert "Turned down by the limits" in report.read_text()

    # the next day the setting is locked, but it can be undone
    again = {**ANSWER, "changes": [ANSWER["changes"][0] | {"value": 2}],
             "undo": [{"knob": "risk.max_trades_per_day", "reason": "didn't help"}]}
    transport, _ = fake_claude(again)
    agent.run(base, date(2026, 11, 3), 1_000_000, periods=["daily"], transport=transport, root=tmp_path)
    assert knobs.load_overrides(base, tmp_path) == {}
    assert [r["action"] for r in agent.State(base, tmp_path).changes()][-2:] == ["undo", "rejected"]


def test_safety_brake_undoes_everything(base, tmp_path):
    (tmp_path / "config.yaml").write_text("rules")
    write_journal(tmp_path, trade("2026-11-02", "AAA", "gap_and_go", 50, 49, 52))
    transport, _ = fake_claude(ANSWER)
    agent.run(base, date(2026, 11, 2), 1_000_000, transport=transport, root=tmp_path)
    write_journal(tmp_path, trade("2026-11-02", "AAA", "gap_and_go", 50, 49, 52)
                  + trade("2026-11-03", "BBB", "gap_and_go", 100, 90, 50, qty=1000))   # -$50k = -5%
    transport, _ = fake_claude({**ANSWER, "changes": []})
    agent.run(base, date(2026, 11, 3), 1_000_000, transport=transport, root=tmp_path)
    assert knobs.load_overrides(base, tmp_path) == {}
    assert "auto-undo" in [r["action"] for r in agent.State(base, tmp_path).changes()]


def test_last_pattern_cannot_be_switched_off(base, tmp_path):
    (tmp_path / "config.yaml").write_text("rules")
    write_journal(tmp_path, trade("2026-11-02", "AAA", "gap_and_go", 50, 49, 52))
    knobs.save_overrides(base, {"strategies.vwap_pullback.enabled": False,
                                "strategies.bull_flag.enabled": False}, tmp_path)
    off = {**ANSWER, "changes": [{"knob": "strategies.gap_and_go.enabled", "value": False,
                                  "reason": "r", "expect": "e"}]}
    transport, _ = fake_claude(off)
    agent.run(base, date(2026, 11, 2), 1_000_000, periods=["daily"], transport=transport, root=tmp_path)
    assert "must stay switched on" in agent.State(base, tmp_path).changes()[-1]["reason"]
