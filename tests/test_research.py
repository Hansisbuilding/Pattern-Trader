"""Exit-rule outcomes used by the signal research."""

import pytest
from conftest import make_bars

from patterntrader.config import parse_time
from patterntrader.research import outcomes

FLAT = parse_time("15:50")


def test_target_and_stop_outcomes():
    # signal on bar 0 at 100, fill at bar 1's open (100), stop 99 -> 1R = $1; price runs to 102.5
    b = make_bars([100, 100, 101, 102, 102.5, 102.5], wick=0)
    res = outcomes(b, 0, stop=99.0, target=102.0, slip=0.0, flatten=FLAT)
    assert res["r_t1"] == pytest.approx(1.0)
    assert res["r_t2"] == pytest.approx(2.0)
    assert res["r_t3"] == pytest.approx(2.5)  # never reached: exits at the last close
    assert res["r_orig"] == pytest.approx(2.0)


def test_stop_checked_before_target_and_breakeven():
    # up 1R, then back through entry: breakeven rule exits flat, fixed 2R target loses 1R
    b = make_bars([100, 100, 101.2, 100.5, 99.5, 98.5], wick=0)
    res = outcomes(b, 0, stop=99.0, target=102.0, slip=0.0, flatten=FLAT)
    assert res["r_t2"] == pytest.approx(-1.0)
    assert res["r_be2"] == pytest.approx(0.0)
    assert res["r_t1"] == pytest.approx(1.0)
