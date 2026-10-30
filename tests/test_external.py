"""Outside data: SEC filing classification, VIX context and the macro calendar (all offline)."""

from datetime import date, datetime

import pandas as pd
import pytest

from patterntrader.config import ET
from patterntrader.context import add_outside_data
from patterntrader.external import External, catalyst_window
from conftest import make_ctx


@pytest.fixture
def ext(tmp_path, monkeypatch):
    from patterntrader import external

    monkeypatch.setattr(external, "ROOT", tmp_path)
    x = External(".cache")
    x._filings["TEST"] = pd.DataFrame({
        "accepted": pd.DatetimeIndex([
            datetime(2026, 3, 1, 16, 30, tzinfo=ET),   # after the prior close: counts
            datetime(2026, 3, 2, 7, 0, tzinfo=ET),     # premarket: counts
            datetime(2026, 3, 2, 10, 0, tzinfo=ET),    # after the open: not known yet
            datetime(2026, 2, 20, 9, 0, tzinfo=ET),    # old
        ]),
        "form": ["8-K", "424B5", "8-K", "10-Q"],
        "items": ["2.02,9.01", "", "1.01", ""],
    })
    closes = [20.0] * 25 + [30.0]
    x._vix = pd.DataFrame({"close": closes}, index=[date(2026, 1, 1) + pd.Timedelta(days=i) for i in range(26)])
    x._events = {date(2026, 3, 2): ["CPI"]}
    x.short_interest = lambda sym: pd.DataFrame({
        "settlement": [date(2026, 2, 13), date(2026, 2, 27)], "short_shares": [8e6, 10e6],
        "avg_volume": [2e6, 2e6], "days_to_cover": [4.0, 5.0]})
    x.shares_outstanding = lambda sym, day: 100e6
    x._shvol = {}
    x._short_volume_day = lambda d: None
    x._insider = pd.DataFrame({"symbol": ["TEST", "TEST", "TEST"],
                               "filed": [date(2026, 2, 20), date(2026, 2, 25), date(2025, 12, 1)],
                               "code": ["P", "S", "P"], "value": [50_000.0, 20_000.0, 9e9]})
    x.fred = lambda sid: None

    class NoAttention:
        def wiki_attention(self, sym, day, max_age_hours=0):
            return {"wiki_views_ratio": 3.0}

    x._attention = NoAttention()
    x.score_headlines = False
    x.cboe = lambda idx: pd.DataFrame(columns=["close", "skew"])
    return x


def test_filings_known_before_the_open(ext):
    assert ext.catalysts("TEST", *catalyst_window(date(2026, 3, 2), 18)) == ["earnings", "offering"]


def test_vix_context_uses_prior_closes_only(ext):
    v = ext.vix_context(date(2026, 1, 26))  # the 30 close is on 01-26 itself: not known yet
    assert v["vix"] == 20 and v["vix_ratio"] == pytest.approx(1.0)
    v = ext.vix_context(date(2026, 1, 27))
    assert v["vix"] == 30 and v["vix_ratio"] > 1.4 and v["vix_5d"] == pytest.approx(10)


def test_context_gets_outside_data(ext):
    ctx = add_outside_data(make_ctx(), ext, 18)
    assert ctx.filings == ["earnings", "offering"] and ctx.events == ["CPI"]
    assert not [n for n in ctx.notes if "unavailable" in n]
    assert ctx.info["insider_buys_30d"] == 1 and ctx.info["insider_net_30d"] == 30_000
    assert ctx.info["wiki_views_ratio"] == 3.0 and ctx.info["news_count"] == 0


def test_short_interest_only_once_published(ext):
    # 02-27 settlement isn't public until ~03-11, so on 03-02 the 02-13 figure is the latest known
    si = ext.short_context("TEST", date(2026, 3, 2))
    assert si["short_pct_out"] == pytest.approx(8.0) and si["days_to_cover"] == 4.0
    si = ext.short_context("TEST", date(2026, 3, 12))
    assert si["short_pct_out"] == pytest.approx(10.0) and si["short_change_pct"] == pytest.approx(25.0)


def test_earnings_surprise_timing(ext, monkeypatch):
    hist = pd.DataFrame({"reportedDate": [date(2026, 2, 27), date(2026, 3, 2)],
                         "reportTime": ["post-market", "pre-market"], "surprisePercentage": [12.0, -3.0],
                         "reportedEPS": [1, 1], "estimatedEPS": [1, 1]})
    monkeypatch.setattr(ext, "earnings_history", lambda sym, fetch=False: hist)
    assert ext.earnings_surprise("TEST", date(2026, 3, 2)) == 12.0  # Friday after the close -> Monday
    hist.loc[0, "reportTime"] = "pre-market"
    assert ext.earnings_surprise("TEST", date(2026, 3, 2)) == -3.0  # Monday before the open


def test_expiration_days():
    from patterntrader.external import expiration_events

    assert expiration_events(date(2026, 3, 20)) == ["quad witching"]
    assert expiration_events(date(2026, 4, 17)) == ["opex"]
    assert expiration_events(date(2026, 6, 26)) == ["Russell recon"]
    assert expiration_events(date(2026, 3, 19)) == []
