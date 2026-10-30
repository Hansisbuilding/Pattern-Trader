"""Sentiment: analyst actions, headline tone summaries and the market mood score (offline)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from patterntrader.sentiment import HeadlineScorer, analyst_actions, market_mood, mood_label

HEADLINES = [
    "Morgan Stanley Upgrades Acme to Overweight, Raises Price Target to $120",
    "Goldman Sachs Initiates Coverage On Acme With Buy Rating",
    "Acme Q3 EPS $1.20 Beats $1.05 Estimate, Raises FY Guidance",
    "Acme, Beta And Other Big Stocks Moving Higher In Tuesday's Pre-Market Session",  # roundup
    "Barclays Downgrades Beta, Lowers Price Target",
]


def test_analyst_actions_skip_roundups():
    a = analyst_actions(HEADLINES)
    assert a == {"upgrades": 1, "downgrades": 1, "initiations": 1, "pt_raises": 1, "pt_cuts": 1}


def test_tone_summary_from_cache(tmp_path, monkeypatch):
    from patterntrader import sentiment

    monkeypatch.setattr(sentiment, "ROOT", tmp_path)
    s = HeadlineScorer(".cache")
    s.scores = {s.key(h): v for h, v in zip(HEADLINES, [0.6, 0.2, 0.9, -0.9, -0.7])}
    out = s.summarize(HEADLINES, model=False)
    assert out["news_count"] == 4  # the roundup is dropped from the tone
    assert out["news_tone"] == pytest.approx(np.mean([0.6, 0.2, 0.9, -0.7]))
    assert out["news_tone_min"] == pytest.approx(-0.7)


class FakeExt:
    def __init__(self, vix):
        idx = [date(2025, 1, 1) + pd.Timedelta(days=i) for i in range(len(vix))]
        self.v = pd.DataFrame({"close": vix}, index=idx)

    def cboe(self, name):
        return self.v if name == "VIX" else self.v.assign(close=20.0)

    def fred(self, sid):
        return None

    def aaii(self):
        return None


def test_market_mood_extremes():
    n = 300
    spy = pd.DataFrame({"close": np.linspace(100, 130, n)},
                       index=pd.DatetimeIndex([pd.Timestamp("2025-01-01") + pd.Timedelta(days=i) for i in range(n)]))
    calm = market_mood(FakeExt([20.0] * (n - 1) + [12.0]), spy, date(2026, 1, 1))
    panic = market_mood(FakeExt([15.0] * (n - 1) + [45.0]), spy, date(2026, 1, 1))
    assert calm["mood"] > 60 and panic["mood"] < calm["mood"]
    assert panic["parts"]["vix"]["greed_score"] == pytest.approx(0, abs=1)
    assert mood_label(10) == "extreme fear" and mood_label(50) == "neutral" and mood_label(90) == "extreme greed"


# -- rules -------------------------------------------------------------------------
def rules_cfg(cfg, **over):
    cfg.sentiment.update({"enabled": True, "sit_out_vix_term_above": 1.0, "half_size_below_mood": 25,
                          "half_size_events": ["FOMC"], "block_news_tone_below": -0.3,
                          "block_offering": True, "min_quality": None, "half_size_below_quality": 51, **over})
    cfg.sentiment.quality.update({"news_tone": 25, "eps_surprise": 15, "analyst": 8, "squeeze": 10,
                                  "squeeze_short_pct": 5, "insider_buy": 5, "offering": 15})
    return cfg


def test_day_size_factor(cfg):
    from patterntrader.sentiment import day_size_factor

    c = rules_cfg(cfg)
    assert day_size_factor(50, 0.9, [], c) == (1.0, [])
    assert day_size_factor(50, 1.1, [], c)[0] == 0.0
    assert day_size_factor(20, 0.9, ["FOMC"], c)[0] == 0.25
    c.sentiment["enabled"] = False
    assert day_size_factor(5, 2.0, ["FOMC"], c) == (1.0, [])


def test_gates_and_ranking(cfg):
    from conftest import make_ctx

    from patterntrader.context import evaluate
    from patterntrader.strategies import build_strategies

    c = rules_cfg(cfg)
    strategies = build_strategies(c)
    good = make_ctx(gap_pct=5.0)
    good.info.update({"news_tone": 0.8, "eps_surprise_pct": 12.0, "upgrades": 1})
    bad = make_ctx(gap_pct=5.0)
    bad.info.update({"news_tone": -0.6})
    dilution = make_ctx(gap_pct=5.0, filings=["offering"])
    good, bad, dilution = (evaluate(x, strategies, c) for x in (good, bad, dilution))
    assert good.eligible and good.info["catalyst_quality"] > 80
    assert not bad.eligible and bad.info["sentiment_block"].startswith("news tone")
    assert bad.info["eligible_without_sentiment"]  # the baseline would have traded it
    assert not dilution.eligible and "offering" in dilution.info["sentiment_block"]
    plain = evaluate(make_ctx(gap_pct=5.0), strategies, c)
    assert good.score > plain.score  # better news ranks higher on the watchlist


def test_size_factor_scales_risk(cfg):
    from datetime import datetime

    from patterntrader.config import ET
    from patterntrader.models import Signal
    from patterntrader.risk import RiskManager

    sig = Signal("orb", "TEST", "long", 50.0, 49.5, 51.0, datetime(2026, 3, 2, 10, 0, tzinfo=ET), "t")
    assert RiskManager(cfg, 100_000, 0.5).size(sig, 100_000) == 500
    assert not RiskManager(cfg, 100_000, 0.0).approve(sig, sig.time)[0]


def test_setup_size_factor(cfg):
    from conftest import make_ctx

    from patterntrader.sentiment import setup_size_factor

    c = rules_cfg(cfg)
    plain, squeeze = make_ctx(), make_ctx()
    plain.info["catalyst_quality"], squeeze.info["catalyst_quality"] = 50, 60
    assert setup_size_factor(plain, c) == 0.5 and setup_size_factor(squeeze, c) == 1.0
