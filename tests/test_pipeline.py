"""End-to-end runs of the backtest and the live scanner analysis against fake market data."""

from datetime import date, datetime, time

import numpy as np
import pandas as pd

from patterntrader.backtest import run_backtest
from patterntrader.config import ET
from patterntrader.scanner import analyze, pick_watchlist
from patterntrader.strategies import build_strategies


class FakeMD:
    """Synthetic trending daily bars and noisy minute bars with a strong open."""

    def __init__(self, end=date(2026, 3, 31)):
        self.days = [d.date() for d in pd.bdate_range("2024-06-03", end)]

    def _daily(self, sym):
        n = len(self.days)
        drift = {"SPY": 0.05, "UP": 0.3, "GAP": 0.2}.get(sym, 0.0)
        close = 100 + drift * np.arange(n)
        idx = pd.DatetimeIndex([pd.Timestamp(d, tz=ET) for d in self.days])
        opens = close * np.where(np.arange(n) % 7 == 0, 1.05 if sym == "GAP" else 1.0, 1.0)
        return pd.DataFrame(
            {"open": opens, "high": np.maximum(opens, close) + 1.5, "low": np.minimum(opens, close) - 1.5,
             "close": close, "volume": 2e6},
            index=idx,
        )

    def daily(self, symbols, start, end, feed):
        out = {}
        for s in symbols:
            df = self._daily(s)
            out[s] = df[(df.index.date >= start) & (df.index.date <= end)]
        return out

    def _minutes(self, sym, day):
        d = self._daily(sym)
        row = d[d.index.date == day].iloc[0]
        rng = np.random.default_rng(abs(hash((sym, day))) % 2**32)
        n = 390
        path = row.open + np.cumsum(rng.normal(0.01, 0.08, n))
        t0 = datetime.combine(day, time(9, 30), ET)
        idx = pd.DatetimeIndex([t0 + pd.Timedelta(minutes=i) for i in range(n)])
        vol = np.full(n, 5000.0)
        vol[:5] = 40000 if day.toordinal() % 3 == 0 else 15000
        opens = np.concatenate([[row.open], path[:-1]])
        return pd.DataFrame(
            {"open": opens, "high": np.maximum(opens, path) + 0.05, "low": np.minimum(opens, path) - 0.05,
             "close": path, "volume": vol},
            index=idx,
        )

    def minutes(self, symbols, start, end, feed):
        return {
            s: pd.concat([self._minutes(s, d) for d in self.days if start <= d <= end]) for s in symbols
        }

    def headlines(self, symbol, start, end):
        return ["Big news"] if symbol == "GAP" else []

    def fetch_bars(self, symbols, minutes, start, end, feed):
        out = {}
        for s in symbols:
            df = self._minutes(s, start.date())
            out[s] = df[df.index <= end]
        return out

    def opening_volumes(self, symbols, days, minutes, feed):
        return {s: {d: float(self._minutes(s, d)["volume"].iloc[:minutes].sum()) for d in days} for s in symbols}


def test_backtest_runs_end_to_end(cfg):
    cfg.scan["min_avg_dollar_volume"] = 1e6
    cfg.scan["min_score"] = 0
    summary = run_backtest(cfg, FakeMD(), ["UP", "GAP", "FLAT"], date(2026, 1, 5), date(2026, 3, 31))
    assert summary["days_by_market_regime"]
    df = summary["trades_df"]
    assert summary["overall"]["trades"] == len(df)
    if len(df):
        assert set(df.strategy) <= {s.name for s in build_strategies(cfg)}
        assert (df.groupby("day").size() <= cfg.risk.max_trades_per_day).all()
        assert (df.exit_time.dt.time <= time(15, 59)).all()


def test_live_analysis_builds_watchlist(cfg):
    cfg.scan["min_avg_dollar_volume"] = 1e6
    md = FakeMD()
    now = datetime(2026, 3, 30, 9, 36, tzinfo=ET)
    rows = [
        {"symbol": s, "price": 150.0, "prev_close": 150.0, "gap_pct": g, "avg_dollar_volume": 3e8, "shortable": True}
        for s, g in (("UP", 0.5), ("GAP", 5.0), ("FLAT", 0.1))
    ]
    regime, notes, contexts = analyze(cfg, md, rows, now, build_strategies(cfg), edge=None)
    assert regime in {"trend_up", "range", "mixed", "trend_down", "chaos"}
    by = {c.symbol: c for c in contexts}
    assert by["GAP"].has_catalyst is True and by["UP"].has_catalyst is None
    assert all(c.rvol is not None for c in contexts)  # after 09:35 the RVOL is confirmed
    assert pick_watchlist(contexts, cfg) == [c for c in contexts if c.eligible and c.score >= cfg.scan.min_score][:5]
