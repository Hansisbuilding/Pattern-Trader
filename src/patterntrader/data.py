"""Market data access via Alpaca with an on-disk parquet cache for history."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from patterntrader.config import ET, ROOT

log = logging.getLogger(__name__)
COLS = ["open", "high", "low", "close", "volume"]
EMPTY = pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], tz=ET))


def chunks(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def regular_session(df: pd.DataFrame, day: date | None = None) -> pd.DataFrame:
    """Bars from 09:30 to 15:59 ET (optionally only for one day)."""
    if df.empty:
        return df
    if day is not None:
        df = df[df.index.date == day]
    t = df.index.time
    return df[(t >= time(9, 30)) & (t < time(16, 0))]


def premarket_session(df: pd.DataFrame, day: date) -> pd.DataFrame:
    if df.empty:
        return df
    df = df[df.index.date == day]
    t = df.index.time
    return df[(t >= time(4, 0)) & (t < time(9, 30))]


class MarketData:
    def __init__(self, api_key: str, secret_key: str, cache_dir: str | Path = ".cache"):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.historical.news import NewsClient

        self.client = StockHistoricalDataClient(api_key, secret_key)
        self.news_client = NewsClient(api_key, secret_key)
        self.cache = ROOT / cache_dir
        self._mem: dict[Path, pd.DataFrame] = {}

    # -- raw fetch -----------------------------------------------------------
    def fetch_bars(
        self, symbols: list[str], minutes: int | None, start: datetime, end: datetime, feed: str,
        strict: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """minutes=None means daily bars. strict=True raises instead of skipping a failed batch."""
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf = TimeFrame.Day if minutes is None else TimeFrame(minutes, TimeFrameUnit.Minute)
        out: dict[str, pd.DataFrame] = {}
        for batch in chunks(sorted(set(symbols)), 100):
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=tf,
                start=start,
                end=end,
                feed=DataFeed(feed),
                adjustment=Adjustment.SPLIT,
            )
            try:
                df = self._with_retry(lambda: self.client.get_stock_bars(req).df)
            except Exception as exc:  # one bad symbol shouldn't kill a whole scan
                if strict:
                    raise RuntimeError(
                        f"Alpaca bar request failed ({str(exc).strip()[:120]}). Check the API keys; "
                        "a missing-data backtest would be meaningless, so stopping."
                    ) from exc
                log.warning("bar fetch failed for %d symbols: %s", len(batch), exc)
                continue
            if df.empty:
                continue
            for sym, g in df.groupby(level=0):
                g = g.droplevel(0)[COLS].astype(float)
                g.index = pd.DatetimeIndex(g.index).tz_convert(ET)
                out[str(sym)] = g
        return out

    @staticmethod
    def _with_retry(call, tries: int = 6):
        """Alpaca allows ~200 requests a minute; when busy, wait and try again instead of failing."""
        import time as _time

        for n in range(tries):
            try:
                return call()
            except Exception as exc:
                if "too many requests" not in str(exc).lower() or n == tries - 1:
                    raise
                wait = 15 * (n + 1)
                log.info("Alpaca rate limit hit; waiting %ds", wait)
                _time.sleep(wait)

    # -- cached history ------------------------------------------------------
    def _cached(self, kind, symbols, periods, minutes, feed) -> dict[str, pd.DataFrame]:
        parts: dict[str, list[pd.DataFrame]] = {s: [] for s in symbols}
        now = datetime.now(ET)
        for label, p_start, p_end in periods:
            complete = p_end < now - timedelta(days=1)
            folder = self.cache / kind / feed / label
            missing = []
            for s in symbols:
                f = folder / f"{s}.parquet"
                if complete and f.exists():
                    if f not in self._mem:  # multi-year backtests re-read the same months daily
                        if len(self._mem) > 400:
                            self._mem.clear()
                        self._mem[f] = pd.read_parquet(f)
                    parts[s].append(self._mem[f])
                else:
                    missing.append(s)
            if not missing:
                continue
            end = p_end
            if feed == "sip":  # free plan: SIP data must be >15 min old
                end = min(end, now - timedelta(minutes=16))
            fetched = self.fetch_bars(missing, minutes, p_start, end, feed, strict=True)
            if complete:
                folder.mkdir(parents=True, exist_ok=True)
            for s in missing:
                df = fetched.get(s, EMPTY)
                if complete:
                    df.to_parquet(folder / f"{s}.parquet")
                parts[s].append(df)
        out = {}
        for s, dfs in parts.items():
            dfs = [d for d in dfs if not d.empty]
            if dfs:
                df = pd.concat(dfs)
                out[s] = df[~df.index.duplicated()].sort_index()
        return out

    def daily(self, symbols: list[str], start: date, end: date, feed: str) -> dict[str, pd.DataFrame]:
        periods = []
        for year in range(start.year, end.year + 1):
            periods.append(
                (str(year), datetime(year, 1, 1, tzinfo=ET), datetime(year, 12, 31, 23, 59, tzinfo=ET))
            )
        data = self._cached("daily", symbols, periods, None, feed)
        lo, hi = pd.Timestamp(start, tz=ET), pd.Timestamp(end, tz=ET) + pd.Timedelta(days=1)
        return {s: df[(df.index >= lo) & (df.index < hi)] for s, df in data.items()}

    def minutes(self, symbols: list[str], start: date, end: date, feed: str) -> dict[str, pd.DataFrame]:
        periods = []
        cur = date(start.year, start.month, 1)
        while cur <= end:
            nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
            periods.append(
                (
                    cur.strftime("%Y-%m"),
                    datetime.combine(cur, time(0), ET),
                    datetime.combine(nxt, time(0), ET) - timedelta(minutes=1),
                )
            )
            cur = nxt
        data = self._cached("minute", symbols, periods, 1, feed)
        lo, hi = pd.Timestamp(start, tz=ET), pd.Timestamp(end, tz=ET) + pd.Timedelta(days=1)
        return {s: df[(df.index >= lo) & (df.index < hi)] for s, df in data.items()}

    def opening_volumes(
        self, symbols: list[str], days: list[date], minutes: int, feed: str
    ) -> dict[str, dict[date, float]]:
        """Volume traded in the first N minutes of each given session, per symbol."""
        out: dict[str, dict[date, float]] = {s: {} for s in symbols}
        for d in days:
            start = datetime.combine(d, time(9, 30), ET)
            bars = self.fetch_bars(symbols, 1, start, start + timedelta(minutes=minutes - 1), feed)
            for s, df in bars.items():
                out.setdefault(s, {})[d] = float(df["volume"].sum())
        return out

    # -- live ----------------------------------------------------------------
    def snapshots(self, symbols: list[str], feed: str) -> dict:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockSnapshotRequest

        out = {}
        for batch in chunks(symbols, 200):
            try:
                res = self.client.get_stock_snapshot(
                    StockSnapshotRequest(symbol_or_symbols=batch, feed=DataFeed(feed))
                )
                out.update(res)
            except Exception as exc:
                log.warning("snapshot batch failed: %s", exc)
        return out

    def latest_quote(self, symbol: str, feed: str):
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestQuoteRequest

        res = self.client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed(feed))
        )
        return res[symbol]

    # -- news ----------------------------------------------------------------
    def headlines(self, symbol: str, start: datetime, end: datetime) -> list[str] | None:
        """Headlines mentioning the symbol in [start, end); None if news couldn't be checked."""
        from alpaca.data.requests import NewsRequest

        cache_file = self.cache / "news" / symbol / f"{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        try:
            res = self._with_retry(lambda: self.news_client.get_news(
                NewsRequest(symbols=symbol, start=start, end=end, limit=50, exclude_contentless=False)
            ))
            items = res.data.get("news", []) if hasattr(res, "data") else []
            heads = [n.headline for n in items]
        except Exception as exc:
            log.warning("news fetch failed for %s: %s", symbol, exc)
            return None
        if end < datetime.now(ET) - timedelta(minutes=5):
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(heads))
        return heads
