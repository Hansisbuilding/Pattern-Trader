"""Signal-level research: every setup each strategy finds, what the situation looked like, and how
it would have ended under several exit rules.

The backtester answers "what would the bot have made?". This answers "which setups are worth
taking, and how should they be managed?" without the noise of daily limits and position caps.
Each signal is filled the same conservative way as the backtester (next bar's open plus slippage,
stop checked before target), and features use only what was known at the signal bar.
"""

from __future__ import annotations

import copy
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd

from patterntrader.backtest import build_day_candidates
from patterntrader.external import External
from patterntrader.sentiment import market_mood
from patterntrader.config import ROOT, parse_time
from patterntrader.data import regular_session
from patterntrader.indicators import session_vwap
from patterntrader.regime import market_regime
from patterntrader.scanner import coarse_select, prior_daily
from patterntrader.strategies import build_strategies

log = logging.getLogger(__name__)
EVERY = ["trend_up", "trend_down", "range", "mixed"]
SECTORS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU"]  # all trade since 1998


def breadth(etf_daily: dict, spy_prior: pd.DataFrame, day: date) -> dict:
    """Market breadth before the open: sectors above their 50-day average, equal-weight vs SPY."""
    above = []
    for sym in SECTORS:
        c = prior_daily(etf_daily.get(sym, pd.DataFrame()), day).get("close")
        if c is not None and len(c) >= 50:
            above.append(float(c.iloc[-1] > c.iloc[-50:].mean()))
    rsp = prior_daily(etf_daily.get("RSP", pd.DataFrame()), day).get("close")
    spy = spy_prior["close"]
    rel = np.nan
    if rsp is not None and len(rsp) >= 21:
        rel = ((rsp.iloc[-1] / rsp.iloc[-21]) - (spy.iloc[-1] / spy.iloc[-21])) * 100
    return {"sectors_above_50d": np.mean(above) if above else np.nan, "equal_weight_vs_spy_20d": rel}
TARGETS = (1.0, 1.5, 2.0, 3.0)


def research_config(cfg, pool: int = 20):
    """Loosen the situation gates so their effect can be measured instead of assumed."""
    c = copy.deepcopy(cfg)
    for p in c.strategies.values():
        p["enabled"], p["market_regimes"], p["stock_regimes"] = True, EVERY, EVERY
        p["in_play"] = False  # measured as features instead
        p.pop("target_r", None)  # exits are measured as variants instead
        p.pop("min_rvol", None)
        p.pop("max_rvol", None)
        p["max_stop_atr"] = max(p.get("max_stop_atr", 0.5), 1.0)
        p["min_rr"] = 1.0
        for k in ("require_catalyst", "require_no_catalyst"):
            if k in p:
                p[k] = False
    c.strategies.gap_and_go["min_abs_gap_pct"] = 2.0
    c.scan["max_watchlist"], c.scan["min_score"] = pool, 0
    c.risk["min_stop_pct"] = 0.15  # keep tight-stop setups in the sample; stop_pct is a feature
    return c


def outcomes(bars: pd.DataFrame, i: int, stop: float, target: float, slip: float, flatten) -> dict | None:
    """R multiples for one long signal on bar i under each exit rule (entry at bar i+1's open)."""
    if i + 1 >= len(bars):
        return None
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    t = bars.index
    fill = o[i + 1] * (1 + slip)
    risk = fill - stop
    if risk <= 0:
        return None
    end = i + 1
    while end + 1 < len(bars) and t[end + 1].time() < flatten:
        end += 1
    r = lambda px: (px - fill) / risk  # noqa: E731
    stop_px = lambda j, s: min(o[j], s) * (1 - slip)  # noqa: E731

    def fixed(tgt, max_bars=None):
        for j in range(i + 1, end + 1):
            if l[j] <= stop:
                return r(stop_px(j, stop))
            if tgt is not None and h[j] > tgt:
                return r(max(o[j], tgt))
            if max_bars and j - i >= max_bars:
                return r(c[j] * (1 - slip))
        return r(c[end] * (1 - slip))

    def managed(tgt, trail):
        """Stop to breakeven once +1R is reached (trail=False) or trail 1R behind the high (True)."""
        s, peak = stop, fill
        for j in range(i + 1, end + 1):
            if l[j] <= s:
                return r(stop_px(j, s))
            if tgt is not None and h[j] > tgt:
                return r(max(o[j], tgt))
            peak = max(peak, h[j])
            if peak - fill >= risk:
                s = max(s, peak - risk) if trail else max(s, fill)
        return r(c[end] * (1 - slip))

    seg_h, seg_l = h[i + 1 : end + 1], l[i + 1 : end + 1]
    out = {f"r_t{k:g}": fixed(fill + k * risk) for k in TARGETS}
    out.update(
        r_orig=fixed(target),
        r_hold=fixed(None),
        r_be2=managed(fill + 2 * risk, trail=False),
        r_trail=managed(None, trail=True),
        r_t2_60m=fixed(fill + 2 * risk, max_bars=60),
        mfe_r=r(seg_h.max()),
        mae_r=r(seg_l.min()),
        fill=fill,
    )
    return out


def _features(ctx, bars, i, spy_bars, daily_prior, spy_daily_prior) -> dict:
    row = bars.iloc[i]
    close_prior = daily_prior["close"]
    spy_close = spy_daily_prior["close"]
    f = {
        "minute": i,
        "ret_open": (row.close / bars["open"].iloc[0] - 1) * 100,
        "ret_prev": (row.close / ctx.prev_close - 1) * 100,
        "vwap_dist_atr": (row.close - row.vwap) / ctx.daily_atr,
        "range_atr": (bars["high"].iloc[: i + 1].max() - bars["low"].iloc[: i + 1].min()) / ctx.daily_atr,
        "ret_5d": (close_prior.iloc[-1] / close_prior.iloc[-6] - 1) * 100,
        "ret_20d": (close_prior.iloc[-1] / close_prior.iloc[-21] - 1) * 100,
        "off_20d_high": (close_prior.iloc[-1] / daily_prior["high"].iloc[-20:].max() - 1) * 100,
        "spy_ret_5d": (spy_close.iloc[-1] / spy_close.iloc[-6] - 1) * 100,
        "spy_ret_20d": (spy_close.iloc[-1] / spy_close.iloc[-21] - 1) * 100,
        "spy_ret_open": np.nan,
        "spy_above_vwap": np.nan,
    }
    if spy_bars is not None:
        s = spy_bars[spy_bars.index <= bars.index[i]]
        if len(s):
            f["spy_ret_open"] = (s["close"].iloc[-1] / s["open"].iloc[0] - 1) * 100
            f["spy_above_vwap"] = float(s["close"].iloc[-1] > s["vwap"].iloc[-1])
    f["rs_open"] = f["ret_open"] - f["spy_ret_open"]
    return f


def collect_range(cfg, symbols: list[str], start: date, end: date) -> pd.DataFrame:
    """Every first signal per symbol, strategy and day in [start, end], with features and outcomes."""
    from patterntrader.config import paper_credentials
    from patterntrader.data import MarketData

    md = MarketData(*paper_credentials(), cfg.data.cache_dir)
    ext = External(cfg.data.cache_dir, filings_max_age_hours=24 * 365)
    ext.limit_insider(symbols)
    ext.earnings_calls_left = 0  # workers only read cached earnings; scripts/fetch_earnings.py fills it
    ext.score_headlines = False  # headlines were scored once in the main process
    strategies = {s.name: s for s in build_strategies(cfg)}
    spy, feed = cfg.regime.market_symbol, cfg.data.backtest_feed
    slip = cfg.backtest.slippage_bps / 10_000
    flatten = parse_time(cfg.session.flatten_at)
    no_before, no_after = parse_time(cfg.session.no_entry_before), parse_time(cfg.session.no_entry_after)
    daily = md.daily(sorted(set(symbols) | {spy}), start - timedelta(days=400), end, feed)
    spy_d = daily[spy]
    etfs = md.daily(SECTORS + ["RSP"], start - timedelta(days=400), end, feed)
    spy_m = md.minutes([spy], start, end, feed).get(spy, pd.DataFrame())
    rows = []
    for day in [d for d in sorted(set(spy_d.index.date)) if start <= d <= end]:
        spy_prior, spy_today = prior_daily(spy_d, day), spy_d[spy_d.index.date == day]
        spy_gap = (spy_today["open"].iloc[0] / spy_prior["close"].iloc[-1] - 1) * 100
        regime, _ = market_regime(spy_prior, spy_gap, cfg.regime)
        if regime == "chaos":
            continue
        cands = []
        for sym in symbols:
            d = daily.get(sym)
            if d is None:
                continue
            prior, today = prior_daily(d, day), d[d.index.date == day]
            if len(prior) < 60 or today.empty:
                continue
            open_px, prev_close = float(today["open"].iloc[0]), float(prior["close"].iloc[-1])
            cands.append({
                "symbol": sym, "price": open_px, "prev_close": prev_close,
                "gap_pct": (open_px / prev_close - 1) * 100,
                "avg_dollar_volume": float((prior["close"] * prior["volume"]).iloc[-20:].mean()),
            })
        cands = coarse_select(cands, cfg)
        lookback = sorted(set(spy_prior.index.date))[-12]
        minutes = md.minutes([r["symbol"] for r in cands], lookback, day, feed)
        spy_bars = regular_session(spy_m, day).copy() if not spy_m.empty else None
        if spy_bars is not None and len(spy_bars):
            spy_bars["vwap"] = session_vwap(spy_bars)
        min_stop_pct = cfg.risk.min_stop_pct
        market = breadth(etfs, spy_prior, day)
        mood = market_mood(ext, spy_d, day)
        market["mood"] = mood["mood"]
        market.update({f"mood_{k}": v["greed_score"] for k, v in mood["parts"].items()})
        for ctx, bars in build_day_candidates(day, cands, daily, minutes, regime, cfg, strategies, md, ext):
            todo = set(ctx.eligible)
            for i in range(len(bars) - 1):
                ts = bars.index[i]
                if not todo or ts.time() > no_after:
                    break
                if ts.time() < no_before:
                    continue
                hist = bars.iloc[: i + 1]
                for name in sorted(todo):
                    sig = strategies[name].signal(ctx, hist)
                    if sig is None:
                        continue
                    todo.discard(name)
                    if sig.side != "long" or sig.risk_per_share < max(0.02, sig.entry * min_stop_pct / 100):
                        continue
                    res = outcomes(bars, i, sig.stop, sig.target, slip, flatten)
                    if res is None:
                        continue
                    rows.append({
                        "day": day, "symbol": ctx.symbol, "strategy": name, "time": ts.strftime("%H:%M"),
                        "market_regime": regime, "stock_regime": ctx.stock_regime,
                        "gap_pct": ctx.gap_pct, "rvol": ctx.rvol, "atr_pct": ctx.atr_pct, "adx": ctx.adx,
                        "has_catalyst": ctx.has_catalyst, "score": ctx.score,
                        "stop_atr": sig.risk_per_share / ctx.daily_atr,
                        "stop_pct": sig.risk_per_share / sig.entry * 100,
                        "orig_rr": sig.reward_risk,
                        "filings": ",".join(ctx.filings), "earnings": "earnings" in ctx.filings,
                        "sec_8k": any(f not in ("10-Q", "10-K", "offering") for f in ctx.filings),
                        "offering": "offering" in ctx.filings,
                        "vix": ctx.vix, "vix_ratio": ctx.vix_ratio, "events": ",".join(ctx.events),
                        **ctx.info,
                        **market,
                        **_features(ctx, bars, i, spy_bars, prior_daily(daily[ctx.symbol], day), spy_prior),
                        **res,
                    })
    return pd.DataFrame(rows)


def _chunk(args):
    cfg, symbols, start, end = args
    logging.basicConfig(level=logging.WARNING)
    df = collect_range(cfg, symbols, start, end)
    log.warning("signals %s → %s: %d", start, end, len(df))
    return df


def _prescore_headlines(ext) -> None:
    """Score every cached headline once with FinBERT (a few minutes) so workers only read scores."""
    import glob
    import json

    heads = set()
    for f in glob.glob(str(ext.dir.parent / "news" / "*" / "*.json")):
        heads.update(json.loads(open(f).read()))
    scorer = ext.scorer()
    todo = [h for h in heads if scorer.key(h) not in scorer.scores]
    log.warning("scoring %d new headlines (%d cached)", len(todo), len(heads) - len(todo))
    scorer.score_many(todo)


def collect(cfg, symbols: list[str], start: date, end: date, workers: int = 12, months: int = 3) -> pd.DataFrame:
    """collect_range split into chunks of `months` and run in parallel processes."""
    from patterntrader.config import paper_credentials
    from patterntrader.data import MarketData

    md = MarketData(*paper_credentials(), cfg.data.cache_dir)
    # Warm every shared cache once so parallel workers only read from disk.
    md.minutes([cfg.regime.market_symbol], start, end, cfg.data.backtest_feed)
    md.daily(SECTORS + ["RSP"], start - timedelta(days=400), end, cfg.data.backtest_feed)
    ext = External(cfg.data.cache_dir, filings_max_age_hours=24 * 365)
    ext.vix(), ext.events(), ext.insider_trades()
    for idx in ("VIX3M", "VIX9D", "SKEW"):
        ext.cboe(idx)
    ext.aaii()
    _prescore_headlines(ext)
    ext.attention().articles(symbols)
    for sym in symbols:
        ext.attention().page_views(sym, max_age_hours=24 * 365)
    for sid in ext.FRED_SERIES.values():
        ext.fred(sid)
    for sym in symbols:
        ext.filings(sym), ext.short_interest(sym), ext.shares_outstanding(sym, end)
    d = start
    while d <= end:  # FINRA daily short volume
        if d.weekday() < 5:
            ext._short_volume_day(d)
        d += timedelta(days=1)
    chunks, cur = [], start
    while cur <= end:
        nxt = (pd.Timestamp(cur) + pd.DateOffset(months=months)).date()
        chunks.append((cfg, symbols, cur, min(nxt - timedelta(days=1), end)))
        cur = nxt
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_chunk, chunks))
    return pd.concat([p for p in parts if not p.empty], ignore_index=True)


def save(df: pd.DataFrame, name: str):
    out = ROOT / "results" / "research"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.parquet"
    df.to_parquet(path)
    return path
