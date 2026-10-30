"""Live scanner: full-market pass for stocks in play, then a disciplined situation check."""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta

import pandas as pd

from patterntrader.config import ET
from patterntrader.context import add_outside_data, build_context, evaluate
from patterntrader.models import SetupContext
from patterntrader.regime import market_regime

log = logging.getLogger(__name__)


def prior_daily(df: pd.DataFrame, day: date) -> pd.DataFrame:
    return df[df.index.date < day]


def coarse_select(rows: list[dict], cfg) -> list[dict]:
    """Liquid, tradable price, and either gapping or among the most liquid names."""
    sc = cfg.scan
    ok = [
        r for r in rows
        if sc.min_price <= r["price"] <= sc.max_price and r["avg_dollar_volume"] >= sc.min_avg_dollar_volume
    ]
    gappers = sorted(
        (r for r in ok if abs(r["gap_pct"]) >= sc.min_abs_gap_pct), key=lambda r: -abs(r["gap_pct"])
    )[: sc.coarse_limit]
    liquid = sorted(ok, key=lambda r: -r["avg_dollar_volume"])[: sc.get("include_top_liquid", 0)]
    seen, out = set(), []
    for r in gappers + liquid:
        if r["symbol"] not in seen:
            seen.add(r["symbol"])
            out.append(r)
    return out


def coarse_universe(cfg, md, trading, day: date) -> list[dict]:
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    assets = trading.get_all_assets(
        GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    )
    exchanges = set(cfg.scan.exchanges)
    assets = {
        a.symbol: a for a in assets
        if a.tradable and getattr(a.exchange, "value", str(a.exchange)) in exchanges and a.symbol.isalpha()
    }
    log.info("universe: %d tradable stocks", len(assets))
    snaps = md.snapshots(list(assets), cfg.data.live_feed)
    # Liquidity comes from consolidated (SIP) daily history; IEX volume alone understates it ~30x.
    start = datetime.combine(day - timedelta(days=35), time(0), ET)
    end = datetime.combine(day - timedelta(days=1), time(23, 59), ET)
    daily = md.fetch_bars(list(snaps), None, start, end, cfg.data.backtest_feed)
    rows = []
    for sym, snap in snaps.items():
        d = daily.get(sym)
        trade = getattr(snap, "latest_trade", None)
        if d is None or d.empty or trade is None:
            continue
        d = prior_daily(d, day)
        if d.empty:
            continue
        prev_close = float(d["close"].iloc[-1])
        traded_today = trade.timestamp.astimezone(ET).date() == day
        price = float(trade.price) if traded_today else prev_close
        a = assets[sym]
        rows.append(
            {
                "symbol": sym,
                "price": price,
                "prev_close": prev_close,
                "gap_pct": (price / prev_close - 1) * 100,
                "avg_dollar_volume": float((d["close"] * d["volume"]).iloc[-20:].mean()),
                "shortable": bool(a.shortable and a.easy_to_borrow),
            }
        )
    return coarse_select(rows, cfg)


def analyze(cfg, md, rows: list[dict], now: datetime, strategies, edge) -> tuple[str, list[str], list[SetupContext]]:
    """Full situation analysis for the coarse candidates at time `now`."""
    day = now.date()
    spy = cfg.regime.market_symbol
    symbols = [r["symbol"] for r in rows]
    daily = md.daily(symbols + [spy], day - timedelta(days=260), day - timedelta(days=1), cfg.data.backtest_feed)
    spy_daily = prior_daily(daily.get(spy, pd.DataFrame()), day)
    if spy_daily.empty:
        raise RuntimeError("no SPY history: cannot judge the market regime")

    feed = cfg.data.live_feed
    rv_min = cfg.session.rvol_minutes
    opened = now.time() >= time(9, 30 + rv_min)
    session_start = datetime.combine(day, time(4, 0), ET)
    today = md.fetch_bars(symbols + [spy], 1, session_start, now, feed) if now.time() >= time(4, 0) else {}

    def open_price(sym):
        df = today.get(sym)
        if not opened or df is None:
            return None
        reg = df[df.index.time >= time(9, 30)]
        return float(reg["open"].iloc[0]) if len(reg) else None

    spy_prev = float(spy_daily["close"].iloc[-1])
    spy_today = today.get(spy)
    spy_px = open_price(spy) or (float(spy_today["close"].iloc[-1]) if spy_today is not None and len(spy_today) else spy_prev)
    regime, notes = market_regime(spy_daily, (spy_px / spy_prev - 1) * 100, cfg.regime)

    today_vols, past_vols = {}, {}
    if opened:
        past_days = [d for d in sorted(set(spy_daily.index.date))][-10:]
        past_vols = md.opening_volumes(symbols, past_days, rv_min, feed)
        today_vols = {s: v[day] for s, v in md.opening_volumes(symbols, [day], rv_min, feed).items() if day in v}

    from patterntrader.backtest import outside_data

    ext = outside_data(cfg)
    if ext is not None:
        try:
            v, ev = ext.vix_context(day), ext.events_on(day)
            if v["vix"] is not None:
                notes.append(f"VIX {v['vix']:.1f} ({v['vix_ratio']:.2f}x its 20-day average)")
            if ev:
                notes.append("macro release today: " + ", ".join(ev))
            rates = ext.rates_context(day)
            if rates.get("hy_spread") is not None:
                notes.append(f"credit spread {rates['hy_spread']:.2f}% ({rates['hy_spread_20d']:+.2f} in 20 days), "
                             f"10-year yield {rates['t10y']:.2f}%")
        except Exception as exc:
            notes.append(f"VIX/calendar unavailable ({str(exc)[:60]})")
        try:
            halts = ext.halts_today()
        except Exception as exc:
            halts = {}
            notes.append(f"halt feed unavailable ({str(exc)[:60]})")
        try:
            reddit = ext.attention().reddit_mentions()
        except Exception as exc:
            reddit = {}
            notes.append(f"Reddit mentions unavailable ({str(exc)[:60]})")
    contexts = []
    for r in rows:
        sym = r["symbol"]
        d = prior_daily(daily.get(sym, pd.DataFrame()), day)
        pre = today.get(sym)
        pre = pre[pre.index.time < time(9, 30)] if pre is not None else None
        headlines = None
        if abs(r["gap_pct"]) >= cfg.scan.min_abs_gap_pct:
            headlines = md.headlines(sym, now - timedelta(hours=cfg.scan.news_lookback_hours), now)
        ctx = build_context(
            sym, day, d, r["price"], regime, cfg,
            open_price=open_price(sym),
            today_open_volume=today_vols.get(sym),
            past_open_volumes=list(past_vols.get(sym, {}).values()),
            today_premarket=pre,
            headlines=headlines,
            shortable=r.get("shortable", False),
        )
        if ctx:
            if ext is not None and abs(r["gap_pct"]) >= cfg.scan.min_abs_gap_pct:
                add_outside_data(ctx, ext, cfg.scan.news_lookback_hours)
            if ext is not None and sym in halts:
                ctx.info["halted"] = halts[sym]
            if ext is not None and sym in reddit:
                ctx.info.update(reddit[sym])
            contexts.append(evaluate(ctx, strategies, cfg, edge))
    contexts.sort(key=lambda c: -c.score)
    return regime, notes, contexts


def pick_watchlist(contexts: list[SetupContext], cfg) -> list[SetupContext]:
    return [c for c in contexts if c.eligible and c.score >= cfg.scan.min_score][: cfg.scan.max_watchlist]
