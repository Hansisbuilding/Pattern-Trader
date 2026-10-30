"""Sentiment: is the news good or bad, are analysts turning, is the crowd paying attention, and what
mood is the whole market in today?

- Headline tone: FinBERT (a finance-trained language model) run locally on the Alpaca headlines.
- Analyst actions: upgrades, downgrades, initiations and price-target changes parsed from headlines.
- Attention: Wikipedia page views (daily since 2015) and, live only, Reddit mentions (ApeWisdom).
- Market mood: a 0-100 fear/greed reading from VIX, VIX term structure, credit spreads,
  the AAII investor survey and SPY's trend, each ranked against its own past year.

Wikimedia asks scripts to identify themselves, so its requests carry SEC_USER_AGENT from .env.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from patterntrader.config import ROOT
from patterntrader.external import _get, _user_agent

log = logging.getLogger(__name__)

# Market roundups list many tickers; their tone says nothing about any one stock.
ROUNDUP = re.compile(
    r"stocks? (moving|to watch|that|making)|movers|mid-day|pre-market session|after-hours|"
    r"market update|stocks hitting|biggest|trending|session'?s? (top|biggest)|here'?s why .* (are|is) (trading|moving)",
    re.I,
)
UPGRADE = re.compile(r"\bupgrades?\b|\bupgraded\b", re.I)
DOWNGRADE = re.compile(r"\bdowngrades?\b|\bdowngraded\b", re.I)
INITIATE = re.compile(r"initiates? coverage|\binitiated\b|assumes coverage", re.I)
PT_UP = re.compile(r"raises? (its )?price target|\bpt (raised|to \$[\d.]+ from \$[\d.]+ \(up)|price target raised", re.I)
PT_DOWN = re.compile(r"lowers? (its )?price target|cuts? (its )?price target|price target (lowered|cut)", re.I)


def analyst_actions(headlines: list[str]) -> dict:
    """Counts of analyst actions in the headlines (Benzinga's 'X Upgrades Y to Buy' style)."""
    out = {"upgrades": 0, "downgrades": 0, "initiations": 0, "pt_raises": 0, "pt_cuts": 0}
    for h in headlines:
        if ROUNDUP.search(h):
            continue
        out["upgrades"] += bool(UPGRADE.search(h))
        out["downgrades"] += bool(DOWNGRADE.search(h))
        out["initiations"] += bool(INITIATE.search(h))
        out["pt_raises"] += bool(PT_UP.search(h))
        out["pt_cuts"] += bool(PT_DOWN.search(h))
    return out


class HeadlineScorer:
    """FinBERT tone per headline, cached on disk by headline text so each is scored once."""

    def __init__(self, cache_dir: str | Path = ".cache"):
        self.path = ROOT / cache_dir / "sentiment" / "headline_scores.parquet"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.scores: dict[str, float] = {}
        if self.path.exists():
            df = pd.read_parquet(self.path)
            self.scores = dict(zip(df.key, df.score))
        self._model = None
        self._dirty = 0

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.strip().encode()).hexdigest()[:16]

    def _clf(self):
        if self._model is None:
            import os

            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from transformers import pipeline

            self._model = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
        return self._model

    def score_many(self, headlines: list[str], batch_size: int = 64) -> list[float]:
        """Positive minus negative probability, from -1 (bad news) to +1 (good news)."""
        todo = list({h for h in headlines if self.key(h) not in self.scores})
        for i in range(0, len(todo), batch_size):
            chunk = todo[i : i + batch_size]
            for h, res in zip(chunk, self._clf()(chunk, batch_size=batch_size, truncation=True)):
                p = {d["label"]: d["score"] for d in res}
                self.scores[self.key(h)] = float(p.get("positive", 0) - p.get("negative", 0))
            self._dirty += len(chunk)
            if self._dirty >= 2000:
                self.save()
        if self._dirty:
            self.save()
        return [self.scores[self.key(h)] for h in headlines]

    def save(self) -> None:
        pd.DataFrame({"key": list(self.scores), "score": list(self.scores.values())}).to_parquet(self.path)
        self._dirty = 0

    def summarize(self, headlines: list[str], model: bool = True) -> dict:
        """Tone of the stock's own headlines (roundups dropped). model=False only reads the cache."""
        own = [h for h in headlines if not ROUNDUP.search(h)]
        out = {"news_count": len(own), "news_tone": None, "news_tone_min": None, **analyst_actions(headlines)}
        if not own:
            return out
        if model:
            s = self.score_many(own)
        else:
            s = [self.scores[k] for k in map(self.key, own) if k in self.scores]
        if s:
            out["news_tone"], out["news_tone_min"] = float(np.mean(s)), float(min(s))
        return out


class Attention:
    """Wikipedia page views (history) and Reddit mentions (live only)."""

    def __init__(self, cache_dir: str | Path = ".cache"):
        self.dir = ROOT / cache_dir / "sentiment"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._views: dict[str, pd.Series] = {}

    def _wiki_headers(self) -> dict:
        return {"User-Agent": _user_agent(), "Accept": "application/json"}

    def articles(self, symbols: list[str]) -> dict[str, str]:
        """Ticker -> English Wikipedia article, from Wikidata's stock-exchange ticker statements."""
        import requests

        p = self.dir / "wiki_articles.json"
        known = json.loads(p.read_text()) if p.exists() else {}
        todo = [s for s in symbols if s not in known]
        for i in range(0, len(todo), 40):
            chunk = todo[i : i + 40]
            values = " ".join(f'"{s}"' for s in chunk)
            q = f"""SELECT ?ticker ?article WHERE {{
              VALUES ?ticker {{ {values} }}
              VALUES ?exchange {{ wd:Q13677 wd:Q82059 wd:Q846626 }}  # NYSE, Nasdaq, NYSE American
              ?item p:P414 ?st . ?st ps:P414 ?exchange ; pq:P249 ?ticker .
              ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }}"""
            r = requests.get("https://query.wikidata.org/sparql", params={"query": q, "format": "json"},
                             headers=self._wiki_headers(), timeout=60)
            r.raise_for_status()
            for b in r.json()["results"]["bindings"]:
                t = b["ticker"]["value"]
                known.setdefault(t, b["article"]["value"].rsplit("/", 1)[-1])
            for s in chunk:
                known.setdefault(s, "")  # remember misses too
        p.write_text(json.dumps(known, indent=1, sort_keys=True))
        return {k: v for k, v in known.items() if v}

    def page_views(self, symbol: str, max_age_hours: float = 20) -> pd.Series | None:
        """Daily English Wikipedia views of the company's article since mid-2015."""
        import time as _t

        import requests

        if symbol in self._views:
            return self._views[symbol]
        title = self.articles([symbol]).get(symbol)
        if not title:
            return None
        p = self.dir / "wiki_views" / f"{symbol}.csv"
        if not (p.exists() and _t.time() - p.stat().st_mtime < max_age_hours * 3600):
            end = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
            url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
                   f"{title}/daily/2015070100/{end}00")
            r = requests.get(url, headers=self._wiki_headers(), timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            items = r.json().get("items", [])
            p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"date": [i["timestamp"][:8] for i in items], "views": [i["views"] for i in items]}).to_csv(p, index=False)
        df = pd.read_csv(p, dtype={"date": str})
        s = pd.Series(df["views"].values, index=pd.to_datetime(df["date"], format="%Y%m%d").dt.date)
        self._views[symbol] = s
        return s

    def wiki_attention(self, symbol: str, day: date, max_age_hours: float = 20) -> dict:
        """Yesterday's views vs the prior 30-day median (a full day of views is known by the open)."""
        s = self.page_views(symbol, max_age_hours)
        if s is None:
            return {"wiki_views_ratio": None}
        prior = s[s.index < day]
        if len(prior) < 31 or prior.iloc[-31:-1].median() <= 0:
            return {"wiki_views_ratio": None}
        return {"wiki_views_ratio": float(prior.iloc[-1] / prior.iloc[-31:-1].median())}

    def reddit_mentions(self) -> dict[str, dict]:
        """Live only: ticker -> 24h Reddit mentions and the prior 24h (ApeWisdom, no key needed)."""
        out, page, pages = {}, 1, 1
        while page <= min(pages, 10):
            j = json.loads(_get(f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}"))
            pages = j.get("pages", 1)
            for r in j.get("results", []):
                out[r["ticker"]] = {"reddit_mentions": r.get("mentions"), "reddit_mentions_prior": r.get("mentions_24h_ago"),
                                    "reddit_rank": r.get("rank")}
            page += 1
        return out


# -- market mood -------------------------------------------------------------------
MOOD_PARTS = {
    # name: (higher value means..., label)
    "vix": ("fear", "VIX level"),
    "vix_term": ("fear", "VIX vs 3-month VIX (>1 = panic)"),
    "hy_spread": ("fear", "junk-bond credit spread"),
    "aaii_spread": ("greed", "AAII bulls minus bears"),
    "spy_vs_125d": ("greed", "SPY vs its 125-day average"),
}


def mood_inputs(ext, spy_daily: pd.DataFrame, day: date) -> pd.DataFrame:
    """Daily history (strictly before `day`) of every mood input, aligned by date."""
    cols = {}
    vix = ext.cboe("VIX")["close"]
    cols["vix"] = vix
    cols["vix_term"] = vix / ext.cboe("VIX3M")["close"]
    hy = ext.fred("BAMLH0A0HYM2")
    if hy is not None:
        cols["hy_spread"] = hy
    aaii = ext.aaii()
    if aaii is not None:
        cols["aaii_spread"] = aaii["bull_bear"]
    c = spy_daily["close"].copy()
    c.index = spy_daily.index.date
    cols["spy_vs_125d"] = c / c.rolling(125).mean() - 1
    df = pd.DataFrame(cols).sort_index().ffill()  # weekly AAII and holiday gaps carry forward
    return df[df.index < day]


def market_mood(ext, spy_daily: pd.DataFrame, day: date) -> dict:
    """0 = extreme fear, 100 = extreme greed: each input's percentile over the past year, averaged."""
    df = mood_inputs(ext, spy_daily, day).iloc[-260:]
    if len(df) < 60:
        return {"mood": None, "mood_label": "unknown", "parts": {}}
    parts = {}
    for name, (direction, label) in MOOD_PARTS.items():
        if name not in df or df[name].dropna().empty:
            continue
        s = df[name].dropna()
        pct = float((s <= s.iloc[-1]).mean() * 100)
        parts[name] = {"label": label, "value": float(s.iloc[-1]),
                       "greed_score": 100 - pct if direction == "fear" else pct}
    mood = float(np.mean([p["greed_score"] for p in parts.values()])) if parts else None
    return {"mood": mood, "mood_label": mood_label(mood), "parts": parts}


def mood_label(m: float | None) -> str:
    if m is None:
        return "unknown"
    return ("extreme fear" if m < 25 else "fear" if m < 45 else "neutral" if m <= 55
            else "greed" if m <= 75 else "extreme greed")


def render_mood(m: dict) -> list[str]:
    if m.get("mood") is None:
        return ["Market mood: unavailable"]
    lines = [f"**Market mood: {m['mood']:.0f}/100 ({m['mood_label']})**", ""]
    for p in m["parts"].values():
        lines.append(f"- {p['label']}: {p['value']:.2f} → {p['greed_score']:.0f}/100")
    return lines


def daily_mood(cfg, md, day: date, ext=None) -> dict:
    """Today's market mood, logged to journal/mood/<day>.json so live-only readings build a history."""
    from patterntrader.external import External

    ext = ext or External(cfg.data.cache_dir)
    spy = cfg.regime.market_symbol
    spy_d = md.daily([spy], day - timedelta(days=400), day - timedelta(days=1), cfg.data.backtest_feed)[spy]
    m = market_mood(ext, spy_d, day)
    m["day"] = day.isoformat()
    m["vix"] = ext.vix_context(day)
    m["rates"] = ext.rates_context(day)
    m["events"] = ext.events_on(day)
    try:
        reddit = ext.attention().reddit_mentions()
        m["reddit_top"] = sorted(reddit.items(), key=lambda kv: kv[1].get("reddit_rank") or 9999)[:25]
    except Exception as exc:
        m["reddit_top"] = []
        log.warning("Reddit mentions unavailable: %s", exc)
    out = ROOT / "journal" / "mood"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{day.isoformat()}.json").write_text(json.dumps(m, indent=1, default=str))
    return m


# -- rules: how the bot acts on sentiment (config: sentiment) ------------------------------
def catalyst_quality(ctx, cfg) -> tuple[float, dict]:
    """0-100 (50 = neutral): how good is the news behind this stock today? Points per config."""
    w, info = cfg.sentiment.quality, ctx.info
    parts: dict[str, float] = {}
    if info.get("news_tone") is not None:
        parts["news tone"] = w.news_tone * info["news_tone"]
    if info.get("eps_surprise_pct") is not None:
        parts["EPS surprise"] = w.eps_surprise * max(-1.0, min(1.0, info["eps_surprise_pct"] / 10))
    net = (info.get("upgrades", 0) + info.get("pt_raises", 0)) - (info.get("downgrades", 0) + info.get("pt_cuts", 0))
    if net:
        parts["analysts"] = w.analyst * max(-2, min(2, net))
    if info.get("insider_buys_30d"):
        parts["insider buying"] = w.insider_buy
    if (info.get("short_pct_out") or 0) >= w.squeeze_short_pct:
        parts["short interest (squeeze fuel)"] = w.squeeze
    if (info.get("short_volume_5d") or 0) > 0.5:
        parts["heavy short selling"] = w.get("short_volume", 0)
    if "earnings" in (ctx.filings or []):
        parts["earnings report"] = w.get("earnings", 0)
    if (info.get("wiki_views_ratio") or 0) >= 2:
        parts["attention spike"] = w.attention
    if "offering" in (ctx.filings or []):
        parts["offering filed"] = -w.offering
    parts = {k: v for k, v in parts.items() if v}
    return max(0.0, min(100.0, 50 + sum(parts.values()))), parts


def setup_size_factor(ctx, cfg) -> float:
    """Half size for setups without any catalyst-quality edge, when configured."""
    s = cfg.get("sentiment", {})
    cut = s.get("half_size_below_quality")
    if not s.get("enabled") or cut is None:
        return 1.0
    return 0.5 if ctx.info.get("catalyst_quality", 50) < cut else 1.0


def stock_gate(ctx, cfg) -> str | None:
    """Reason to skip this stock today on sentiment alone, or None."""
    s = cfg.sentiment
    tone = ctx.info.get("news_tone")
    if tone is not None and s.block_news_tone_below is not None and tone < s.block_news_tone_below:
        return f"news tone {tone:+.2f} is bad news (< {s.block_news_tone_below:+.2f})"
    if s.block_offering and "offering" in (ctx.filings or []):
        return "offering filed before the open: new shares tend to cap a gap"
    if s.min_quality is not None and ctx.info.get("catalyst_quality", 50) < s.min_quality:
        return f"catalyst quality {ctx.info['catalyst_quality']:.0f} < {s.min_quality}"
    return None


def apply_to_setup(ctx, cfg) -> None:
    """Score the catalyst; when sentiment is on, gate the setup and fold quality into the rank."""
    q, parts = catalyst_quality(ctx, cfg)
    ctx.info["catalyst_quality"], ctx.info["quality_parts"] = q, parts
    ctx.info["eligible_without_sentiment"] = list(ctx.eligible)
    if not cfg.get("sentiment", {}).get("enabled") or not ctx.eligible:
        return
    why = stock_gate(ctx, cfg)
    if why:
        ctx.info["sentiment_block"] = why
        for name in ctx.eligible:
            ctx.rejected[name] = f"sentiment: {why}"
        ctx.eligible = []
        ctx.score = 0.0
        return
    ctx.score = round(ctx.score + (q - 50) * cfg.sentiment.rank_weight, 1)


def day_size_factor(mood: float | None, vix_term: float | None, events: list[str], cfg) -> tuple[float, list[str]]:
    """Fraction of normal position size for today (1 = normal, 0 = sit out) and why."""
    s = cfg.sentiment
    if not s.get("enabled"):
        return 1.0, []
    factor, why = 1.0, []
    if vix_term is not None and s.sit_out_vix_term_above is not None and vix_term > s.sit_out_vix_term_above:
        return 0.0, [f"VIX term structure {vix_term:.2f} > {s.sit_out_vix_term_above}: panic, sit out"]
    if mood is not None and s.half_size_below_mood is not None and mood < s.half_size_below_mood:
        factor *= 0.5
        why.append(f"market mood {mood:.0f} < {s.half_size_below_mood}: half size")
    hits = [e for e in events if e in (s.half_size_events or [])]
    if hits:
        factor *= 0.5
        why.append(f"{', '.join(hits)} today: half size")
    return factor, why


def log_decisions(day: date, stage: str, contexts: list, factor: float, reasons: list[str]) -> Path:
    """journal/sentiment_decisions.csv: what sentiment changed, for the monthly review."""
    path = ROOT / "journal" / "sentiment_decisions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"day": day.isoformat(), "stage": stage, "symbol": "*market*", "size_factor": factor,
             "action": "; ".join(reasons) or "normal size"}]
    for c in contexts:
        before = c.info.get("eligible_without_sentiment", [])
        if not before:
            continue
        rows.append({
            "day": day.isoformat(), "stage": stage, "symbol": c.symbol, "size_factor": factor,
            "catalyst_quality": round(c.info.get("catalyst_quality", 50), 1),
            "quality_parts": "; ".join(f"{k} {v:+.0f}" for k, v in c.info.get("quality_parts", {}).items()),
            "news_tone": c.info.get("news_tone"), "eps_surprise_pct": c.info.get("eps_surprise_pct"),
            "action": f"blocked: {c.info['sentiment_block']}" if c.info.get("sentiment_block")
                      else "kept" if c.eligible else "dropped by the baseline rules",
            "strategies_before": ",".join(before), "strategies_after": ",".join(c.eligible),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path
