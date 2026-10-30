"""Free outside data: SEC EDGAR filings (catalyst type, earnings dates), Cboe VIX, and the
macro event calendar (FOMC decisions from the Fed, CPI and jobs reports from the BLS).

Everything is cached under .cache/external. SEC and BLS ask scripts to identify themselves, so
requests to them carry SEC_USER_AGENT from .env; nothing else is sent anywhere.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time as _time
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from patterntrader.config import ET, ROOT

log = logging.getLogger(__name__)
UTC = "UTC"

# 8-K item codes worth knowing at the open (https://www.sec.gov/files/form8-k.pdf)
ITEM_NAMES = {
    "1.01": "material agreement", "1.03": "bankruptcy", "2.01": "acquisition or sale",
    "2.02": "earnings", "2.06": "impairment", "3.01": "delisting notice", "5.02": "executive change",
    "7.01": "Reg FD update", "8.01": "other event",
}
OFFERING_FORMS = ("424B", "S-1", "S-3", "F-3")  # prospectuses and shelf registrations: possible dilution
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], 1)}


def _env(name: str) -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    return os.environ.get(name, "").strip()


def _user_agent() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError("SEC_USER_AGENT is not set in .env (SEC and BLS require a contact name and email)")
    return ua


def _get(url: str, contact: bool = False, timeout: float = 30) -> bytes:
    import requests  # ships its own CA bundle (python.org builds on macOS have none for urllib)

    headers = {"User-Agent": _user_agent() if contact else "PatternTrader research"}
    for attempt in range(4):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as exc:  # transient network errors and SEC's 429s: back off and retry
            if attempt == 3:
                raise RuntimeError(f"download failed: {url} ({exc})") from exc
            _time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


class External:
    def __init__(self, cache_dir: str | Path = ".cache", filings_max_age_hours: float = 12):
        self.filings_max_age = filings_max_age_hours  # research passes a huge value: history doesn't change
        self.earnings_calls_left: int | None = None   # cap on Alpha Vantage calls (free tier: 25/day)
        self.score_headlines = True                   # research workers only read pre-scored headlines
        self._scorer = self._attention = None
        self.cache_name = cache_dir
        self.dir = ROOT / cache_dir / "external"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._filings: dict[str, pd.DataFrame] = {}
        self._vix: pd.DataFrame | None = None
        self._events: dict[date, list[str]] | None = None
        self._last_sec = 0.0

    def _fresh(self, path: Path, max_age_hours: float) -> bool:
        return path.exists() and _time.time() - path.stat().st_mtime < max_age_hours * 3600

    # -- SEC EDGAR -------------------------------------------------------------
    def _sec(self, url: str) -> bytes:
        wait = 0.15 - (_time.monotonic() - self._last_sec)  # SEC allows 10 requests/second
        if wait > 0:
            _time.sleep(wait)
        self._last_sec = _time.monotonic()
        return _get(url, contact=True)

    def cik_map(self) -> dict[str, int]:
        p = self.dir / "sec_company_tickers.json"
        if not self._fresh(p, 24 * 7):
            p.write_bytes(self._sec("https://www.sec.gov/files/company_tickers.json"))
        return {v["ticker"].upper(): int(v["cik_str"]) for v in json.loads(p.read_text()).values()}

    def filings(self, symbol: str) -> pd.DataFrame:
        """All EDGAR filings for a ticker: accepted (ET), form, items. Empty if the SEC doesn't know it."""
        if symbol in self._filings:
            return self._filings[symbol]
        p = self.dir / "sec_filings" / f"{symbol}.parquet"
        if self._fresh(p, self.filings_max_age):
            df = pd.read_parquet(p)
        else:
            cik = self.cik_map().get(symbol.upper().replace(".", "-"))
            df = self._download_filings(cik) if cik else pd.DataFrame(columns=["accepted", "form", "items"])
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p)
        self._filings[symbol] = df
        return df

    def _download_filings(self, cik: int) -> pd.DataFrame:
        base = "https://data.sec.gov/submissions/"
        top = json.loads(self._sec(f"{base}CIK{cik:010d}.json"))
        pages = [top["filings"]["recent"]]
        for f in top["filings"].get("files", []):
            if f.get("filingTo", "9999") >= "2015-01-01":
                pages.append(json.loads(self._sec(base + f["name"])))
        frames = [
            pd.DataFrame({"accepted": pg["acceptanceDateTime"], "form": pg["form"], "items": pg.get("items", [""] * len(pg["form"]))})
            for pg in pages
        ]
        df = pd.concat(frames, ignore_index=True)
        df["accepted"] = pd.to_datetime(df["accepted"], utc=True, errors="coerce").dt.tz_convert(ET)
        return df.dropna(subset=["accepted"]).sort_values("accepted").reset_index(drop=True)

    def catalysts(self, symbol: str, start: datetime, end: datetime) -> list[str]:
        """What the company told the SEC in [start, end): e.g. ['earnings', 'offering']."""
        df = self.filings(symbol)
        if df.empty:
            return []
        w = df[(df.accepted >= start) & (df.accepted < end)]
        out: list[str] = []
        for form, items in zip(w.form, w["items"]):
            if form.startswith("8-K"):
                codes = [c.strip() for c in str(items).split(",") if c.strip()]
                out += [ITEM_NAMES[c] for c in codes if c in ITEM_NAMES] or ["8-K"]
            elif form.startswith(OFFERING_FORMS):
                out.append("offering")
            elif form in ("10-Q", "10-K"):
                out.append(form)
            elif form.startswith("SC 13D"):
                out.append("activist stake")
        return sorted(set(out))

    # -- VIX ---------------------------------------------------------------------
    def cboe(self, index: str) -> pd.DataFrame:
        """Daily history of a Cboe index (VIX, VIX9D, VIX3M, SKEW), indexed by date."""
        cache = self.__dict__.setdefault("_cboe", {})
        if index not in cache:
            p = self.dir / f"{index.lower()}.csv"
            if not self._fresh(p, 12):
                p.write_bytes(_get(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{index}_History.csv"))
            df = pd.read_csv(p)
            df.index = pd.to_datetime(df.pop("DATE"), format="%m/%d/%Y").dt.date
            cache[index] = df.rename(columns=str.lower)
        return cache[index]

    def vix(self) -> pd.DataFrame:
        if self._vix is None:
            self._vix = self.cboe("VIX")
        return self._vix

    def aaii(self) -> pd.DataFrame | None:
        """AAII weekly investor survey since 1987: bullish, bearish and bull-minus-bear (in %)."""
        if "_aaii" not in self.__dict__:
            # AAII's site often answers scripts with a bot check. If so, save the file from a browser
            # (aaii.com/sentimentsurvey/sent_results, "Download Historical Data") as data/aaii_sentiment.xls.
            manual = ROOT / "data" / "aaii_sentiment.xls"
            p = self.dir / "aaii_sentiment.xls"
            blocked = self.dir / "aaii.blocked"
            if not self._fresh(p, 24 * 3) and not manual.exists() and not self._fresh(blocked, 24):
                try:
                    raw = _get("https://www.aaii.com/files/surveys/sentiment.xls")
                    if raw[:4] == b"\xd0\xcf\x11\xe0":  # a real Excel file, not a bot-check page
                        p.write_bytes(raw)
                    else:
                        blocked.touch()  # don't ask again today
                        log.warning("AAII sent a bot check instead of the survey; download it manually to %s", manual)
                except RuntimeError as exc:
                    log.warning("AAII survey unavailable: %s", exc)
            if manual.exists() and (not p.exists() or manual.stat().st_mtime > p.stat().st_mtime):
                p = manual
            df = None
            if p.exists():
                raw = pd.read_excel(p, header=None)
                d = pd.to_datetime(raw[0], errors="coerce")
                ok = d.notna() & pd.to_numeric(raw[1], errors="coerce").notna()
                df = pd.DataFrame({"bullish": pd.to_numeric(raw.loc[ok, 1]) * 100,
                                   "bearish": pd.to_numeric(raw.loc[ok, 3]) * 100}, )
                df.index = d[ok].dt.date
                # published on the survey date (Thursday); counted from the next session on
                df.index = [i + timedelta(days=1) for i in df.index]
                df["bull_bear"] = df.bullish - df.bearish
            self._aaii = df
        return self._aaii

    def vix_context(self, day: date) -> dict:
        """VIX as known before `day`'s open: level, vs its 20-day average, and 5-day change."""
        v = self.vix()["close"]
        v = v[v.index < day]
        if len(v) < 21:
            return {"vix": None, "vix_ratio": None, "vix_5d": None}
        out = {
            "vix": float(v.iloc[-1]),
            "vix_ratio": float(v.iloc[-1] / v.iloc[-20:].mean()),
            "vix_5d": float(v.iloc[-1] - v.iloc[-6]),
            "vix_term": None, "vix_9d_ratio": None, "skew": None,
        }
        try:  # term structure: short-dated fear above 3-month fear = panic
            for name, col in (("VIX3M", "vix_term"), ("VIX9D", "vix_9d_ratio")):
                other = self.cboe(name)["close"]
                other = other[other.index < day]
                if len(other) and other.index[-1] == v.index[-1]:
                    out[col] = float(v.iloc[-1] / other.iloc[-1]) if col == "vix_term" else float(other.iloc[-1] / v.iloc[-1])
            sk = self.cboe("SKEW")["skew"]
            sk = sk[sk.index < day]
            out["skew"] = float(sk.iloc[-1]) if len(sk) else None
        except Exception as exc:
            log.warning("Cboe term structure unavailable: %s", exc)
        return out

    # -- macro calendar ----------------------------------------------------------
    def events(self) -> dict[date, list[str]]:
        """Scheduled macro events: FOMC decision days, CPI and jobs-report release days."""
        if self._events is None:
            p = self.dir / "macro_events.json"
            if not self._fresh(p, 24 * 7):
                ev: dict[str, list[str]] = {}
                for name, days in (("FOMC", self._fomc_days()), ("CPI", self._bls_days("cpi")),
                                   ("jobs", self._bls_days("empsit"))):
                    for d in days:
                        ev.setdefault(d.isoformat(), []).append(name)
                p.write_text(json.dumps(ev, indent=1, sort_keys=True))
            self._events = {date.fromisoformat(k): v for k, v in json.loads(p.read_text()).items()}
        return self._events

    def events_on(self, day: date) -> list[str]:
        return self.events().get(day, []) + expiration_events(day)

    def _fomc_days(self) -> list[date]:
        """Last day of each scheduled meeting (the 2pm decision). Current page covers ~5 years."""
        out: set[date] = set()
        html = _get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm").decode("utf8", "ignore")
        year = None
        pat = r"(\d{4}) FOMC Meetings|fomc-meeting__month[^>]*>\s*(?:<strong>)?([A-Za-z/]+)|fomc-meeting__date[^>]*>\s*([\d\-]+)"
        month = None
        for m in re.finditer(pat, html):
            if m.group(1):
                year = int(m.group(1))
            elif m.group(2):
                month = m.group(2).split("/")[-1].lower()
            elif m.group(3) and year and month and "-" in m.group(3):
                full = next((k for k in MONTHS if k.startswith(month[:3])), None)
                if full:
                    out.add(date(year, MONTHS[full], int(m.group(3).split("-")[-1])))
        first = min(d.year for d in out) if out else date.today().year
        for y in range(2015, first):
            page = _get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm").decode("utf8", "ignore")
            for m in re.finditer(r"<h5[^>]*>\s*([A-Za-z]+)(?:/([A-Za-z]+))? (\d+)-(\d+) Meeting - (\d{4})", page):
                mon = (m.group(2) or m.group(1)).lower()
                full = next((k for k in MONTHS if k.startswith(mon[:3])), None)
                if full:
                    out.add(date(int(m.group(5)), MONTHS[full], int(m.group(4))))
        return sorted(out)

    def _bls_days(self, release: str) -> list[date]:
        """Release days from the BLS archive (past) plus its published schedule (upcoming)."""
        out: set[date] = set()
        html = _get(f"https://www.bls.gov/bls/news-release/{release}.htm", contact=True).decode("utf8", "ignore")
        for mm, dd, yyyy in re.findall(rf"{release}_(\d{{2}})(\d{{2}})(\d{{4}})\.htm", html):
            out.add(date(int(yyyy), int(mm), int(dd)))
        sched = _get(f"https://www.bls.gov/schedule/news_release/{release}.htm", contact=True).decode("utf8", "ignore")
        for mon, dd, yyyy in re.findall(r"<td>([A-Z][a-z]{2})[a-z]*\.? (\d{1,2}), (\d{4})</td>", sched):
            full = next(k for k in MONTHS if k.startswith(mon.lower()))
            out.add(date(int(yyyy), MONTHS[full], int(dd)))
        return sorted(out)

    # -- sentiment (patterntrader.sentiment) -----------------------------------------------
    def scorer(self):
        if self._scorer is None:
            from patterntrader.sentiment import HeadlineScorer

            self._scorer = HeadlineScorer(self.cache_name)
        return self._scorer

    def attention(self):
        if self._attention is None:
            from patterntrader.sentiment import Attention

            self._attention = Attention(self.cache_name)
        return self._attention

    def news_sentiment(self, headlines: list[str]) -> dict:
        return self.scorer().summarize(headlines, model=self.score_headlines)

    # -- FINRA short interest (twice a month, published ~8 business days after settlement) ----
    def short_interest(self, symbol: str) -> pd.DataFrame:
        p = self.dir / "finra_short_interest" / f"{symbol}.parquet"
        if not self._fresh(p, self.filings_max_age):
            import requests

            body = {"limit": 5000, "compareFilters": [
                {"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": symbol}]}
            r = requests.post("https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest",
                              json=body, headers={"Accept": "application/json"}, timeout=60)
            r.raise_for_status()
            rows = r.json() if r.content else []
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df[["settlementDate", "currentShortPositionQuantity", "averageDailyVolumeQuantity", "daysToCoverQuantity"]]
                df.columns = ["settlement", "short_shares", "avg_volume", "days_to_cover"]
                df["settlement"] = pd.to_datetime(df["settlement"]).dt.date
                df = df.sort_values("settlement").drop_duplicates("settlement", keep="last")
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p)
        return pd.read_parquet(p)

    def short_context(self, symbol: str, day: date) -> dict:
        """Latest short interest that was public before `day` (settlement + 12 calendar days)."""
        out = {"short_pct_out": None, "days_to_cover": None, "short_change_pct": None}
        si = self.short_interest(symbol)
        if si.empty:
            return out
        known = si[si.settlement <= day - timedelta(days=12)]
        if known.empty:
            return out
        last = known.iloc[-1]
        out["days_to_cover"] = float(last.days_to_cover) if pd.notna(last.days_to_cover) else None
        if len(known) > 1 and known.iloc[-2].short_shares:
            out["short_change_pct"] = float(last.short_shares / known.iloc[-2].short_shares * 100 - 100)
        shares = self.shares_outstanding(symbol, day)
        if shares:
            out["short_pct_out"] = float(last.short_shares / shares * 100)
        return out

    # -- FINRA daily short-sale volume (files exist from late 2018) ----------------------------
    def short_volume_ratio(self, symbol: str, day: date, days: int = 5) -> float | None:
        """Share of the last `days` sessions' volume that was sold short (FINRA reporting facilities)."""
        ratios, d = [], day
        for _ in range(days * 2 + 6):
            d -= timedelta(days=1)
            if d.weekday() >= 5:
                continue
            df = self._short_volume_day(d)
            if df is None:
                continue
            row = df[df.Symbol == symbol]
            if len(row) and row.TotalVolume.iloc[0] > 0:
                ratios.append(float(row.ShortVolume.iloc[0] / row.TotalVolume.iloc[0]))
            if len(ratios) == days:
                break
        return float(sum(ratios) / len(ratios)) if ratios else None

    def _short_volume_day(self, d: date) -> pd.DataFrame | None:
        key = d.isoformat()
        cache = self.__dict__.setdefault("_shvol", {})
        if key in cache:
            return cache[key]
        p = self.dir / "finra_short_volume" / f"{key}.parquet"
        missing = p.with_suffix(".missing")
        if p.exists():
            df = pd.read_parquet(p)
        elif missing.exists() or d < date(2018, 8, 1) or d >= date.today():
            df = None
        else:
            import requests

            r = requests.get(f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d:%Y%m%d}.txt", timeout=30)
            p.parent.mkdir(parents=True, exist_ok=True)
            if r.status_code != 200:  # holiday or not published
                missing.touch()
                df = None
            else:
                df = pd.read_csv(io.StringIO(r.text), sep="|", usecols=["Symbol", "ShortVolume", "TotalVolume"])
                df = df.dropna(subset=["Symbol"])
                df.to_parquet(p)
        if len(cache) > 60:
            cache.clear()
        cache[key] = df
        return df

    # -- SEC XBRL: shares outstanding as reported on 10-Q/10-K covers ---------------------
    def shares_outstanding(self, symbol: str, day: date) -> float | None:
        cache = self.__dict__.setdefault("_shares", {})
        if symbol not in cache:
            p = self.dir / "sec_shares" / f"{symbol}.parquet"
            if not self._fresh(p, self.filings_max_age):
                cik = self.cik_map().get(symbol.upper().replace(".", "-"))
                rows = []
                if cik:
                    try:
                        j = json.loads(self._sec(
                            f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/dei/EntityCommonStockSharesOutstanding.json"))
                        rows = j["units"]["shares"]
                    except RuntimeError:  # some filers (funds, foreign issuers) don't tag it
                        rows = []
                df = pd.DataFrame(rows or [], columns=["end", "val", "accn", "filed"])
                if not df.empty:  # several share classes in one filing: add them up
                    df = df.groupby(["accn", "filed"], as_index=False)["val"].sum()
                    df["filed"] = pd.to_datetime(df["filed"]).dt.date
                    df = df.sort_values("filed")[["filed", "val"]]
                p.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(p)
            cache[symbol] = pd.read_parquet(p)
        df = cache[symbol]
        known = df[df.filed < day] if not df.empty else df
        return float(known.val.iloc[-1]) if len(known) else None

    # -- SEC insider transactions (quarterly bulk files of every Form 3/4/5) -----------------
    def insider_trades(self) -> pd.DataFrame:
        """Open-market buys (P) and sells (S) by insiders, every issuer, 2015 onward."""
        if "_insider" in self.__dict__:
            return self._insider
        folder = self.dir / "sec_insider"
        folder.mkdir(parents=True, exist_ok=True)
        today = date.today()
        y, q = 2015, 1
        while (y, q) <= (today.year, (today.month - 1) // 3 + 1):
            p = folder / f"{y}q{q}.parquet"
            current = (y, q) == (today.year, (today.month - 1) // 3 + 1)
            missing = p.with_suffix(".missing")  # SEC publishes each quarter some weeks after it ends
            if not p.exists() and not current and not self._fresh(missing, 24 * 7):
                if not self._download_insider_quarter(y, q, p):
                    missing.touch()
            y, q = (y + 1, 1) if q == 4 else (y, q + 1)
        parts = [pd.read_parquet(f) for f in sorted(folder.glob("*.parquet"))]
        self._insider = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return self._insider

    def _download_insider_quarter(self, y: int, q: int, out: Path) -> bool:
        import zipfile

        url = f"https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip"
        try:
            raw = self._sec(url)
        except RuntimeError as exc:
            log.warning("insider data %sq%s not published yet: %s", y, q, str(exc)[:80])
            return False
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            sub = pd.read_csv(z.open("SUBMISSION.tsv"), sep="\t", dtype=str,
                              usecols=["ACCESSION_NUMBER", "FILING_DATE", "ISSUERTRADINGSYMBOL"])
            tr = pd.read_csv(z.open("NONDERIV_TRANS.tsv"), sep="\t", dtype=str,
                             usecols=["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"])
        tr = tr[tr.TRANS_CODE.isin(["P", "S"])].merge(sub, on="ACCESSION_NUMBER")
        df = pd.DataFrame({
            "symbol": tr.ISSUERTRADINGSYMBOL.str.upper().str.strip(),
            "filed": pd.to_datetime(tr.FILING_DATE, format="%d-%b-%Y", errors="coerce").dt.date,
            "code": tr.TRANS_CODE,
            "value": pd.to_numeric(tr.TRANS_SHARES, errors="coerce") * pd.to_numeric(tr.TRANS_PRICEPERSHARE, errors="coerce"),
        }).dropna()
        df.to_parquet(out)
        return True

    def limit_insider(self, symbols: list[str]) -> None:
        """Keep only these symbols in memory (research workers load the file once each)."""
        t = self.insider_trades()
        self._insider = t[t.symbol.isin(set(symbols))] if not t.empty else t

    def insider_context(self, symbol: str, day: date, days: int = 30) -> dict:
        """Insider open-market buying and selling filed in the `days` before `day`."""
        t = self.insider_trades()
        if t.empty:
            return {"insider_buys_30d": None, "insider_net_30d": None}
        cache = self.__dict__.setdefault("_insider_by_sym", {})
        if symbol not in cache:
            cache[symbol] = t[t.symbol == symbol]
        w = cache[symbol]
        w = w[(w.filed < day) & (w.filed >= day - timedelta(days=days))]
        buys = w[w.code == "P"]
        return {"insider_buys_30d": int(len(buys)),
                "insider_net_30d": float(buys.value.sum() - w[w.code == "S"].value.sum())}

    # -- Alpha Vantage: earnings surprise (free key, 25 requests/day) -------------------------
    def earnings_history(self, symbol: str, fetch: bool = True) -> pd.DataFrame | None:
        """Every quarter: reported date, before/after market, actual vs expected EPS. None if not fetched yet."""
        p = self.dir / "alphavantage_earnings" / f"{symbol}.json"
        if not self._fresh(p, 24 * 30) and fetch and self.earnings_calls_left != 0:
            key = _env("ALPHAVANTAGE_API_KEY")
            if key:
                if self.earnings_calls_left is not None:
                    self.earnings_calls_left -= 1
                import requests

                r = requests.get("https://www.alphavantage.co/query",
                                 params={"function": "EARNINGS", "symbol": symbol, "apikey": key}, timeout=30)
                j = r.json()
                if "quarterlyEarnings" in j and j["quarterlyEarnings"]:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(j["quarterlyEarnings"]))
                elif not p.exists():
                    log.warning("Alpha Vantage gave no earnings for %s: %s", symbol, str(j)[:120])
        if not p.exists():
            return None
        df = pd.DataFrame(json.loads(p.read_text()))
        df["reportedDate"] = pd.to_datetime(df["reportedDate"], errors="coerce").dt.date
        for c in ("reportedEPS", "estimatedEPS", "surprisePercentage"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def earnings_surprise(self, symbol: str, day: date, fetch: bool = False) -> float | None:
        """EPS surprise % if the company reported between the prior session and this open."""
        df = self.earnings_history(symbol, fetch=fetch)
        if df is None:
            return None
        prev = day - timedelta(days=3 if day.weekday() == 0 else 1)
        hit = df[((df.reportedDate == day) & (df.reportTime != "post-market"))
                 | ((df.reportedDate >= prev) & (df.reportedDate < day) & (df.reportTime == "post-market"))]
        return float(hit.surprisePercentage.iloc[0]) if len(hit) and pd.notna(hit.surprisePercentage.iloc[0]) else None

    # -- FRED: Treasury yields and credit spreads (free key) ----------------------------------
    FRED_SERIES = {"hy_spread": "BAMLH0A0HYM2", "t10y": "DGS10", "t2y": "DGS2"}

    def fred(self, series: str) -> pd.Series | None:
        p = self.dir / "fred" / f"{series}.csv"
        if not self._fresh(p, 12):
            key = _env("FRED_API_KEY")
            if key:
                import requests

                r = requests.get("https://api.stlouisfed.org/fred/series/observations", timeout=30, params={
                    "series_id": series, "api_key": key, "file_type": "json", "observation_start": "2014-01-01"})
                r.raise_for_status()
                obs = pd.DataFrame(r.json()["observations"])[["date", "value"]]
                p.parent.mkdir(parents=True, exist_ok=True)
                obs.to_csv(p, index=False)
        if not p.exists():
            return None
        df = pd.read_csv(p)
        s = pd.to_numeric(df["value"], errors="coerce")
        s.index = pd.to_datetime(df["date"]).dt.date
        return s.dropna()

    def rates_context(self, day: date) -> dict:
        """Prior-day credit spread and yields, with their 20-day changes (in percentage points)."""
        out = {}
        for name, sid in self.FRED_SERIES.items():
            s = self.fred(sid)
            s = s[s.index < day] if s is not None else None
            ok = s is not None and len(s) > 21
            out[name] = float(s.iloc[-1]) if ok else None
            out[f"{name}_20d"] = float(s.iloc[-1] - s.iloc[-21]) if ok else None
        return out

    # -- trading halts (live only: Nasdaq publishes today's halts) ---------------------------
    def halts_today(self) -> dict[str, str]:
        """Symbols halted today across US markets -> reason code (e.g. T1 news pending, LUDP volatility)."""
        xml = _get("https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts").decode("utf8", "ignore")
        out = {}
        for item in re.findall(r"<item>(.*?)</item>", xml, re.S):
            sym = re.search(r"<ndaq:IssueSymbol>([^<]+)", item)
            why = re.search(r"<ndaq:ReasonCode>([^<]*)", item)
            if sym:
                out[sym.group(1).strip()] = why.group(1).strip() if why else ""
        return out


def catalyst_window(day: date, lookback_hours: float) -> tuple[datetime, datetime]:
    """Same window the news check uses: the hours before `day`'s 09:30 open."""
    open_dt = datetime.combine(day, time(9, 30), ET)
    return open_dt - timedelta(hours=lookback_hours), open_dt


def expiration_events(day: date) -> list[str]:
    """Options expiration (3rd Friday), quad witching / S&P rebalance (March, June, Sept, Dec),
    and the Russell reconstitution (last Friday of June). Holiday shifts are ignored."""
    if day.weekday() != 4:
        return []
    out = []
    if 15 <= day.day <= 21:
        out.append("quad witching" if day.month in (3, 6, 9, 12) else "opex")
    if day.month == 6 and day.day + 7 > 30:
        out.append("Russell recon")
    return out


if __name__ == "__main__":  # python -m patterntrader.external --check
    x = External()
    day = date.today()
    checks = {
        "SEC filings": lambda: f"{len(x.filings('AAPL'))} AAPL filings",
        "SEC shares outstanding": lambda: f"AAPL {x.shares_outstanding('AAPL', day):,.0f} shares",
        "FINRA short interest": lambda: f"AAPL {x.short_context('AAPL', day)}",
        "VIX": lambda: f"{x.vix_context(day)}",
        "Macro calendar": lambda: f"{len(x.events())} event days",
        "Alpha Vantage earnings": lambda: ("no key" if not _env("ALPHAVANTAGE_API_KEY")
                                           else f"{len(x.earnings_history('IBM'))} IBM quarters"),
        "FRED rates": lambda: "no key" if not _env("FRED_API_KEY") else f"{x.rates_context(day)}",
    }
    for name, fn in checks.items():
        try:
            print(f"OK    {name}: {fn()}")
        except Exception as exc:
            print(f"FAIL  {name}: {str(exc)[:150]}")
