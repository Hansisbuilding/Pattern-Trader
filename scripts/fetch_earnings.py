"""Fill the Alpha Vantage earnings cache a little each day (free tier: 25 calls/day).

Usage: .venv/bin/python scripts/fetch_earnings.py [max_calls]
Run it once a day until it reports nothing left; histories then refresh monthly.
"""
import sys
import time

from patterntrader.config import ROOT, load_config
from patterntrader.external import External, _env


def main():
    if not _env("ALPHAVANTAGE_API_KEY"):
        raise SystemExit("ALPHAVANTAGE_API_KEY is not set: run scripts/set-data-keys.sh")
    cfg = load_config()
    symbols = [l.strip().upper() for l in open(ROOT / cfg.backtest.universe_file) if l.strip() and not l.startswith("#")]
    x = External(cfg.data.cache_dir)
    todo = [s for s in symbols if not (x.dir / "alphavantage_earnings" / f"{s}.json").exists()]
    research = ROOT / "results" / "research" / "signals_insample_2016_2021.parquet"
    if research.exists():  # most gap-up setups first: that's where an earnings surprise matters
        import pandas as pd

        sig = pd.read_parquet(research)
        counts = sig[sig.gap_pct >= 1.5].groupby("symbol").size()
        todo.sort(key=lambda s: -counts.get(s, 0))
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    x.earnings_calls_left = budget
    got = 0
    for s in todo[:budget]:
        if x.earnings_history(s) is not None:
            got += 1
        time.sleep(13)  # stay under the per-minute limit as well
    print(f"fetched {got} of {min(budget, len(todo))} tried; {len(todo) - got} symbols still missing")


if __name__ == "__main__":
    main()
