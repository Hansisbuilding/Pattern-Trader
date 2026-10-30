"""Collect research signals: in-sample (development) and out-of-sample (sealed) files.

Usage: .venv/bin/python scripts/collect_signals.py
"""
import logging
import sys
from datetime import date

from patterntrader.config import ROOT, load_config
from patterntrader.research import collect, research_config, save


def main():
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    cfg = research_config(load_config())
    symbols = [l.strip().upper() for l in open(ROOT / cfg.backtest.universe_file) if l.strip() and not l.startswith("#")]
    periods = {
        "signals_insample_2016_2021": (date(2016, 10, 28), date(2021, 10, 27)),
        "signals_oos_2021_2026": (date(2021, 10, 28), date(2026, 10, 27)),
    }
    for name in sys.argv[1:] or periods:
        df = collect(cfg, symbols, *periods[name])
        print(name, len(df), save(df, name), flush=True)


if __name__ == "__main__":  # worker processes re-import this file
    main()
