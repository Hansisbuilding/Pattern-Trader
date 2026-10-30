"""Build the edge-gate stats file from two backtest runs of the same config.

Each run counts as one "half" for the both-halves check, so an edge must hold in both periods
(e.g. 2016-21 and 2021-26), not just in one recent 5-year window.

    python scripts/edge_stats_from_runs.py results/backtest_A results/backtest_B
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from patterntrader.backtest import _stats
from patterntrader.config import ET, ROOT


def main(first: str, second: str) -> None:
    runs = [Path(first), Path(second)]
    summaries = [json.loads((r / "summary.json").read_text()) for r in runs]
    trades = [pd.read_csv(r / "trades.csv") for r in runs]
    equity = float(summaries[0]["starting_equity"])
    both = pd.concat(trades)

    strategies = {}
    for name, g in both.groupby("strategy"):
        st = _stats(g, equity)
        st["expectancy_r_first_half"] = _stats(trades[0][trades[0].strategy == name], equity)["expectancy_r"]
        st["expectancy_r_second_half"] = _stats(trades[1][trades[1].strategy == name], equity)["expectancy_r"]
        strategies[name] = st

    out = ROOT / "results" / "strategy_stats.json"
    if out.exists():
        shutil.copy(out, out.with_name(f"strategy_stats_before_{datetime.now(ET):%Y%m%d_%H%M}.json"))
    out.write_text(json.dumps({
        "period": [summaries[0]["period"][0], summaries[1]["period"][1]],
        "halves": [s["period"] for s in summaries],
        "source_runs": [str(r) for r in runs],
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "starting_equity": equity,
        "strategies": strategies,
    }, indent=2))
    for name, st in strategies.items():
        print(f"{name:14} {st['trades']:4} trades  {st['expectancy_r']:+.2f}R  PF {st['profit_factor']:.2f}  "
              f"halves {st['expectancy_r_first_half']:+.2f} / {st['expectancy_r_second_half']:+.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
