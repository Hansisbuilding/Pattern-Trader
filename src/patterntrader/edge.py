"""Edge gate: a strategy only trades once a backtest shows it has a positive edge."""

from __future__ import annotations

import json

from patterntrader.config import ROOT


class EdgeGate:
    def __init__(self, cfg):
        self.cfg = cfg.edge_gate
        self.stats: dict = {}
        path = ROOT / self.cfg.stats_file
        if path.exists():
            self.stats = json.loads(path.read_text()).get("strategies", {})

    def allows(self, name: str) -> tuple[bool, str]:
        if not self.cfg.enabled:
            return True, ""
        st = self.stats.get(name)
        if not st:
            return False, "no backtest results yet (edge gate)"
        if st["trades"] < self.cfg.min_trades:
            return False, f"only {st['trades']} backtest trades (< {self.cfg.min_trades})"
        if st["expectancy_r"] < self.cfg.min_expectancy_r:
            return False, f"backtest expectancy {st['expectancy_r']:+.2f}R (< {self.cfg.min_expectancy_r})"
        if st["profit_factor"] < self.cfg.min_profit_factor:
            return False, f"backtest profit factor {st['profit_factor']:.2f} (< {self.cfg.min_profit_factor})"
        if self.cfg.get("require_both_halves") and min(
            st.get("expectancy_r_first_half", 0), st.get("expectancy_r_second_half", 0)
        ) <= 0:
            return False, "edge did not hold in both halves of the backtest"
        return True, ""
