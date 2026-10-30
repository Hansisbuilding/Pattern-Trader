"""Risk manager: position sizing and the daily discipline rules. Shared by backtest and live."""

from __future__ import annotations

import math
from datetime import datetime

from patterntrader.config import parse_time
from patterntrader.models import Signal


class RiskManager:
    def __init__(self, cfg, day_start_equity: float, size_factor: float = 1.0):
        self.r = cfg.risk
        self.size_factor = size_factor  # market-mood adjustment for today (1 = normal size)
        self.no_before = parse_time(cfg.session.no_entry_before)
        self.no_after = parse_time(cfg.session.no_entry_after)
        self.start_equity = day_start_equity
        self.trades_today = 0
        self.consecutive_losses = 0
        self.realized = 0.0
        self.open_symbols: set[str] = set()
        self.losers: set[str] = set()
        self.halted: str | None = None
        self.flatten_now = False  # set when the daily loss limit is hit

    def update_equity(self, equity: float) -> None:
        change_pct = (equity - self.start_equity) / self.start_equity * 100
        if change_pct <= -self.r.max_daily_loss_pct and not self.flatten_now:
            self.halted = f"daily loss limit hit ({change_pct:.2f}%)"
            self.flatten_now = True

    def size(
        self,
        sig: Signal,
        equity: float,
        buying_power: float | None = None,
        open_notional: float = 0.0,
        avg_dollar_volume: float | None = None,
        setup_factor: float = 1.0,
    ) -> int:
        """Shares such that hitting the stop loses risk_per_trade_pct of equity.

        The risk rule sets the size. The caps only stop it in edge cases: one position
        bigger than max_position_pct of equity, total exposure above max_gross_exposure_pct,
        more than the broker's buying power, or more than a sliver of the stock's daily volume.
        """
        rps = sig.risk_per_share
        if rps <= 0:
            return 0
        qty = math.floor(equity * self.r.risk_per_trade_pct * self.size_factor * setup_factor / 100 / rps)
        caps = [
            equity * self.r.max_position_pct / 100,
            equity * self.r.get("max_gross_exposure_pct", self.r.max_position_pct) / 100 - open_notional,
        ]
        if buying_power is not None:
            caps.append(buying_power * 0.95)
        if avg_dollar_volume:
            caps.append(avg_dollar_volume * self.r.get("max_adv_pct", 1.0) / 100)
        qty = min(qty, math.floor(max(min(caps), 0) / sig.entry))
        return max(qty, 0)

    def approve(self, sig: Signal, now: datetime) -> tuple[bool, str]:
        if self.size_factor <= 0:
            return False, "sitting out today (market sentiment)"
        if self.halted:
            return False, f"halted: {self.halted}"
        if not (self.no_before <= now.time() <= self.no_after):
            return False, "outside entry hours"
        if self.trades_today >= self.r.max_trades_per_day:
            return False, f"max {self.r.max_trades_per_day} trades/day reached"
        if len(self.open_symbols) >= self.r.max_open_positions:
            return False, f"max {self.r.max_open_positions} open positions"
        if sig.symbol in self.open_symbols:
            return False, "already in this symbol"
        if self.r.no_reentry_after_loss and sig.symbol in self.losers:
            return False, "no re-entry after a loss in this symbol"
        min_stop = max(0.02, sig.entry * self.r.min_stop_pct / 100)
        if sig.risk_per_share < min_stop:
            return False, f"stop {sig.risk_per_share:.2f} tighter than noise ({min_stop:.2f})"
        return True, ""

    def on_entry(self, sig: Signal) -> None:
        self.trades_today += 1
        self.open_symbols.add(sig.symbol)

    def on_cancel(self, symbol: str) -> None:
        """Entry never filled: free the slot but keep the trade count (no chasing)."""
        self.open_symbols.discard(symbol)

    def on_exit(self, symbol: str, pnl: float) -> None:
        self.realized += pnl
        self.open_symbols.discard(symbol)
        if pnl < 0:
            self.consecutive_losses += 1
            self.losers.add(symbol)
            if self.consecutive_losses >= self.r.max_consecutive_losses:
                self.halted = f"{self.consecutive_losses} losses in a row"
        else:
            self.consecutive_losses = 0
