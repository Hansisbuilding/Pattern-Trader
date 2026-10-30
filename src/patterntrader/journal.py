"""Trade journal: every signal, rejection, entry and exit, appended to a CSV."""

from __future__ import annotations

import csv
import logging
from datetime import date, datetime

from patterntrader.config import ET, ROOT

log = logging.getLogger("journal")
FIELDS = ["time", "event", "symbol", "strategy", "side", "qty", "price", "stop", "target", "pnl", "note"]


class Journal:
    def __init__(self, day: date):
        self.dir = ROOT / "journal"
        self.dir.mkdir(exist_ok=True)
        self.path = self.dir / "trades.csv"
        self.day = day
        if not self.path.exists():
            with open(self.path, "w", newline="") as fh:
                csv.writer(fh).writerow(FIELDS)

    def record(self, event: str, symbol: str = "", sig=None, qty=None, price=None, pnl=None, note: str = ""):
        row = {
            "time": datetime.now(ET).isoformat(timespec="seconds"),
            "event": event,
            "symbol": symbol or (sig.symbol if sig else ""),
            "strategy": sig.strategy if sig else "",
            "side": sig.side if sig else "",
            "qty": qty if qty is not None else "",
            "price": price if price is not None else (sig.entry if sig else ""),
            "stop": sig.stop if sig else "",
            "target": sig.target if sig else "",
            "pnl": round(pnl, 2) if pnl is not None else "",
            "note": note or (sig.reason if sig else ""),
        }
        with open(self.path, "a", newline="") as fh:
            csv.DictWriter(fh, FIELDS).writerow(row)
        log.info("%-8s %-6s %s %s", event, row["symbol"], row["strategy"], row["note"])
