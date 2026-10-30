"""Shared data types: the setup context for a symbol-day and trade signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime


@dataclass
class SetupContext:
    """Everything known about a symbol before/at the open, used to judge the situation."""

    symbol: str
    day: date
    price: float
    prev_close: float
    prev_high: float
    prev_low: float
    daily_atr: float
    atr_pct: float
    avg_volume: float
    avg_dollar_volume: float
    gap_pct: float
    adx: float
    stock_regime: str
    market_regime: str
    rvol: float | None = None          # None until the opening minutes have traded
    has_catalyst: bool | None = None   # None when news wasn't checked
    headlines: list[str] = field(default_factory=list)
    premarket_high: float | None = None
    premarket_low: float | None = None
    shortable: bool = False
    eligible: list[str] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)
    score: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["day"] = self.day.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SetupContext":
        d = dict(d)
        d["day"] = date.fromisoformat(d["day"])
        return cls(**d)


@dataclass
class Signal:
    strategy: str
    symbol: str
    side: str          # "long" | "short"
    entry: float       # reference price (close of the signal bar)
    stop: float
    target: float
    time: datetime
    reason: str

    @property
    def direction(self) -> int:
        return 1 if self.side == "long" else -1

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def reward_risk(self) -> float:
        risk = self.risk_per_share
        return abs(self.target - self.entry) / risk if risk > 0 else 0.0
