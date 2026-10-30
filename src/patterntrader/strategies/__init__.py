from patterntrader.strategies.base import Strategy
from patterntrader.strategies.momentum import BullFlag, GapAndGo, OpeningRangeBreakout, VWAPPullback
from patterntrader.strategies.reversion import GapFade, MeanReversion

REGISTRY = {
    cls.name: cls
    for cls in (OpeningRangeBreakout, GapAndGo, VWAPPullback, BullFlag, MeanReversion, GapFade)
}


def build_strategies(cfg) -> list[Strategy]:
    """Enabled strategies, highest priority (lowest number) first."""
    out = [REGISTRY[name](params, cfg) for name, params in cfg.strategies.items() if params.get("enabled")]
    return sorted(out, key=lambda s: s.priority)
