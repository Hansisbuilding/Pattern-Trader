"""Strategy base class: regime gating, direction rules and signal validation."""

from __future__ import annotations

import pandas as pd

from patterntrader.config import parse_time
from patterntrader.models import SetupContext, Signal

TICK = 0.01


def mirror(bars: pd.DataFrame) -> pd.DataFrame:
    """Flip prices so a short setup can be detected with the long-side logic."""
    m = bars.copy()
    m["open"], m["close"] = -bars["open"], -bars["close"]
    m["high"], m["low"] = -bars["low"], -bars["high"]
    if "vwap" in bars:
        m["vwap"] = -bars["vwap"]
    return m


class Strategy:
    name = "base"
    kind = "momentum"  # or "reversion"

    def __init__(self, params, cfg):
        self.p = params
        self.cfg = cfg
        self.priority = params.get("priority", 9)
        w0, w1 = params.get("window", ["09:35", "15:00"])
        self.window = (parse_time(w0), parse_time(w1))
        # Momentum uses the global 2R default; reversion plays can opt for a lower R:R because
        # they win more often. The edge gate decides whether that trade-off actually pays.
        self.min_rr = max(params.get("min_rr", cfg.risk.min_reward_risk), 1.0)

    # -- situation gating ------------------------------------------------------
    def check(self, ctx: SetupContext) -> tuple[bool, str]:
        """Is this strategy allowed for this stock today? Returns (ok, note-or-reason)."""
        p, notes = self.p, []
        if ctx.market_regime not in p.get("market_regimes", [ctx.market_regime]):
            return False, f"market is {ctx.market_regime}"
        if ctx.stock_regime not in p.get("stock_regimes", [ctx.stock_regime]):
            return False, f"stock is {ctx.stock_regime}"
        if ctx.rvol is None:
            if "min_rvol" in p:
                notes.append(f"RVOL confirms after open (needs >= {p['min_rvol']})")
        else:
            if ctx.rvol < p.get("min_rvol", 0):
                return False, f"RVOL {ctx.rvol:.1f} < {p['min_rvol']}"
            if ctx.rvol > p.get("max_rvol", float("inf")):
                return False, f"RVOL {ctx.rvol:.1f} > {p['max_rvol']} (too much momentum to fade)"
        if p.get("in_play"):
            ok, why = self.in_play(ctx)
            if not ok:
                return False, why
        if not any(self.side_allowed(ctx, side) for side in self.sides(ctx)):
            return False, "no allowed direction (no longs in a down market; shorts off or not shortable)"
        ok, why = self.extra_check(ctx)
        if not ok:
            return False, why
        if why:
            notes.append(why)
        return True, "; ".join(notes)

    def in_play(self, ctx: SetupContext) -> tuple[bool, str]:
        """Momentum only pays on stocks with a reason to move: a gap with news behind it.

        2016-21 research: the same breakout patterns lost money on quiet stocks and made money
        on gappers with a catalyst, in every market regime. Longs need a gap up; shorts (when
        enabled) the mirror image, a gap down with news.
        """
        rule = self.cfg.in_play
        if not (self._gap_fits(ctx, "long") or self._gap_fits(ctx, "short")):
            return False, f"not in play: gap {ctx.gap_pct:+.1f}% (needs {rule.min_gap_pct}% with news)"
        if rule.require_catalyst:
            if ctx.has_catalyst is None:
                return False, "not in play: news not checked"
            if not ctx.has_catalyst:
                return False, "not in play: no news behind the gap"
        return True, ""

    def extra_check(self, ctx: SetupContext) -> tuple[bool, str]:
        return True, ""

    def sides(self, ctx: SetupContext) -> list[str]:
        return ["long", "short"]

    def _gap_fits(self, ctx: SetupContext, side: str) -> bool:
        gap = self.cfg.in_play.min_gap_pct
        if side == "long":
            return ctx.gap_pct >= gap
        return self.shorts_on(ctx) and ctx.gap_pct <= -gap

    def shorts_on(self, ctx: SetupContext) -> bool:
        r = self.cfg.risk
        return bool(r.allow_short and ctx.shortable
                    and ctx.market_regime in r.get("short_market_regimes", ["trend_down", "range", "mixed"]))

    def side_allowed(self, ctx: SetupContext, side: str) -> bool:
        if side not in self.sides(ctx):
            return False
        in_play = self.p.get("in_play")
        if side == "long":
            if in_play and (not self._gap_fits(ctx, "long") or ctx.stock_regime == "trend_down"):
                return False  # only buy gap-ups, never a stock in a daily downtrend
            # don't buy against a falling market (research can switch this off to measure it)
            return ctx.market_regime != "trend_down" or self.cfg.risk.get("long_in_down_market", False)
        if in_play and (not self._gap_fits(ctx, "short") or ctx.stock_regime == "trend_up"):
            return False  # only short gap-downs, never a stock in a daily uptrend
        return self.shorts_on(ctx) and ctx.market_regime != "trend_up"

    def in_window(self, ts: pd.Timestamp) -> bool:
        return self.window[0] <= ts.time() <= self.window[1]

    # -- signals ---------------------------------------------------------------
    def signal(self, ctx: SetupContext, bars: pd.DataFrame) -> Signal | None:
        """bars: today's regular-session 1-minute bars up to now, with a 'vwap' column."""
        raise NotImplementedError

    def emit(self, ctx, side, entry, stop, target, ts, reason, mirrored=False) -> Signal | None:
        """Validate and build a signal. Prices may be in mirrored space for shorts."""
        if mirrored:
            entry, stop, target = -entry, -stop, -target
        d = 1 if side == "long" else -1
        entry, stop, target = round(entry, 2), round(stop, 2), round(target, 2)
        if (entry - stop) * d <= 0 or (target - entry) * d <= 0:
            return None
        if abs(entry - stop) > self.p.get("max_stop_atr", 0.5) * ctx.daily_atr:
            return None  # stop too wide for the stock's normal range: bad trade location
        sig = Signal(self.name, ctx.symbol, side, entry, stop, target, ts.to_pydatetime(), reason)
        if sig.reward_risk < self.min_rr - 0.02:  # small slack for penny rounding
            return None
        return sig

    def target_for(self, entry: float, stop: float) -> float:
        """target_r sets the target distance in R; a large value lets the winner run to the close."""
        return entry + self.p.get("target_r", self.min_rr) * (entry - stop)
