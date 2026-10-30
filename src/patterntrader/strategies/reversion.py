"""Reversion playbooks: Bollinger/RSI mean reversion and no-news gap fades."""

from __future__ import annotations

import pandas as pd

from patterntrader.indicators import bollinger, resample_complete, rsi
from patterntrader.strategies.base import TICK, Strategy, mirror


class MeanReversion(Strategy):
    """In a quiet, range-bound stock, fade a stretch beyond the 5-minute Bollinger Band once it turns."""

    name = "mean_reversion"
    kind = "reversion"

    def extra_check(self, ctx):
        if abs(ctx.gap_pct) > self.p.max_abs_gap_pct:
            return False, f"gap {ctx.gap_pct:+.1f}% too big for a range day"
        if self.p.require_no_catalyst and ctx.has_catalyst:
            return False, "has news: moves with news don't revert reliably"
        if self.p.require_no_catalyst and ctx.has_catalyst is None:
            return True, "news not checked"
        return True, ""

    def signal(self, ctx, bars):
        ts = bars.index[-1]
        if not self.in_window(ts):
            return None
        session_open = ts.normalize() + pd.Timedelta(hours=9, minutes=30)
        if int((ts - session_open).total_seconds() // 60 + 1) % 5:
            return None  # only act when a 5-minute bar completes
        b5 = resample_complete(bars, 5, session_open)
        n = self.p.bb_period
        if len(b5) < n + 2:
            return None
        for side, work in (("long", b5), ("short", mirror(b5))):
            if not self.side_allowed(ctx, side):
                continue
            mid, _, lower = bollinger(work["close"], n, self.p.bb_std)
            r = rsi(work["close"], self.p.rsi_period)
            # In mirrored space RSI flips: RSI(-x) = 100 - RSI(x), so the same threshold applies.
            rsi_limit = self.p.rsi_low if side == "long" else 100 - self.p.rsi_high
            c = work["close"]
            stretched = c.iloc[-2] < lower.iloc[-2] and r.iloc[-2] <= rsi_limit
            turned = c.iloc[-1] > lower.iloc[-1] and c.iloc[-1] > c.iloc[-2]
            if stretched and turned:
                stop = work["low"].iloc[-3:].min() - TICK
                return self.emit(
                    ctx, side, c.iloc[-1], stop, mid.iloc[-1], ts,
                    f"stretched past {self.p.bb_std}σ band (RSI {r.iloc[-2]:.0f}) and turned back",
                    mirrored=side == "short",
                )
        return None


class GapFade(Strategy):
    """A gap with no news that fails: trade back toward yesterday's close."""

    name = "gap_fade"
    kind = "reversion"

    def sides(self, ctx):
        return ["long"] if ctx.gap_pct < 0 else ["short"]

    def extra_check(self, ctx):
        if abs(ctx.gap_pct) < self.p.min_abs_gap_pct:
            return False, f"gap {ctx.gap_pct:+.1f}% < {self.p.min_abs_gap_pct}%"
        if self.p.require_no_catalyst:
            if ctx.has_catalyst is None:
                return False, "news not checked: won't fade an unexplained gap blind"
            if ctx.has_catalyst:
                return False, "gap has news: don't fade it"
        return True, ""

    def signal(self, ctx, bars):
        n = self.p.range_minutes
        if len(bars) <= n + 1 or not self.in_window(bars.index[-1]):
            return None
        side = self.sides(ctx)[0]
        if not self.side_allowed(ctx, side):
            return None
        mirrored = side == "short"
        work = mirror(bars) if mirrored else bars
        prev_close = -ctx.prev_close if mirrored else ctx.prev_close
        last, prev = work.iloc[-1], work.iloc[-2]
        rng_hi = work["high"].iloc[:n].max()
        if last.close > rng_hi >= prev.close and last.close > last.vwap:
            day_open = work["open"].iloc[0]
            target = day_open + self.p.fill_fraction * (prev_close - day_open)
            stop = work["low"].min() - TICK
            return self.emit(
                ctx, side, last.close, stop, target, bars.index[-1],
                f"no-news gap {ctx.gap_pct:+.1f}% failing: reclaimed opening range and VWAP",
                mirrored=mirrored,
            )
        return None
