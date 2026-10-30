"""Momentum playbooks: opening range breakout, gap and go, VWAP pullback, bull/bear flag."""

from __future__ import annotations

import pandas as pd

from patterntrader.models import SetupContext, Signal
from patterntrader.strategies.base import TICK, Strategy, mirror


class OpeningRangeBreakout(Strategy):
    """Break of the first N-minute range, on volume, on the right side of VWAP."""

    name = "orb"

    def extra_check(self, ctx):
        if abs(ctx.gap_pct) < self.p.min_abs_gap_pct and ctx.atr_pct < 2:
            return False, f"not in play (gap {ctx.gap_pct:+.1f}%, ATR {ctx.atr_pct:.1f}%)"
        return True, ""

    def signal(self, ctx: SetupContext, bars: pd.DataFrame) -> Signal | None:
        n = self.p.range_minutes
        if len(bars) <= n + 1 or not self.in_window(bars.index[-1]):
            return None
        last, prev = bars.iloc[-1], bars.iloc[-2]
        avg_vol = bars["volume"].iloc[-21:-1].mean()
        if last.volume < self.p.breakout_volume_mult * avg_vol:
            return None
        hi, lo = bars["high"].iloc[:n].max(), bars["low"].iloc[:n].min()
        for side, work in (("long", bars), ("short", mirror(bars))):
            if not self.side_allowed(ctx, side):
                continue
            w_last, w_prev = work.iloc[-1], work.iloc[-2]
            w_hi, w_lo = (hi, lo) if side == "long" else (-lo, -hi)
            if w_last.close > w_hi >= w_prev.close and w_last.close > w_last.vwap:
                stop = w_lo - TICK
                if w_last.close - stop > self.p.max_stop_atr * ctx.daily_atr:
                    stop = (w_hi + w_lo) / 2 - TICK  # wide range: use the midpoint
                return self.emit(
                    ctx, side, w_last.close, stop, self.target_for(w_last.close, stop),
                    bars.index[-1], f"{n}m opening range {'high' if side == 'long' else 'low'} break on volume",
                    mirrored=side == "short",
                )
        return None


class GapAndGo(Strategy):
    """A gap with a real catalyst, breaking to a new high of day above the premarket high."""

    name = "gap_and_go"

    def sides(self, ctx):
        return ["long"] if ctx.gap_pct > 0 else ["short"]

    def extra_check(self, ctx):
        if abs(ctx.gap_pct) < self.p.min_abs_gap_pct:
            return False, f"gap {ctx.gap_pct:+.1f}% < {self.p.min_abs_gap_pct}%"
        if self.p.require_catalyst:
            if ctx.has_catalyst is None:
                return False, "catalyst required but news was not checked"
            if not ctx.has_catalyst:
                return False, "no news catalyst behind the gap"
        return True, ""

    def signal(self, ctx, bars):
        if len(bars) < 6 or not self.in_window(bars.index[-1]):
            return None
        side = self.sides(ctx)[0]
        if not self.side_allowed(ctx, side):
            return None
        work = bars if side == "long" else mirror(bars)
        last = work.iloc[-1]
        level = work["high"].iloc[:-1].max()
        pm = ctx.premarket_high if side == "long" else (-ctx.premarket_low if ctx.premarket_low else None)
        if pm:
            level = max(level, pm)
        avg_vol = work["volume"].iloc[-11:-1].mean()
        if last.close > level and last.close > last.vwap and last.volume >= self.p.breakout_volume_mult * avg_vol:
            stop = work["low"].iloc[-self.p.stop_lookback :].min() - TICK
            return self.emit(
                ctx, side, last.close, stop, self.target_for(last.close, stop), bars.index[-1],
                f"gap {ctx.gap_pct:+.1f}% with news, new {'high' if side == 'long' else 'low'} of day",
                mirrored=side == "short",
            )
        return None


class VWAPPullback(Strategy):
    """In a trending stock, buy the first clean bounce off VWAP (sell the rejection in a downtrend)."""

    name = "vwap_pullback"

    def sides(self, ctx):
        return ["long"] if ctx.stock_regime == "trend_up" else ["short"]

    def signal(self, ctx, bars):
        if len(bars) < 22 or not self.in_window(bars.index[-1]):
            return None
        side = self.sides(ctx)[0]
        if not self.side_allowed(ctx, side):
            return None
        work = bars if side == "long" else mirror(bars)
        recent = work.iloc[-21:-1]
        if (recent["close"] > recent["vwap"]).mean() < self.p.above_vwap_frac:
            return None
        prev, last = work.iloc[-2], work.iloc[-1]
        tol = abs(prev.vwap) * self.p.touch_tolerance_pct / 100
        pierce = abs(prev.vwap) * self.p.max_pierce_pct / 100
        touched = prev.vwap - pierce <= prev.low <= prev.vwap + tol
        if touched and last.close > last.vwap and last.close > prev.high:
            stop = work["low"].iloc[-4:].min() - TICK
            return self.emit(
                ctx, side, last.close, stop, self.target_for(last.close, stop), bars.index[-1],
                "trend day: VWAP held and bounced", mirrored=side == "short",
            )
        return None


class BullFlag(Strategy):
    """Sharp pole, shallow low-volume flag, breakout of the flag on volume (bear flag for shorts)."""

    name = "bull_flag"

    def find_flag(self, work: pd.DataFrame, atr: float):
        p, last = self.p, work.iloc[-1]
        for f in range(p.min_flag_bars, p.max_flag_bars + 1):
            if len(work) < f + p.max_pole_bars + 1:
                break
            flag = work.iloc[-1 - f : -1]
            pole = work.iloc[-1 - f - p.max_pole_bars : -1 - f]
            hi_pos = int(pole["high"].values.argmax())
            if hi_pos < len(pole) - 3:
                continue  # the pole must run right into the flag
            lo_pos = int(pole["low"].values[: hi_pos + 1].argmin())
            pole_hi, pole_lo = pole["high"].iloc[hi_pos], pole["low"].iloc[lo_pos]
            height = pole_hi - pole_lo
            if height < p.pole_min_atr * atr:
                continue
            if flag["high"].max() > pole_hi or flag["low"].min() < pole_hi - p.max_retrace * height:
                continue
            if flag["volume"].mean() >= pole["volume"].iloc[lo_pos : hi_pos + 1].mean():
                continue  # a real flag is a quiet rest
            if (
                last.close > flag["high"].max()
                and last.volume > flag["volume"].mean()
                and last.close > last.vwap
            ):
                return flag["low"].min() - TICK, f
        return None

    def signal(self, ctx, bars):
        if not self.in_window(bars.index[-1]):
            return None
        for side, work in (("long", bars), ("short", mirror(bars))):
            if not self.side_allowed(ctx, side):
                continue
            found = self.find_flag(work, ctx.daily_atr)
            if found:
                stop, f = found
                entry = work["close"].iloc[-1]
                return self.emit(
                    ctx, side, entry, stop, self.target_for(entry, stop), bars.index[-1],
                    f"{'bull' if side == 'long' else 'bear'} flag breakout ({f}-bar flag)",
                    mirrored=side == "short",
                )
        return None
