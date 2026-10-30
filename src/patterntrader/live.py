"""Live paper trader: trades the approved watchlist with bracket orders, then flattens."""

from __future__ import annotations

import logging
import time as systime
from dataclasses import replace
from datetime import datetime, time, timedelta

from patterntrader import watchlist
from patterntrader.config import ET, paper_credentials, parse_time
from patterntrader.data import MarketData, regular_session
from patterntrader.edge import EdgeGate
from patterntrader.indicators import session_vwap
from patterntrader.journal import Journal
from patterntrader.models import Signal
from patterntrader.risk import RiskManager
from patterntrader.scanner import analyze
from patterntrader.strategies import build_strategies

log = logging.getLogger(__name__)


def now_et() -> datetime:
    return datetime.now(ET)


def sleep_until(target: datetime) -> None:
    while (delta := (target - now_et()).total_seconds()) > 0:
        systime.sleep(min(delta, 30))


class PaperBroker:
    """Alpaca trading client that can only ever point at the paper endpoint."""

    def __init__(self, key: str, secret: str):
        from alpaca.trading.client import TradingClient

        self.client = TradingClient(key, secret, paper=True)
        acct = self.client.get_account()
        if not str(acct.account_number).startswith("PA"):
            log.warning("account number %s doesn't look like a paper account", acct.account_number)

    def account(self):
        return self.client.get_account()

    def positions(self) -> dict:
        return {p.symbol: p for p in self.client.get_all_positions()}

    def submit_bracket(self, sig: Signal, qty: int, day) -> str:
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

        order = self.client.submit_order(
            MarketOrderRequest(
                symbol=sig.symbol,
                qty=qty,
                side=OrderSide.BUY if sig.side == "long" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(sig.target, 2)),
                stop_loss=StopLossRequest(stop_price=round(sig.stop, 2)),
                client_order_id=f"pt-{day:%Y%m%d}-{sig.symbol}-{sig.strategy}-{sig.time:%H%M}",
            )
        )
        return str(order.id)

    def order(self, order_id: str):
        from alpaca.trading.requests import GetOrderByIdRequest

        return self.client.get_order_by_id(order_id, GetOrderByIdRequest(nested=True))

    def last_exit_price(self, symbol: str, entry_side: str) -> float | None:
        """Price of the most recent filled order that closed a position in `symbol`."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        closing = "sell" if entry_side == "long" else "buy"
        orders = self.client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol], limit=20, nested=True)
        )
        candidates = []
        for o in orders:
            for leg in [o] + list(o.legs or []):
                if leg.filled_avg_price and getattr(leg.side, "value", leg.side) == closing:
                    candidates.append(leg)
        if not candidates:
            return None
        return float(max(candidates, key=lambda x: x.filled_at).filled_avg_price)

    def flatten(self) -> None:
        self.client.close_all_positions(cancel_orders=True)


class Trader:
    def __init__(self, cfg, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        key, secret = paper_credentials()
        self.broker = PaperBroker(key, secret)
        self.md = MarketData(key, secret, cfg.data.cache_dir)
        self.strategies = {s.name: s for s in build_strategies(cfg)}
        self.edge = EdgeGate(cfg)
        self.day = now_et().date()
        self.journal = Journal(self.day)
        self.tracked: dict[str, dict] = {}
        self.fired: set[tuple[str, str]] = set()

    # -- setup -------------------------------------------------------------------
    def load_targets(self):
        if self.cfg.watchlist.require_approval:
            picks = watchlist.approved(self.day)
        else:
            picks = [
                watchlist.SetupContext.from_dict({k: v for k, v in c.items() if k != "status"})
                for c in watchlist.load(self.day)["candidates"]
            ]
        return picks

    def confirm_at_open(self, picks):
        """Re-run the situation check with real opening data (RVOL, official open gap)."""
        rows = [
            {"symbol": c.symbol, "price": c.price, "prev_close": c.prev_close, "gap_pct": c.gap_pct,
             "avg_dollar_volume": c.avg_dollar_volume, "shortable": c.shortable}
            for c in picks
        ]
        regime, notes, contexts = analyze(
            self.cfg, self.md, rows, now_et(), list(self.strategies.values()), self.edge
        )
        self.journal.record("regime", note=f"{regime}: {'; '.join(notes)}")
        from patterntrader.sentiment import day_size_factor, log_decisions

        mood = watchlist.load(self.day).get("market_mood")
        m = mood or {}
        self.size_factor, why = day_size_factor(
            m.get("mood"), (m.get("vix") or {}).get("vix_term"), m.get("events") or [], self.cfg)
        log_decisions(self.day, "open", contexts, self.size_factor, why)
        if why:
            self.journal.record("sentiment", note="; ".join(why))
        active = []
        for c in contexts:
            if c.eligible:
                active.append(c)
                self.journal.record("confirm", c.symbol, note=f"rvol {c.rvol}, strategies {c.eligible}")
            else:
                self.journal.record("drop", c.symbol, note="; ".join(f"{k}: {v}" for k, v in c.rejected.items()))
        return regime, active

    # -- main loop -----------------------------------------------------------------
    def run(self) -> None:
        cfg = self.cfg
        clock = self.broker.client.get_clock()
        if not clock.is_open and clock.next_open.astimezone(ET).date() != self.day:
            log.info("market is closed today")
            return
        picks = self.load_targets()
        if not picks:
            log.info("no approved symbols for %s: no trades today (approve with `patterntrader approve`)", self.day)
            return
        log.info("approved: %s", ", ".join(c.symbol for c in picks))

        open_dt = datetime.combine(self.day, parse_time(cfg.session.open), ET)
        sleep_until(open_dt + timedelta(minutes=cfg.session.rvol_minutes, seconds=20))
        regime, active = self.confirm_at_open(picks)
        if (regime == "chaos" and not cfg.regime.get("trade_chaos", False)) or not active:
            log.info("nothing to trade after the open check (market %s)", regime)
            return

        start_equity = float(self.broker.account().last_equity)
        risk = RiskManager(cfg, start_equity, getattr(self, "size_factor", 1.0))
        if risk.size_factor <= 0:
            log.info("sentiment says sit out today")
            return
        flatten_at = datetime.combine(self.day, parse_time(cfg.session.flatten_at), ET)
        no_after = parse_time(cfg.session.no_entry_after)

        while now_et() < flatten_at:
            minute = now_et().replace(second=0, microsecond=0)
            sleep_until(minute + timedelta(minutes=1, seconds=3))
            now = now_et()
            try:
                acct = self.broker.account()
                risk.update_equity(float(acct.equity))
                if risk.flatten_now:
                    self.journal.record("halt", note=risk.halted)
                    break
                self.sync(risk)
                if risk.halted or now.time() > no_after:
                    continue
                self.scan_for_entries(active, risk, now, float(acct.equity), float(acct.buying_power))
            except Exception:  # keep the loop alive; brackets protect open positions
                log.exception("loop error")

        if not self.dry_run:
            self.broker.flatten()
            systime.sleep(5)
            self.sync(risk)
        self.journal.record("close", note=f"day done: realized {risk.realized:+.2f}, trades {risk.trades_today}")

    def scan_for_entries(self, active, risk, now, equity, buying_power):
        symbols = [c.symbol for c in active if c.symbol not in self.tracked]
        if not symbols:
            return
        open_dt = datetime.combine(self.day, time(9, 30), ET)
        current_minute = now.replace(second=0, microsecond=0)
        all_bars = self.md.fetch_bars(symbols, 1, open_dt, now, self.cfg.data.live_feed)
        for ctx in active:
            if ctx.symbol not in symbols or ctx.symbol not in all_bars:
                continue
            bars = regular_session(all_bars[ctx.symbol], self.day)
            bars = bars[bars.index < current_minute].copy()  # completed minutes only
            if len(bars) < 2:
                continue
            bars["vwap"] = session_vwap(bars)
            for name in ctx.eligible:
                if (ctx.symbol, name) in self.fired:
                    continue
                sig = self.strategies[name].signal(ctx, bars)
                if sig is None:
                    continue
                self.fired.add((ctx.symbol, name))
                if self.try_enter(sig, ctx, risk, now, equity, buying_power):
                    break

    def try_enter(self, sig: Signal, ctx, risk, now, equity, buying_power) -> bool:
        ok, why = risk.approve(sig, now)
        if not ok:
            self.journal.record("reject", sig=sig, note=why)
            return False
        quote = self.md.latest_quote(sig.symbol, self.cfg.data.live_feed)
        bid, ask = float(quote.bid_price or 0), float(quote.ask_price or 0)
        if bid <= 0 or ask <= 0:
            self.journal.record("reject", sig=sig, note="no valid quote")
            return False
        spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
        if spread_pct > self.cfg.risk.max_spread_pct:
            self.journal.record("reject", sig=sig, note=f"spread {spread_pct:.2f}% too wide")
            return False
        live = replace(sig, entry=ask if sig.side == "long" else bid)
        strat = self.strategies[sig.strategy]
        d = live.direction
        if (live.entry - live.stop) * d <= 0 or (live.target - live.entry) * d <= 0 or live.reward_risk < strat.min_rr * 0.9:
            self.journal.record("reject", sig=live, note=f"price ran away (R:R now {live.reward_risk:.2f})")
            return False
        open_notional = sum(abs(float(p.market_value)) for p in self.broker.positions().values())
        from patterntrader.sentiment import setup_size_factor

        qty = risk.size(live, equity, buying_power, open_notional, ctx.avg_dollar_volume,
                        setup_factor=setup_size_factor(ctx, self.cfg))
        if qty < 1:
            self.journal.record("reject", sig=live, note="position size rounds to 0")
            return False
        if self.dry_run:
            self.journal.record("dry_signal", sig=live, qty=qty)
            return True
        order_id = self.broker.submit_bracket(live, qty, self.day)
        risk.on_entry(live)
        self.tracked[live.symbol] = {"order_id": order_id, "sig": live, "qty": qty}
        self.journal.record("entry", sig=live, qty=qty, note=f"{live.reason} (order {order_id})")
        return True

    def sync(self, risk) -> None:
        """Detect fills, exits and dead orders for tracked symbols."""
        if not self.tracked:
            return
        positions = self.broker.positions()
        for sym, tr in list(self.tracked.items()):
            order = self.broker.order(tr["order_id"])
            filled = float(order.filled_qty or 0)
            status = getattr(order.status, "value", str(order.status))
            if filled == 0 and status in ("canceled", "expired", "rejected"):
                risk.on_cancel(sym)
                self.journal.record("cancel", sym, sig=tr["sig"], note=f"entry {status}")
                del self.tracked[sym]
            elif filled > 0 and sym not in positions:
                entry = float(order.filled_avg_price)
                exit_px = self.broker.last_exit_price(sym, tr["sig"].side)
                if exit_px is None:
                    log.warning("%s closed but no exit fill found; recording 0 P&L", sym)
                    exit_px = entry
                pnl = (exit_px - entry) * filled * tr["sig"].direction
                risk.on_exit(sym, pnl)
                self.journal.record("exit", sym, sig=tr["sig"], qty=filled, price=exit_px, pnl=pnl)
                del self.tracked[sym]
