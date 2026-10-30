"""Command line: scan, approve, backtest, trade, account."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date, datetime, timedelta

from patterntrader import watchlist
from patterntrader.config import ET, ROOT, load_config, paper_credentials


def _clients(cfg):
    from patterntrader.data import MarketData
    from patterntrader.live import PaperBroker

    key, secret = paper_credentials()
    return MarketData(key, secret, cfg.data.cache_dir), PaperBroker(key, secret)


def cmd_scan(cfg, args):
    from patterntrader.edge import EdgeGate
    from patterntrader.scanner import analyze, coarse_universe, pick_watchlist
    from patterntrader.strategies import build_strategies

    md, broker = _clients(cfg)
    now = datetime.now(ET)
    rows = coarse_universe(cfg, md, broker.client, now.date())
    logging.info("coarse scan kept %d symbols", len(rows))
    regime, notes, contexts = analyze(cfg, md, rows, now, build_strategies(cfg), EdgeGate(cfg))
    picks = pick_watchlist(contexts, cfg)
    others = [c for c in contexts if c not in picks]
    mood = _mood(cfg, md, now.date())
    path = watchlist.save(now.date(), regime, notes, picks, others, mood=mood)
    from patterntrader.sentiment import day_size_factor, log_decisions

    factor, why = day_size_factor(*_mood_inputs(mood), cfg)
    log_decisions(now.date(), "scan", contexts, factor, why)
    if why:
        print("Sentiment: " + "; ".join(why))
    print(path.with_suffix(".md").read_text())
    print(f"Saved {path}")


def _mood(cfg, md, day):
    from patterntrader.sentiment import daily_mood

    try:
        return daily_mood(cfg, md, day)
    except Exception as exc:  # sentiment is context, never a reason to fail the scan
        logging.warning("market mood unavailable: %s", exc)
        return None


def _mood_inputs(mood):
    if not mood:
        return None, None, []
    return mood.get("mood"), (mood.get("vix") or {}).get("vix_term"), mood.get("events") or []


def cmd_mood(cfg, args):
    from patterntrader.sentiment import render_mood

    md, _ = _clients(cfg)
    day = date.fromisoformat(args.day) if args.day else datetime.now(ET).date()
    m = _mood(cfg, md, day)
    if m is None:
        raise SystemExit("market mood unavailable (see the warning above)")
    print("\n".join(render_mood(m)))
    v, r = m["vix"], m["rates"]
    if v.get("vix_term"):
        print(f"\nVIX {v['vix']:.1f}, {v['vix_ratio']:.2f}x its 20-day average; term structure {v['vix_term']:.2f}"
              f" ({'PANIC: short-term fear above 3-month' if v['vix_term'] > 1 else 'normal'})")
    if r.get("hy_spread") is not None:
        print(f"Credit spread {r['hy_spread']:.2f}% ({r['hy_spread_20d']:+.2f} in 20 days); "
              f"10-year {r['t10y']:.2f}% ({r['t10y_20d']:+.2f}); 2-year {r['t2y']:.2f}%")
    if m["events"]:
        print("Today: " + ", ".join(m["events"]))
    if m["reddit_top"]:
        print("Reddit's most-mentioned (24h): " + ", ".join(
            f"{t} {d['reddit_mentions']}" for t, d in m["reddit_top"][:10]))
    print(f"\nLogged to journal/mood/{m['day']}.json")


def cmd_approve(cfg, args, status="approved"):
    day = date.fromisoformat(args.day) if args.day else datetime.now(ET).date()
    symbols = [] if args.all else [s.upper() for s in args.symbols]
    if not symbols and not args.all:
        raise SystemExit("name symbols to change, or pass --all")
    changed = watchlist.set_status(day, symbols, status)
    print(f"{status}: {', '.join(changed) or 'nothing matched'}")


def cmd_show(cfg, args):
    day = date.fromisoformat(args.day) if args.day else datetime.now(ET).date()
    print(watchlist.path_for(day).with_suffix(".md").read_text())


def cmd_backtest(cfg, args):
    from patterntrader.backtest import render_summary, run_backtest, write_results

    md, broker = _clients(cfg)
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        path = ROOT / (args.universe or cfg.backtest.universe_file)
        symbols = [l.strip().upper() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    if args.no_news:
        cfg.backtest["use_news"] = False
    if args.no_sentiment and "sentiment" in cfg:  # baseline for comparison with the live setting
        cfg.sentiment["enabled"] = False
    if args.all_regimes:  # research mode: let every strategy trade in every non-chaos market
        every = ["trend_up", "trend_down", "range", "mixed"]
        for params in cfg.strategies.values():
            params["enabled"], params["market_regimes"], params["stock_regimes"] = True, every, every
    if args.account_equity:
        cfg.backtest["starting_equity"] = float(broker.account().equity)
    elif args.equity:
        cfg.backtest["starting_equity"] = args.equity
    end = date.fromisoformat(args.end) if args.end else datetime.now(ET).date() - timedelta(days=1)
    if args.days:
        spy = md.daily([cfg.regime.market_symbol], end - timedelta(days=args.days * 2 + 10), end, cfg.data.backtest_feed)
        sessions = sorted(set(spy[cfg.regime.market_symbol].index.date))
        start, end = sessions[-args.days], sessions[-1]
    elif args.start:
        start = date.fromisoformat(args.start)
    else:
        raise SystemExit("give --start or --days")
    logging.info(f"backtest {start} → {end}, starting equity ${cfg.backtest.starting_equity:,.2f}")
    summary = run_backtest(cfg, md, symbols, start, end)
    out = write_results(summary, save_stats=not (args.no_save_stats or args.all_regimes), tag=args.tag or "")
    print(render_summary(summary))
    print(f"Results in {out}")


def cmd_trade(cfg, args):
    from patterntrader.live import Trader

    if not args.dry_run and cfg.get("live", {}).get("github_only") and os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit(
            "This bot only places orders from GitHub Actions (live.github_only in config.yaml), "
            "so it can't double-trade the account from this machine. Use --dry-run to watch signals here."
        )
    stop_at = datetime.strptime(args.stop_at, "%H:%M").time() if args.stop_at else None
    Trader(cfg, dry_run=args.dry_run).run(stop_at=stop_at, resume=args.resume)


def cmd_account(cfg, args):
    _, broker = _clients(cfg)
    a = broker.account()
    print(f"account {a.account_number}  status {a.status}  equity ${float(a.equity):,.2f}  "
          f"buying power ${float(a.buying_power):,.2f}  (paper endpoint)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="patterntrader", description=__doc__)
    p.add_argument("--config", help="path to config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scan", help="scan the market and write today's watchlist for review")
    mo = sub.add_parser("mood", help="today's market mood and sentiment check (logged daily)")
    mo.add_argument("--day")
    for name, help_ in (("approve", "approve watchlist symbols for trading"), ("reject", "reject watchlist symbols")):
        a = sub.add_parser(name, help=help_)
        a.add_argument("symbols", nargs="*")
        a.add_argument("--all", action="store_true")
        a.add_argument("--day")
    s = sub.add_parser("show", help="print a watchlist report")
    s.add_argument("--day")

    b = sub.add_parser("backtest", help="replay the full process over history")
    b.add_argument("--start")
    b.add_argument("--end", help="default: yesterday")
    b.add_argument("--days", type=int, help="last N trading sessions ending at --end")
    b.add_argument("--equity", type=float, help="starting equity (default from config)")
    b.add_argument("--account-equity", action="store_true", help="start from the paper account's current equity")
    b.add_argument("--symbols", help="comma-separated; default is the universe file")
    b.add_argument("--universe", help="file with one symbol per line")
    b.add_argument("--no-news", action="store_true", help="skip news checks (news-gated strategies won't fire)")
    b.add_argument("--no-save-stats", action="store_true", help="don't update the edge-gate stats file")
    b.add_argument("--all-regimes", action="store_true",
                   help="research: enable every strategy in every market regime (never saves edge stats)")
    b.add_argument("--tag", help="suffix for the results folder name")
    b.add_argument("--no-sentiment", action="store_true", help="baseline: ignore the sentiment rules")

    t = sub.add_parser("trade", help="trade approved symbols in the PAPER account until the close")
    t.add_argument("--dry-run", action="store_true", help="log signals without placing orders")
    t.add_argument("--stop-at", help="HH:MM ET: hand off to a later --resume run instead of flattening")
    t.add_argument("--resume", action="store_true", help="continue the day a --stop-at run handed off")
    sub.add_parser("account", help="show paper account status")

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    handlers = {
        "scan": cmd_scan,
        "mood": cmd_mood,
        "approve": cmd_approve,
        "reject": lambda c, a: cmd_approve(c, a, "rejected"),
        "show": cmd_show,
        "backtest": cmd_backtest,
        "trade": cmd_trade,
        "account": cmd_account,
    }
    handlers[args.cmd](cfg, args)
