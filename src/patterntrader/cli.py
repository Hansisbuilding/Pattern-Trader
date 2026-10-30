"""Command line: scan, approve, backtest, trade, account."""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime

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
    path = watchlist.save(now.date(), regime, notes, picks, others)
    print(path.with_suffix(".md").read_text())
    print(f"Saved {path}")


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

    md, _ = _clients(cfg)
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        path = ROOT / (args.universe or cfg.backtest.universe_file)
        symbols = [l.strip().upper() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    if args.no_news:
        cfg.backtest["use_news"] = False
    summary = run_backtest(cfg, md, symbols, date.fromisoformat(args.start), date.fromisoformat(args.end))
    out = write_results(summary, save_stats=not args.no_save_stats)
    print(render_summary(summary))
    print(f"Results in {out}")


def cmd_trade(cfg, args):
    from patterntrader.live import Trader

    Trader(cfg, dry_run=args.dry_run).run()


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
    for name, help_ in (("approve", "approve watchlist symbols for trading"), ("reject", "reject watchlist symbols")):
        a = sub.add_parser(name, help=help_)
        a.add_argument("symbols", nargs="*")
        a.add_argument("--all", action="store_true")
        a.add_argument("--day")
    s = sub.add_parser("show", help="print a watchlist report")
    s.add_argument("--day")

    b = sub.add_parser("backtest", help="replay the full process over history")
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)
    b.add_argument("--symbols", help="comma-separated; default is the universe file")
    b.add_argument("--universe", help="file with one symbol per line")
    b.add_argument("--no-news", action="store_true", help="skip news checks (news-gated strategies won't fire)")
    b.add_argument("--no-save-stats", action="store_true", help="don't update the edge-gate stats file")

    t = sub.add_parser("trade", help="trade approved symbols in the PAPER account until the close")
    t.add_argument("--dry-run", action="store_true", help="log signals without placing orders")
    sub.add_parser("account", help="show paper account status")

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    handlers = {
        "scan": cmd_scan,
        "approve": cmd_approve,
        "reject": lambda c, a: cmd_approve(c, a, "rejected"),
        "show": cmd_show,
        "backtest": cmd_backtest,
        "trade": cmd_trade,
        "account": cmd_account,
    }
    handlers[args.cmd](cfg, args)
