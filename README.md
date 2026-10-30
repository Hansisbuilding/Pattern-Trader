# Pattern Trader

A disciplined, regime-aware day-trading bot for **Alpaca paper accounts**. It scans the
whole US stock market for setups, writes a watchlist for you to review, and trades only the
symbols you approve. It uses six classic intraday playbooks, and each one is allowed to run
only in the market situation it is built for.

> Paper trading only. The bot refuses live keys (`AK…`) and always connects to Alpaca's
> paper endpoint. This is a research tool, not financial advice.

## How a day works

```
 08:45  scan      Full-market pass → in-play stocks → situation check → watchlist (≤5)
        review    You read watchlists/<date>.md and approve what you agree with
 09:36  trade     Re-checks each approved name with real opening volume, then watches 1-min bars
 …      entries   Bracket orders (stop + target attached), sized from the stop
 15:50  flatten   Everything closed. No overnight holds.
```

```bash
patterntrader scan                 # write today's watchlist
patterntrader approve NVDA AMD     # or: approve --all / reject TSLA
patterntrader trade --dry-run      # log signals only
patterntrader trade                # paper orders
```

## The discipline layer

The bot spends most of its logic deciding when *not* to trade.

**1. Market regime (SPY), judged before the open**

| Regime | Meaning | What's allowed |
|---|---|---|
| `trend_up` / `trend_down` | Price, 20 EMA and 50 SMA aligned, ADX ≥ 20 | Momentum and trend pullbacks, in the trend's direction only |
| `range` | ADX < 20 | Breakouts, mean reversion and no-news gap fades |
| `mixed` | Trend turning | Very little |
| `chaos` | SPY gap ≥ 1.5% or volatility in the top 5% of the year | **Nothing. Sit out.** |

**2. Each strategy has a situation it must match** (see `config.yaml`, `market_regimes` / `stock_regimes`)

| Strategy | Needs | Entry | Stop | Target |
|---|---|---|---|---|
| `gap_and_go` | Gap ≥ 4% **with news**, RVOL ≥ 3 | New high of day above the premarket high, on volume, above VWAP | Recent 5-bar low | 2R |
| `orb` | In play (gap or ATR), RVOL ≥ 2 | Break of the 15-min opening range on volume, right side of VWAP | Range low (midpoint if the range is wide) | 2R |
| `vwap_pullback` | Trending stock **and** trending market | First clean bounce off VWAP | Below the pullback | 2R |
| `bull_flag` | RVOL ≥ 2 | Breakout of a shallow, low-volume flag after a sharp pole | Flag low | 2R |
| `gap_fade` | Gap ≥ 3% with **no** news, range market | Gap fails: reclaims the opening range and VWAP | Low of day | 75% gap fill (≥ 1.5R) |
| `mean_reversion` | Range-bound stock in a range market, no news, normal volume | Close beyond the 5-min Bollinger Band with extreme RSI, then a turn back | Swing low | Band midline (≥ 1.2R) |

On top of that, every signal must keep its stop within a fraction of the stock's daily ATR (a stop
that far away means bad trade location), and it must meet its minimum reward:risk.

**3. Risk rules** (`risk:` in `config.yaml`)

- Risk 0.5% of equity per trade, sized from the stop. A single position is capped at 20% of equity.
- Daily loss of 2% means flatten everything and stop for the day.
- Two losses in a row means no new trades for the day.
- Max 4 trades per day and 2 open at once. No re-entering a symbol that just stopped you out.
- No entries before 09:35 or after 15:00. Skip if the spread is wide or price has run away from the signal.
- Long-only by default (`allow_short: false`).

**4. Edge gate.** A strategy only trades live after a backtest shows ≥ 30 trades, positive expectancy,
a profit factor ≥ 1.1, and a positive edge **in both halves** of the test period. With no backtest
results, nothing trades. Backtest first.

## Setup

1. Create a new Alpaca account, switch to **Paper Trading**, and generate API keys (they start with `PK`).
2. Install:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   cp .env.example .env     # paste your paper keys in
   patterntrader account    # sanity check
   ```

## Backtesting

```bash
patterntrader backtest --start 2025-01-01 --end 2026-09-30
```

This replays the live process day by day: it judges SPY's regime, picks the watchlist with the same
scanner rules using only information known at 09:35, then runs the same strategy and risk code
over 1-minute bars. Fills are conservative: entry at the next bar's open plus slippage, stops filled
worse on gaps, and stop-first when a bar touches both stop and target.

Results go to `results/backtest_<start>_<end>/` (`trades.csv`, `summary.md`), and the edge-gate file
`results/strategy_stats.json` is updated. The summary breaks results down by strategy **and by market
regime**, so you can see whether each playbook works in the situations it's allowed to trade.

Data is cached in `.cache/`. The first run of a long period downloads a lot of minute data.

Honest-testing tips:
- Run one period to tune, then judge on a **different** period you haven't looked at.
- The default universe (`universe/liquid_us.txt`) is today's survivors, which flatters results. Widen it.
- News-gated strategies (`gap_and_go`, `gap_fade`) need `use_news: true`. For mega-caps there is almost always
  *some* headline, so "has news" is a loose catalyst test.

## GitHub

- `tests` workflow: runs the test suite on every push.
- `trading day` workflow: runs the whole paper-trading day on GitHub, no machine of your own needed.
  The morning job scans at ~08:50 ET and trades from the open; at 12:00 ET it hands off to the
  afternoon job (GitHub caps a job at 6 hours), which keeps trading and flattens at 15:50. Picks are
  auto-approved (`watchlist.require_approval: false`). After the close the journal and watchlists are
  committed to the `trading-log` branch.
  - Secrets (Settings → Secrets and variables → Actions): `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
    (paper keys), `SEC_USER_AGENT`, and optionally `ALPHAVANTAGE_API_KEY`, `FRED_API_KEY`.
  - Pause: add the repository variable `BOT_PAUSED` = `true`.
  - Manual run: Actions → trading day → Run workflow (dry run by default: logs signals, no orders).
- Don't also run `patterntrader trade` on your own machine while the workflow is on: it would trade
  the same account twice.

`.env`, `.cache/`, `results/`, `journal/` and `watchlists/` are git-ignored. Keys never get committed.

## Known limitations

- The free Alpaca plan gives real-time data from **IEX only**, about 2–3% of volume. Live RVOL compares IEX
  against IEX history, so the ratio stays fair, but 1-minute bars are thinner than the consolidated tape.
  Backtests use full SIP history. A paid data plan (`live_feed: sip`) makes live behave more like the backtest.
- Paper fills are optimistic compared with real markets, especially in fast, gappy stocks.
- Order-flow / Level 2 tape reading isn't implemented (Alpaca doesn't provide depth). Head-and-shoulders and
  double tops are left out: they're unreliable intraday and hard to define without hindsight.
- Shortability in backtests is assumed. Live, it's checked against Alpaca's easy-to-borrow list.

## Layout

```
config.yaml              every rule and threshold
src/patterntrader/
  regime.py              market + stock regime classification
  context.py             per symbol-day setup context, eligibility and scoring
  scanner.py             full-market coarse scan and situation analysis
  strategies/            momentum.py, reversion.py (+ base.py gating/validation)
  risk.py                sizing and daily discipline rules (shared by backtest and live)
  backtest.py            event-driven intraday simulator and reports
  live.py                paper broker and live trading loop
  edge.py                backtest-based edge gate
  watchlist.py journal.py data.py indicators.py cli.py
tests/                   unit tests + end-to-end runs on synthetic data
```
