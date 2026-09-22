# stock-robot

A daily swing-trading AI for S&P 500 stocks that paper-trades on Interactive Brokers.

Each weekday before the open it ranks every S&P 500 stock by expected return
over the next 5 trading days. It holds the top 50, long-only and weighted by market
cap, and sends market-on-open orders to your **IBKR paper account**.
Every week it downloads fresh data and retrains itself.

All training data is free:

| Source | What | Notes |
|---|---|---|
| Yahoo Finance (`yfinance`) | Daily OHLCV since 2003, splits, dividends | Adjusted for total return |
| SEC EDGAR XBRL `companyfacts` | Revenue, earnings, cash flow, balance sheet | **Point-in-time**: each number becomes visible on the day it was filed |
| FRED | VIX, 10y-2y curve, 3m and 10y Treasury yields | Lagged one day |
| SEC EDGAR submissions | Earnings-release dates (8-K item 2.02) | Optional research features, off by default |
| [fja05680/sp500](https://github.com/fja05680/sp500) | Historical S&P 500 membership | Removes most survivorship bias |

## How it works

```
data  ->  features (≈60)  ->  LightGBM + neural net ensemble  ->  portfolio  ->  risk checks  ->  IBKR paper
```

* **Features** are cross-sectional percentile ranks, compared day by day:
  * momentum at 1d/1w/1m/3m/6m/12m and 12-1
  * short-term reversal, RSI, ATR
  * volatility, downside volatility, skew, beta, idiosyncratic volatility
  * distance from 52-week highs and lows, moving-average gaps, overnight vs intraday returns
  * liquidity (dollar volume, Amihud)
  * sector-relative momentum
  * value, quality and growth from SEC filings (earnings, sales and FCF yield, book-to-market, ROE, ROA, margins, leverage, accruals, revenue and earnings growth, days since the last filing)

  Market-regime features are added raw: SPY trend, volatility, breadth, dispersion, change in VIX.
  Raw rate and VIX *levels* are excluded because the trees used them to memorize eras.
* **Target**: the cross-sectional rank of the forward return from the next open to the open `horizon` days later. This matches how the robot trades: it decides at the close and executes at the next open.
* **Models**:
  * LightGBM (70%).
  * Your original `FeedForward` network (LayerNorm + SiLU MLP) retrained on the ranked features (30%). It trains on the M-series GPU (MPS).
  * Their per-day ranks are blended.
* **Backtest**:
  * Walk-forward: the model is retrained every 6 months and only ever sees data from before its trading period.
  * A purge gap keeps training labels from overlapping the test window.
  * Simulated day by day: decide at the close, trade at the next open, 5.5 bps cost on turnover, idle cash earns the T-bill rate.
* **Portfolio** (frozen 2026-09-22 from dev-period research; see below):
  * Scores are smoothed with a 1-day half-life EMA.
  * Holds the top 50, cap-weighted, max 10% per name. A current holding is kept while it ranks inside the top 150.
  * Rebalances every 5 trading days. Fully invested, no regime filter.
  * An enhanced-index "tilt" mode is also available (`portfolio.mode: tilt`).
* **Risk** (enforced in code, independent of the model):
  * Refuses any account that isn't a paper account (`DU…`/`DF…`) and refuses the live ports 7496 and 4001.
  * Refuses any single buy over $50k.
  * Stops buying for the day after a 4% daily loss.
  * Flattens and halts at a 25% drawdown from peak.
  * `robot kill` stops all trading immediately.

## Setup

```bash
cd ~/dev/stock-robot
python3 -m venv .venv && source .venv/bin/activate
brew install libomp                     # LightGBM runtime on macOS
pip install -e ".[nn,dev]"              # add -e ../ib_fundamental for `robot inspect --ib`
export SEC_USER_AGENT="Your Name your@email.com"   # SEC asks bots to identify themselves
```

In **IB Gateway** or **TWS**, log in to the **paper** account and enable the API:
*Configure → Settings → API → Enable ActiveX and Socket Clients*. Also untick
*Read-Only API*. Ports: Gateway paper = `4002` (the default here), TWS paper = `7497`.
Set `broker.port` in `config.yaml` or `IB_PORT=7497`.

## Usage

```bash
robot data                 # first run downloads ~20 years for ~700 tickers + SEC filings (~5 min)
robot backtest --no-nn     # walk-forward backtest, GBM only (~10 min); drop --no-nn for the ensemble
robot train                # train the production model on everything
robot signals              # today's ranking, no broker needed
robot trade --dry-run      # connect to paper, show the orders it would send
robot trade                # actually send them
robot status               # account, positions, P/L, pending orders, recent runs
robot inspect NVDA --ib    # feature breakdown + IBKR fundamentals via ib_fundamental
robot kill / robot resume  # emergency stop / clear halt
robot flatten --yes        # sell everything
```

### Running it automatically

```bash
robot install-service
launchctl load -w ~/Library/LaunchAgents/com.stockrobot.scheduler.plist
```

The scheduler trades at 09:00 US/Eastern on weekdays with market-on-open orders.
It refreshes data and retrains on Saturdays. IB Gateway must be running and
logged in; use [IBC](https://github.com/IbcAlpha/IBC) if you want Gateway to
log in and restart by itself.

The robot manages **every stock position in the paper account**. Anything it
didn't choose will be sold at the next rebalance.

## Layout

```
robot/
  config.py          defaults + config.yaml + env overrides
  data/              universe, prices (yfinance), fundamentals (EDGAR), macro (FRED)
  features.py        wide feature computation -> long ranked panel
  models.py          GBM, NN (ported from the old models.py), ensemble
  backtest.py        walk-forward training, daily simulator, metrics, report
  portfolio.py       scores -> target weights (shared by backtest and live)
  risk.py            kill switch, paper-account guard, drawdown/daily-loss limits
  broker.py          ib_async execution (MOO before the open, Adaptive during RTH)
  journal.py         SQLite log of runs, equity, orders, signals
  live.py            the daily trading job
  scheduler.py       daily trade + weekly retrain loop
  cli.py             `robot` command
legacy/              the original scripts and notebooks
tests/
```

Everything the robot downloads or produces lives in `data/` (gitignored):
`data/reports/backtest-*/` holds `report.json`, `backtest.png`, the equity
curve and feature importances. `data/state/journal.sqlite` holds the trading
history.

## Results: can it beat the S&P 500?

Short answer: **not demonstrably.** The model has a small, real stock-ranking signal, but every
honest measurement leaves "beats SPY" open. The figures below follow an independent adversarial
audit that found and fixed data and protocol problems, so they are lower than this README's
earlier draft.

### What the audit found

Six auditors (leakage, data quality, simulator, live parity, statistics, protocol) reported 30
findings. The material ones, all fixed or disclosed:

| Finding | Fix |
|---|---|
| Market caps corrupted for ~20 names: some SEC share counts were 1e6× too large, and GOOGL's 20:1 split was applied twice; cap weighting pinned them at 10% | share-count scale filter; splits applied from the filing date; 5× rolling guard |
| Ticker renames (FB→META, UTX→RTX, ANTM→ELV and 20 more) made companies vanish from the universe for their earlier years | rename map in `robot/data/universe.py` |
| Recycled symbols pointed at the wrong instrument (PARA became a micro-cap with closes near 100,000) | feed validation inside each membership window |
| Live risk config: a $50k order cap rejected every 10% position on a $1M paper account; a 25% drawdown stop would have flattened the book on 2020-03-12 | 15%-of-equity cap; 45% catastrophe stop; loss days postpone the rebalance instead of half-executing it |
| **The holdout was not sealed.** The final model's 2012–2026 yearly table was printed hours before the dev/holdout split was declared, and the decision to drop rate-level features came from a full-period run whose gain was mostly 2020 | the exclusion was re-tested on 2012–2019 only (below); 2020–2026 results are reported as **post-hoc**, not as a clean holdout |
| **Survivorship bias that free data cannot fix.** About 55% of the companies that left the index have no Yahoo history, so no delisting ever hits the simulator; a survivor-only equal-weight universe beat the true equal-weight index by about 1.3–1.5%/yr on 2012–2019 | disclosed; treat every excess return here as an upper bound |

### Corrected results, 2012–2019 (after 5.5 bps costs, walk-forward)

Frozen construction (cap-weighted top 50, 1-day score smoothing, hold buffer 150, weekly):

| Model variant | Rank IC | Top-50 spread t | Excess vs SPY | Info. ratio | Years ahead |
|---|---|---|---|---|---|
| **Frozen: rate-level features excluded** | 0.021 | 1.6 | **+4.3%/yr** | 0.63 | 5 of 8 |
| Rate levels included | 0.024 | 1.8 | +1.7%/yr | 0.28 | 5 of 8 |
| Earnings-date + industry-momentum features on | 0.025 | 1.5 | +1.5%/yr | 0.27 | 4 of 8 |

The three variants rank stocks about equally well (paired IC differences have t ≈ 0.5). The
frozen one produces the best portfolio, but that is the noisy metric ~39 configurations were
tuned on, so it is weak evidence; the best of 30 random tries would show a t-stat of about 2.

### Post-hoc 2020–2026 (not a clean holdout)

FILL_POSTHOC

### Statistical reality

* The earlier "+5.6%/yr on the holdout" had t = 1.2 and a bootstrap 95% CI of [−6%, +19%].
  Excluding 2020 it was **−1.5%/yr**, and 2023–2026 was −6%/yr.
* About 40% of the raw excess is beta (1.15 vs SPY), not stock selection; CAPM alpha is not
  significant in any period.
* The strategy's edge is a short-term reversal signal that pays off in sharp rebounds (2020)
  and lags in narrow mega-cap rallies (2023–2025).

### Verdict

The model ranks stocks a little better than chance (IC ≈ 0.02, t ≈ 3.7), and the portfolio built
on it kept up with or modestly beat SPY on the corrected 2012–2019 data. Whether that survives
real trading is unknown; the only honest test left is the forward paper record, which starts now.

## Caveats

* Backtests flatter. About 300 of the historical members were delisted and Yahoo no longer serves their prices, so some survivorship bias remains. Treat the backtest as an upper bound.
* Paper fills are simulated by IBKR and are kinder than real ones.
* This is a research project, not financial advice. Keep it on paper until it has months of live paper results that agree with the backtest.
