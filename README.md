# stock-robot

A daily swing-trading AI for S&P 500 stocks that paper-trades on Interactive Brokers.

Each weekday before the open it ranks every S&P 500 stock by expected return
over the next 5 trading days. It holds the top 20, long-only, weighted by inverse
volatility, and sends market-on-open orders to your **IBKR paper account**.
Every week it downloads fresh data and retrains itself.

All training data is free:

| Source | What | Notes |
|---|---|---|
| Yahoo Finance (`yfinance`) | Daily OHLCV since 2003, splits, dividends | Adjusted for total return |
| SEC EDGAR XBRL `companyfacts` | Revenue, earnings, cash flow, balance sheet | **Point-in-time**: each number becomes visible on the day it was filed |
| FRED | VIX, 10y-2y curve, 3m T-bill, high-yield spread | Lagged one day |
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

  Market-regime features are added raw: SPY trend, breadth, dispersion, VIX, credit spread, yield curve.
* **Target**: the cross-sectional rank of the forward return from the next open to the open `horizon` days later. This matches how the robot trades: it decides at the close and executes at the next open.
* **Models**:
  * LightGBM (70%).
  * Your original `FeedForward` network (LayerNorm + SiLU MLP) retrained on the ranked features (30%). It trains on the M-series GPU (MPS).
  * Their per-day ranks are blended.
* **Backtest**:
  * Walk-forward: the model is retrained every 6 months and only ever sees data from before its trading period.
  * A purge gap keeps training labels from overlapping the test window.
  * Simulated day by day: decide at the close, trade at the next open, 5.5 bps cost on turnover, idle cash earns the T-bill rate.
* **Portfolio**:
  * Holds the top 20. A current holding is kept while it still ranks inside the top 40, which cuts turnover.
  * Rebalances every 5 trading days.
  * Maximum 8% per name, 2% cash buffer.
  * Exposure is halved when SPY is below its 200-day average.
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

## Backtest results (walk-forward, out-of-sample, after costs)

Run on 2026-09-22 with the default config. Model retrained every 6 months; Jan 2012 – Sep 2026.

| | CAGR | Vol | Sharpe | Max DD | Rank IC | Turnover/yr |
|---|---|---|---|---|---|---|
| **Ensemble (GBM 70% + NN 30%)** | **15.5%** | 17.4% | **0.82** | **−30.5%** | **0.025** | 53× |
| GBM only | 15.5% | 18.5% | 0.78 | −33.6% | 0.022 | 52× |
| NN only | 14.4% | 16.4% | 0.80 | −27.1% | 0.022 | 49× |
| SPY | 15.2% | 16.5% | 0.83 | −33.7% | | |
| Equal-weight S&P 500 | 14.0% | 17.3% | 0.74 | −40.0% | | |

What these numbers mean:

* The model has a small but real signal: rank IC ≈ 0.025, positive in about two-thirds of half-years.
* It beats the equal-weight S&P and roughly matches SPY with a smaller drawdown.
* It does **not** beat SPY by much: 2012–2026 was dominated by mega-cap tech.
* It lagged SPY in 2014–2019 and caught up after 2020.

What was tried and rejected (see git history):

* Raw interest-rate and VIX levels as features. The trees used them to memorize eras, giving IC 0.016 and a −40% drawdown.
* A 21-day horizon. IC fell to 0.005.

Ideas worth testing next:

* Smooth scores over a few days to cut the 53× turnover. Costs are about 2.9%/yr.
* Earnings-date features.
* A larger `top_k`.

## Caveats

* Backtests flatter. About 300 of the historical members were delisted and Yahoo no longer serves their prices, so some survivorship bias remains. Treat the backtest as an upper bound.
* Paper fills are simulated by IBKR and are kinder than real ones.
* This is a research project, not financial advice. Keep it on paper until it has months of live paper results that agree with the backtest.
