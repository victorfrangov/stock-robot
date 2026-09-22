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

The goal was to beat SPY without fooling ourselves, so the research followed a fixed protocol:

* **Dev period, 2012–2019.** Every design choice was made on this period only.
* **Sealed holdout, 2020 – Sep 2026.** Opened exactly once, after the config was frozen and committed (`8bfbaf7`).
* **Walk-forward scores throughout.** Each 6-month segment is scored by a model trained only on earlier data, with a purge gap.
* **Costs.** 5.5 bps per trade, with a check at 2× costs.
* **Controls.** The same construction was run on random scores and on pure size ranking.

The research lab (`research/lab.py`) prints dev-period numbers unless `--reveal` is passed.

### Final, frozen strategy (after costs)

| | CAGR | SPY | Excess/yr | Sharpe (SPY) | Max DD (SPY) | Tracking err. | Years beating SPY |
|---|---|---|---|---|---|---|---|
| Dev 2012–2019 | 19.7% | 14.8% | **+4.9%** | 1.18 (1.09) | −23.6% (−19.4%) | 6.1% | 6 of 8 |
| **Holdout 2020–2026** | **21.2%** | **15.6%** | **+5.6%** | 0.73 (0.68) | −33.8% (−33.7%) | 14.4% | 3 of 7 |
| Holdout at 2× costs | 19.1% | 15.6% | +3.5% | 0.67 (0.68) | −33.9% | 14.4% | 2 of 7 |

Yearly excess return vs SPY:

| Year | Excess | Year | Excess |
|---|---|---|---|
| 2012 | −7.2 | 2020 | **+56.7** |
| 2013 | +9.6 | 2021 | +1.1 |
| 2014 | −3.9 | 2022 | +9.2 |
| 2015 | +12.4 | 2023 | −10.1 |
| 2016 | +0.4 | 2024 | −0.8 |
| 2017 | +12.2 | 2025 | −8.7 |
| 2018 | +8.4 | 2026 (YTD) | −2.9 |
| 2019 | +8.8 | | |

**Verdict:** the strategy beat SPY on the dev period and on the sealed holdout. But the holdout
win is carried by 2020, when the model's short-term-reversal signal caught the post-crash rebound.
From 2023 on it has lagged SPY every year: that was the concentrated mega-cap/AI rally, a hard
regime for a reversal-driven stock picker. It is a real but regime-dependent edge, not a
reliable yearly outperformer. The only honest remaining test is forward paper trading.

### What the research found (dev period only)

| Finding | Effect on dev excess return vs SPY |
|---|---|
| Regime filter (halve exposure below SPY's 200-day average) | cost about 2.5%/yr; removed |
| Score smoothing and hysteresis instead of trading every signal change | turnover 56× → 23×, about +3%/yr |
| Cap weighting within picks, rather than inverse-vol or equal | about +3–4%/yr; keeps size exposure close to SPY |
| Control: same portfolio construction on **random** scores | −2.5%/yr, so the edge isn't structural |
| Control: same portfolio construction on **size** only | +1.0%/yr |
| Earnings-date (SEC 8-K) and industry-momentum features | IC 0.0195 → 0.0234 (t=1.3), no better top-50 spread; off |
| Sector-neutral target, multi-horizon (5/10/21d) target | no improvement; off |

The model signal is mostly short-term (top-50 spread +5.9%/yr gross raw, +0.8% after 5-day
smoothing), so it has to be harvested with light smoothing and wide hold buffers.

### Risks to know about

* **Sector concentration.** There is no sector cap. The Sep 2026 portfolio is heavy in semiconductors and storage (MU, INTC, LRCX, AMAT, KLAC…).
* **Lumpy tracking error.** Holdout tracking error is 14%, so expect multi-year stretches behind SPY.
* **Survivorship bias.** Tickers without a current SEC mapping are excluded, which removes the leak but keeps some survivorship bias.
* **Next steps, forward-tested only.** Ideas like a sector cap or a core-satellite (e.g. 70% SPY + 30% strategy) should be judged on forward paper results. The historical holdout has been used.

## Caveats

* Backtests flatter. About 300 of the historical members were delisted and Yahoo no longer serves their prices, so some survivorship bias remains. Treat the backtest as an upper bound.
* Paper fills are simulated by IBKR and are kinder than real ones.
* This is a research project, not financial advice. Keep it on paper until it has months of live paper results that agree with the backtest.
