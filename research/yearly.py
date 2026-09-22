"""Yearly strategy vs SPY for the frozen config (run only after the holdout was opened)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robot.backtest import simulate  # noqa: E402
from robot.config import Config, load_config  # noqa: E402
from robot.data.macro import load_macro  # noqa: E402
from robot.data.prices import load_prices, total_return_ohlc  # noqa: E402
from robot.pipeline import get_panel  # noqa: E402

cfg = load_config()
panel = get_panel(cfg)[["date", "ticker", "raw_vol_63", "raw_price", "raw_mcap"]]
prices, macro = load_prices(cfg), load_macro(cfg)
scores = pd.read_parquet(sys.argv[1])
sim = simulate(cfg, scores, panel, prices, macro)
spy = total_return_ohlc(prices)["close"]["SPY"].pct_change(fill_method=None).reindex(sim.returns.index)
yr = pd.DataFrame({"strategy": sim.returns, "spy": spy}).groupby(sim.returns.index.year).apply(lambda g: (1 + g).prod() - 1)
yr["excess"] = yr["strategy"] - yr["spy"]
print((yr * 100).round(1).to_string())
out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
if out:
    pd.DataFrame({"strategy": (1 + sim.returns).cumprod(), "spy": (1 + spy.fillna(0)).cumprod(),
                  "holdings": sim.n_holdings, "turnover": sim.turnover}).to_csv(out)
    last = max(sim.weights)
    print("\nlatest target portfolio", last.date())
    print(sim.weights[last].sort_values(ascending=False).head(15).round(4).to_string())
