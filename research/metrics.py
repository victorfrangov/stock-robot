"""Selection metrics for research experiments. Everything here is restricted to the DEV period.

The pre-declared selection rule for model experiments is `top50_t` (t-stat of the
non-overlapping top-50-minus-universe forward-return spread at score half-life 1).
The portfolio statistics use the frozen construction from config.yaml.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot.backtest import metrics as perf, simulate  # noqa: E402
from robot.config import Config  # noqa: E402
from robot.data.macro import load_macro  # noqa: E402
from robot.data.prices import load_prices, total_return_ohlc  # noqa: E402
from robot.portfolio import smooth_scores  # noqa: E402

DEV = ("2012-01-01", "2019-12-31")
HOLDOUT = ("2020-01-01", "2100-01-01")
PLAIN_PANEL = "panel_h5_rank_im0_ea0.parquet"  # raw_* columns are identical in every panel variant
SIM_COLS = ["date", "ticker", "raw_vol_63", "raw_price", "raw_mcap"]


def period(df: pd.DataFrame, which: str = "DEV") -> pd.DataFrame:
    lo, hi = DEV if which == "DEV" else HOLDOUT
    return df[(df["date"] >= lo) & (df["date"] <= hi)]


def daily_ic(scores: pd.DataFrame, col: str = "score") -> pd.Series:
    s = scores.dropna(subset=[col, "target"])
    return s.groupby("date").apply(lambda g: g[col].corr(g["target"]), include_groups=False)


def ic_stats(scores: pd.DataFrame, col: str = "score") -> dict:
    ic = daily_ic(scores, col)
    blocks = ic.groupby(np.arange(len(ic)) // 5).mean()  # overlapping 5-day labels -> 5-day blocks
    return {"ic": float(ic.mean()), "ic_t": float(blocks.mean() / blocks.std() * np.sqrt(len(blocks))),
            "ic_pos": float((ic > 0).mean())}


def top_spread(scores: pd.DataFrame, col: str = "score", n: int = 50, halflife: float = 1, horizon: int = 5) -> dict:
    """Top-n minus universe forward return per non-overlapping rebalance (raw returns, so any target is comparable)."""
    s = scores[["date", "ticker", "fwd_ret", col]]
    S = smooth_scores(s.pivot(index="date", columns="ticker", values=col), halflife)
    R = s.pivot(index="date", columns="ticker", values="fwd_ret").reindex_like(S)
    out = []
    for d in S.index[::horizon]:
        sc, r = S.loc[d].dropna(), R.loc[d]
        top = sc.nlargest(n).index
        out.append(r[top].mean() - r[sc.index].mean())
    x = pd.Series(out).dropna()
    return {f"top{n}_bp": float(x.mean() * 1e4), f"top{n}_t": float(x.mean() / x.std() * np.sqrt(len(x))),
            f"top{n}_hit": float((x > 0).mean()), f"top{n}_ann": float((1 + x.mean()) ** (252 / horizon) - 1)}


def load_sim_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(cfg.root / "features" / PLAIN_PANEL, columns=SIM_COLS)
    return panel, load_prices(cfg), load_macro(cfg)


def portfolio_stats(cfg: Config, scores: pd.DataFrame, col: str = "score", which: str = "DEV",
                    inputs: tuple | None = None) -> dict:
    """Frozen-construction portfolio vs SPY on one period, after costs."""
    panel, prices, macro = inputs or load_sim_inputs(cfg)
    sim = simulate(cfg, scores, panel, prices, macro, score_col=col)
    o = total_return_ohlc(prices[prices["ticker"] == "SPY"])
    spy = o["close"]["SPY"].pct_change(fill_method=None)
    rf = macro["DGS3MO"].reindex(spy.index.union(macro.index)).ffill() / 100 / 252 if "DGS3MO" in macro else None
    lo, hi = DEV if which == "DEV" else HOLDOUT
    r = sim.returns[lo:hi]
    b = spy.reindex(r.index)
    m, mb = perf(r, rf), perf(b, rf)
    act = r - b
    te = act.std() * np.sqrt(252)
    yr = pd.DataFrame({"s": r, "b": b}).groupby(r.index.year).apply(lambda g: (1 + g).prod() - 1)
    return {"cagr": m["cagr"], "spy_cagr": mb["cagr"], "excess": m["cagr"] - mb["cagr"], "te": float(te),
            "ir": float(act.mean() * 252 / te) if te > 0 else float("nan"), "sharpe": m["sharpe"],
            "spy_sharpe": mb["sharpe"], "maxdd": m["max_drawdown"], "spy_maxdd": mb["max_drawdown"],
            "turnover": float(sim.turnover[lo:hi].sum() / (len(r) / 252)),
            "yrs_beat": float((yr["s"] > yr["b"]).mean()),
            "yearly_excess": {int(y): float(v) for y, v in (yr["s"] - yr["b"]).items()}}


def evaluate(cfg: Config, scores: pd.DataFrame, col: str = "score", which: str = "DEV",
             inputs: tuple | None = None) -> dict:
    """The full DEV scorecard for one score column."""
    sc = period(scores, which)
    out = {"period": which, "col": col, "n_days": int(sc["date"].nunique())}
    out.update(ic_stats(sc, col))
    out.update(top_spread(sc, col, n=30))
    out.update(top_spread(sc, col, n=50))
    out.update(portfolio_stats(cfg, sc, col, which, inputs))
    return out


def with_overrides(cfg: Config, overrides: dict | None) -> Config:
    c = Config(copy.deepcopy(dict(cfg)))
    for section, vals in (overrides or {}).items():
        c[section] = {**c[section], **vals} if isinstance(vals, dict) and isinstance(c.get(section), dict) else vals
    return c
