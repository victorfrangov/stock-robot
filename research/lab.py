"""Portfolio-construction lab with a sealed holdout.

Research protocol (fixed before any tuning, 2026-09-22):
  * DEV     = 2012-01-01 .. 2019-12-31   - every design choice is made here
  * HOLDOUT = 2020-01-01 .. end          - evaluated once, with the frozen config

Usage:
  python research/lab.py --scores <walk-forward scores.parquet> --variants research/variants.yaml
  python research/lab.py ... --only final --reveal     # the one-time holdout evaluation
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot.backtest import metrics, simulate  # noqa: E402
from robot.config import Config, load_config  # noqa: E402
from robot.data import universe  # noqa: E402
from robot.data.macro import load_macro  # noqa: E402
from robot.data.prices import load_prices, total_return_ohlc  # noqa: E402
from robot.pipeline import get_panel  # noqa: E402

DEV = ("2012-01-01", "2019-12-31")
HOLDOUT = ("2020-01-01", "2100-01-01")


def active_stats(r: pd.Series, bench: pd.Series, rf: pd.Series | None) -> dict:
    r, bench = r.align(bench, join="inner")
    m = metrics(r, rf)
    b = metrics(bench, rf)
    act = r - bench
    te = act.std() * np.sqrt(252)
    yr = pd.DataFrame({"s": r, "b": bench}).groupby(r.index.year).apply(lambda g: (1 + g).prod() - 1)
    return {
        "cagr": m["cagr"], "spy_cagr": b["cagr"], "excess_cagr": m["cagr"] - b["cagr"],
        "sharpe": m["sharpe"], "spy_sharpe": b["sharpe"], "maxdd": m["max_drawdown"],
        "spy_maxdd": b["max_drawdown"], "te": te, "ir": act.mean() * 252 / te if te > 0 else np.nan,
        "yrs_beat": float((yr["s"] > yr["b"]).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--reveal", action="store_true", help="also print the sealed HOLDOUT period")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    base = load_config()
    # Only raw_* columns (price, vol, mcap) are used here; they are identical in every
    # panel variant, so reuse the plain one instead of rebuilding feature experiments.
    pcfg = Config(copy.deepcopy(dict(base)))
    pcfg["model"] = {**pcfg["model"], "industry_momentum": False, "earnings_features": False}
    panel = get_panel(pcfg)[["date", "ticker", "raw_vol_63", "raw_price", "raw_mcap"]]
    prices = load_prices(base)
    macro = load_macro(base)
    scores = pd.read_parquet(a.scores)
    o = total_return_ohlc(prices)
    spy = o["close"][universe.BENCHMARK].pct_change(fill_method=None)
    rf = macro["DGS3MO"].reindex(o["close"].index.union(macro.index)).ffill() / 100 / 252

    variants = yaml.safe_load(Path(a.variants).read_text())
    rows = []
    for name, overrides in variants.items():
        if a.only and name not in a.only:
            continue
        cfg = Config(copy.deepcopy(dict(base)))
        for section, vals in (overrides or {}).items():
            cfg[section] = {**cfg[section], **vals}
        sim = simulate(cfg, scores, panel, prices, macro, score_col=a.score_col)
        turn = sim.turnover
        for label, (lo, hi) in [("DEV", DEV)] + ([("HOLDOUT", HOLDOUT)] if a.reveal else []):
            sl = slice(lo, hi)
            st = active_stats(sim.returns[sl], spy[sl], rf)
            st["turnover"] = float(turn[sl].sum() / (len(turn[sl]) / 252))
            st["holdings"] = float(sim.n_holdings[sl].mean())
            rows.append({"variant": name, "period": label, **st})
        print(f"done {name}", file=sys.stderr, flush=True)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    fmt = {c: "{:+.2%}".format for c in ["cagr", "spy_cagr", "excess_cagr", "maxdd", "spy_maxdd", "te", "yrs_beat"]}
    shown = df.copy()
    for c, f in fmt.items():
        shown[c] = shown[c].map(f)
    for c in ["sharpe", "spy_sharpe", "ir", "turnover", "holdings"]:
        shown[c] = shown[c].map("{:.2f}".format)
    print(shown.to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
