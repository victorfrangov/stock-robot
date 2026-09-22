"""Top-N minus universe forward return per rebalance date (DEV only), for model selection.

Uses raw forward returns, so models trained on different targets are comparable.
Non-overlapping: one observation every `horizon` sessions.
usage: top_spread.py scores.parquet[:col] ... [--n 20] [--halflife 10]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robot.portfolio import smooth_scores  # noqa: E402

DEV_END = "2020-01-01"
ap = argparse.ArgumentParser()
ap.add_argument("specs", nargs="+")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--halflife", type=float, default=10)
ap.add_argument("--horizon", type=int, default=5)
a = ap.parse_args()

for spec in a.specs:
    path, _, col = spec.partition(":")
    col = col or "score"
    s = pd.read_parquet(path, columns=["date", "ticker", "fwd_ret", col])
    s = s[s["date"] < DEV_END]
    S = smooth_scores(s.pivot(index="date", columns="ticker", values=col), a.halflife)
    R = s.pivot(index="date", columns="ticker", values="fwd_ret").reindex_like(S)
    days = S.index[:: a.horizon]
    out = []
    for d in days:
        sc, r = S.loc[d].dropna(), R.loc[d]
        top = sc.nlargest(a.n).index
        out.append(r[top].mean() - r[sc.index].mean())
    x = pd.Series(out, index=days).dropna()
    t = x.mean() / x.std() * np.sqrt(len(x))
    ann = (1 + x.mean()) ** (252 / a.horizon) - 1
    print(f"{Path(path).parent.name:26s} {col:10s} top{a.n} - universe: {x.mean() * 1e4:6.1f} bp/period "
          f"(~{ann:+.1%}/yr gross)  t={t:.2f}  hit={(x > 0).mean():.0%}")
