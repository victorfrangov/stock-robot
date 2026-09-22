"""Paired test: is model B's DEV daily rank IC higher than model A's? (Newey-West-ish via 5-day blocks)"""

import sys

import numpy as np
import pandas as pd

DEV_END = "2020-01-01"


def daily_ic(path: str, col: str) -> pd.Series:
    s = pd.read_parquet(path)
    s = s[s["date"] < DEV_END]
    return s.groupby("date").apply(lambda g: g[col].corr(g["target"]), include_groups=False)


a = daily_ic(sys.argv[1], sys.argv[2])
b = daily_ic(sys.argv[3], sys.argv[4])
d = (b - a).dropna()
# overlapping 5-day labels -> average into non-overlapping 5-day blocks before the t-test
blocks = d.groupby(np.arange(len(d)) // 5).mean()
t = blocks.mean() / blocks.std() * np.sqrt(len(blocks))
print(f"mean IC A={a.mean():.4f} B={b.mean():.4f}  diff={d.mean():+.4f}  t={t:.2f}  "
      f"B better in {(blocks > 0).mean():.0%} of 5-day blocks")
