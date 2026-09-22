"""Rank IC of walk-forward scores, restricted to the DEV period (holdout stays sealed)."""

import sys

import pandas as pd

DEV_END = "2020-01-01"

for path in sys.argv[1:]:
    s = pd.read_parquet(path)
    s = s[s["date"] < DEV_END]
    for col in [c for c in s.columns if c.startswith("score")]:
        ic = s.groupby("date").apply(lambda g: g[col].corr(g["target"]), include_groups=False)
        yearly = ic.groupby(ic.index.year).mean()
        print(f"{path.split('/')[-2]:28s} {col:12s} IC={ic.mean():.4f}  IR={ic.mean() / ic.std() * 252 ** 0.5:.2f}  "
              f"pos={(ic > 0).mean():.2f}  yearly=" + " ".join(f"{v:+.3f}" for v in yearly))
