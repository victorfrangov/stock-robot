"""Add control score columns to a walk-forward scores file.

score_rand0..4: random per-day scores (5 seeds)
score_size:     rank by point-in-time market cap (the "just buy the biggest" portfolio)
If a construction beats SPY with these, the edge is structural, not the model's.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robot.config import load_config  # noqa: E402
from robot.pipeline import get_panel  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
s = pd.read_parquet(src)
cfg = load_config()
cfg["model"]["industry_momentum"] = False
cfg["model"]["earnings_features"] = False
panel = get_panel(cfg)[["date", "ticker", "raw_mcap"]]
s = s.merge(panel, on=["date", "ticker"], how="left")
for k in range(5):
    rng = np.random.default_rng(k)
    s[f"score_rand{k}"] = rng.random(len(s))
s["score_size"] = s.groupby("date")["raw_mcap"].rank(pct=True)
s.drop(columns="raw_mcap").to_parquet(dst, index=False)
print("wrote", dst)
