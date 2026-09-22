"""Macro series from FRED (public CSV endpoint, no API key)."""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from robot.config import Config

log = logging.getLogger(__name__)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def update_macro(cfg: Config) -> pd.DataFrame:
    series = []
    for sid in cfg.data.fred_series:
        try:
            text = requests.get(FRED_CSV.format(sid), timeout=60).text
            s = pd.read_csv(io.StringIO(text), index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
            series.append(s.rename(sid).astype(float))
        except Exception as e:
            log.warning("FRED %s failed: %s", sid, e)
    df = pd.concat(series, axis=1).sort_index()
    df.index.name = "date"
    df.to_parquet(cfg.path("macro", "fred.parquet"))
    log.info("macro: %s through %s", list(df.columns), df.index.max().date())
    return df


def load_macro(cfg: Config) -> pd.DataFrame:
    path = cfg.path("macro", "fred.parquet")
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()
