"""Glue: cached feature panel and production model training."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from robot.backtest import make_ensemble
from robot.config import Config
from robot.data.macro import load_macro
from robot.data.prices import load_prices, prices_path
from robot.features import build_panel, feature_columns
from robot.models import Ensemble

log = logging.getLogger(__name__)


def get_panel(cfg: Config, rebuild: bool = False) -> pd.DataFrame:
    """Full-history training panel, cached until prices or fundamentals change."""
    key = f"h{cfg.label.horizon}_{cfg.label.get('target', 'rank')}_im{int(cfg.model.get('industry_momentum', True))}" \
          f"_ea{int(cfg.model.get('earnings_features', True))}"
    path = cfg.path("features", f"panel_{key}.parquet")
    facts = cfg.root / "fundamentals" / "facts"
    newest_input = max([prices_path(cfg).stat().st_mtime,
                        *(p.stat().st_mtime for p in facts.glob("*.parquet"))] if facts.exists()
                       else [prices_path(cfg).stat().st_mtime])
    if path.exists() and not rebuild and path.stat().st_mtime > newest_input:
        log.info("loading cached panel %s", path)
        return pd.read_parquet(path)
    panel = build_panel(cfg, load_prices(cfg), load_macro(cfg))
    panel.to_parquet(path, index=False)
    return panel


def models_dir(cfg: Config) -> Path:
    return cfg.root / "models"


def latest_model_path(cfg: Config) -> Path:
    return models_dir(cfg) / "latest"


def train_production(cfg: Config, use_nn: bool = True, panel: pd.DataFrame | None = None) -> Path:
    panel = panel if panel is not None else get_panel(cfg)
    feats = feature_columns(panel, cfg.model.get("exclude_features"))
    all_dates = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    step = cfg.model.train_sample_every
    keep_days = all_dates[::-1][::step]  # always include the most recent labelled days
    rows = panel["target"].notna() & panel["date"].isin(keep_days)
    ens = make_ensemble(cfg, use_nn).fit(panel.loc[rows, feats], panel.loc[rows, "target"].to_numpy(),
                                         panel.loc[rows, "date"].to_numpy())
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out = models_dir(cfg) / stamp
    ens.save(out, meta={
        "trained_at": stamp,
        "train_start": str(panel.loc[rows, "date"].min().date()),
        "train_end": str(panel.loc[rows, "date"].max().date()),
        "rows": int(rows.sum()),
        "horizon": cfg.label.horizon,
    })
    latest = latest_model_path(cfg)
    if latest.is_symlink() or latest.exists():
        latest.unlink() if latest.is_symlink() else shutil.rmtree(latest)
    latest.symlink_to(out.name)
    log.info("model saved to %s (latest -> %s)", out, out.name)
    return out


def load_model(cfg: Config) -> Ensemble:
    path = latest_model_path(cfg)
    if not path.exists():
        raise FileNotFoundError("no trained model - run `robot train` first")
    return Ensemble.load(path.resolve())


def score_latest(cfg: Config, model: Ensemble | None = None,
                 cutoff: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Scores for the most recent completed session using only recent history (fast).

    When `portfolio.score_halflife` is set, the last ~6 half-lives of sessions are scored
    and smoothed exactly like the backtest does; `score` is the smoothed value.
    """
    from robot.calendar import last_completed_session
    from robot.portfolio import smooth_scores

    model = model or load_model(cfg)
    prices = load_prices(cfg)
    prices = prices[prices["date"] <= (cutoff or last_completed_session())]
    last = prices["date"].max()
    halflife = cfg.portfolio.get("score_halflife", 0)
    sessions = np.sort(prices.loc[prices["ticker"] == "SPY", "date"].unique())
    first = pd.Timestamp(sessions[-min(len(sessions), int(6 * halflife) + 1)]) if halflife else last
    recent = prices[prices["date"] >= first - pd.Timedelta(days=500)]
    panel = build_panel(cfg, recent, load_macro(cfg), labels=False, since=first)
    missing = [f for f in model.features if f not in panel]
    for f in missing:
        panel[f] = np.nan
    if missing:
        log.warning("features missing at inference (filled NaN): %s", missing)
    panel["raw_score"] = model.predict(panel[model.features], panel["date"].to_numpy())
    wide = panel.pivot(index="date", columns="ticker", values="raw_score")
    smoothed = smooth_scores(wide, halflife).iloc[-1]
    panel = panel[panel["date"] == last].copy()
    panel["score"] = panel["ticker"].map(smoothed)
    panel = panel.sort_values("score", ascending=False).reset_index(drop=True)
    panel["rank"] = np.arange(1, len(panel) + 1)
    return last, panel
