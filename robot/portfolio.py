"""Turn model scores into target portfolio weights (shared by backtest and live)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from robot.config import Config


def capped_weights(raw: pd.Series, total: float, cap: float) -> pd.Series:
    """Scale positive `raw` weights to sum to `total`, no weight above `cap`.

    Excess from capped names is redistributed; if everything is capped the
    remainder stays in cash.
    """
    w = raw.clip(lower=0).astype(float)
    if w.sum() <= 0 or total <= 0:
        return w * 0.0
    out = pd.Series(0.0, index=w.index)
    free = w.copy()
    remaining = total
    for _ in range(len(w)):
        scaled = free / free.sum() * remaining
        over = scaled > cap + 1e-12
        if not over.any():
            out[free.index] = scaled
            break
        out[scaled.index[over]] = cap
        remaining -= cap * over.sum()
        free = free[~over]
        if free.empty or remaining <= 1e-12:
            break
    return out


def exposure(cfg: Config, regime_risk_off: bool) -> float:
    p = cfg.portfolio
    e = 1.0 - p.cash_buffer
    if p.regime_filter and regime_risk_off:
        e *= p.regime_exposure
    return e


def select_targets(cfg: Config, scores: pd.Series, current: set[str], vol: pd.Series, price: pd.Series,
                   regime_risk_off: bool) -> pd.Series:
    """Target weights (ticker -> fraction of equity) for one rebalance.

    Keeps existing holdings while they rank inside `hold_buffer` (cuts turnover),
    then fills up to `top_k` with the best new names.
    """
    p = cfg.portfolio
    s = scores.dropna()
    s = s[price.reindex(s.index).fillna(0) >= p.min_price]
    ranked = s.sort_values(ascending=False)
    rank = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)

    keep = [t for t in ranked.index if t in current and rank[t] <= p.hold_buffer][: p.top_k]
    picks = keep + [t for t in ranked.index if t not in keep][: p.top_k - len(keep)]
    if not picks:
        return pd.Series(dtype=float)

    if p.weighting == "inverse_vol":
        v = vol.reindex(picks)
        v = v.fillna(v.median()).clip(lower=0.005)
        raw = 1.0 / v
    else:
        raw = pd.Series(1.0, index=picks)
    return capped_weights(raw, exposure(cfg, regime_risk_off), p.max_weight)
