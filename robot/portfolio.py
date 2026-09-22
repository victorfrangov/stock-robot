"""Turn model scores into target portfolio weights (shared by backtest and live).

Two construction modes:

* ``topk``  - concentrated: hold the best `top_k` names (hysteresis via `hold_buffer`).
* ``tilt``  - enhanced index: start from market-cap weights over the universe, drop the
  worst-ranked names and scale the rest by exp(tilt_strength * rank). Tracks the S&P
  closely and harvests the signal as a steady active return.
"""

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


def smooth_scores(scores: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Causal EMA of each ticker's daily score (dates x tickers). Cuts turnover from noise.

    Scores are per-day ranks in [-0.5, 0.5]; a name that leaves the universe resets.
    """
    if not halflife:
        return scores
    return scores.ewm(halflife=halflife, min_periods=1, ignore_na=True).mean().where(scores.notna())


def _topk(cfg: Config, s: pd.Series, current: set[str], vol: pd.Series, mcap: pd.Series | None) -> pd.Series:
    p = cfg.portfolio
    ranked = s.sort_values(ascending=False)
    rank = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
    keep = [t for t in ranked.index if t in current and rank[t] <= p.hold_buffer][: p.top_k]
    picks = keep + [t for t in ranked.index if t not in keep][: p.top_k - len(keep)]
    if not picks:
        return pd.Series(dtype=float)
    if p.weighting == "inverse_vol":
        v = vol.reindex(picks)
        raw = 1.0 / v.fillna(v.median()).clip(lower=0.005)
    elif p.weighting == "score":
        raw = (s.reindex(picks) + 0.5).clip(lower=0.01)
    elif p.weighting == "cap" and mcap is not None:
        m = mcap.reindex(picks)
        raw = m.fillna(m.median())
    else:
        raw = pd.Series(1.0, index=picks)
    return raw


def _tilt(cfg: Config, s: pd.Series, mcap: pd.Series | None) -> pd.Series:
    p = cfg.portfolio
    if mcap is None:
        raise ValueError("tilt mode needs market caps (raw_mcap)")
    m = mcap.reindex(s.index)
    m = m.fillna(m.median())
    pct = s.rank(pct=True)
    keep = pct > p.tilt_exclude
    raw = m[keep] ** p.tilt_cap_power * np.exp(p.tilt_strength * (pct[keep] - 0.5))
    return raw


def select_targets(cfg: Config, scores: pd.Series, current: set[str], vol: pd.Series, price: pd.Series,
                   regime_risk_off: bool, mcap: pd.Series | None = None) -> pd.Series:
    """Target weights (ticker -> fraction of equity) for one rebalance."""
    p = cfg.portfolio
    s = scores.dropna()
    s = s[price.reindex(s.index).fillna(0) >= p.min_price]
    if s.empty:
        return pd.Series(dtype=float)
    raw = _tilt(cfg, s, mcap) if p.get("mode", "topk") == "tilt" else _topk(cfg, s, current, vol, mcap)
    return capped_weights(raw, exposure(cfg, regime_risk_off), p.max_weight)


def apply_trade_band(target: np.ndarray, current: np.ndarray, band: float) -> np.ndarray:
    """Leave continuing positions alone when the change is smaller than `band` (fraction of equity)."""
    if not band:
        return target
    small = (target > 0) & (current > 0) & (np.abs(target - current) < band)
    return np.where(small, current, target)
