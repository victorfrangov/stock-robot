"""Walk-forward training + realistic daily portfolio simulation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from robot.config import Config
from robot.data import universe
from robot.data.prices import total_return_ohlc
from robot.features import feature_columns
from robot.models import Ensemble
from robot.portfolio import apply_trade_band, select_targets, smooth_scores

log = logging.getLogger(__name__)


def train_rows(panel: pd.DataFrame, cfg: Config, before: pd.Timestamp, all_dates: pd.DatetimeIndex) -> pd.Series:
    """Rows usable for training a model deployed at `before` (no label overlap)."""
    h = cfg.label.horizon
    pos = all_dates.searchsorted(before)
    cutoff = all_dates[max(pos - (h + 2), 0)]  # label at t needs open[t+h+1] < before
    step = cfg.model.train_sample_every
    day_idx = pd.Series(np.arange(len(all_dates)), index=all_dates)
    keep_days = all_dates[(day_idx % step == 0).to_numpy()]
    return (panel["date"] < cutoff) & panel["target"].notna() & panel["date"].isin(keep_days)


def make_ensemble(cfg: Config, use_nn: bool = True) -> Ensemble:
    weights = dict(cfg.model.ensemble)
    if not use_nn:
        weights["nn"] = 0
    return Ensemble(weights, dict(cfg.model.gbm), dict(cfg.model.nn))


def walk_forward(cfg: Config, panel: pd.DataFrame, use_nn: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Out-of-sample scores for every date from backtest.start on."""
    feats = feature_columns(panel, cfg.model.get("exclude_features"))
    all_dates = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    start = max(pd.Timestamp(cfg.backtest.start),
                all_dates[0] + pd.DateOffset(years=cfg.backtest.min_train_years))
    starts = pd.date_range(start, all_dates[-1], freq=f"{cfg.backtest.retrain_months}MS")
    starts = [all_dates[all_dates.searchsorted(d)] for d in starts if d <= all_dates[-1]]
    end = pd.Timestamp(cfg.backtest.get("end") or all_dates[-1] + pd.Timedelta(days=1))
    starts = sorted(d for d in set(starts) if d < end) + [min(end, all_dates[-1] + pd.Timedelta(days=1))]

    out, importance = [], None
    for i, (a, b) in enumerate(zip(starts[:-1], starts[1:]), 1):
        tr = train_rows(panel, cfg, a, all_dates)
        te = (panel["date"] >= a) & (panel["date"] < b)
        ens = make_ensemble(cfg, use_nn).fit(panel.loc[tr, feats], panel.loc[tr, "target"].to_numpy(),
                                             panel.loc[tr, "date"].to_numpy())
        seg = panel.loc[te, ["date", "ticker", "target", "fwd_ret"]].copy()
        X = panel.loc[te, feats]
        for name, p in ens.predict_members(X).items():
            seg[f"score_{name}"] = p
        seg["score"] = ens.predict(X, seg["date"].to_numpy())
        ic = seg.groupby("date").apply(lambda g: g["score"].corr(g["target"]), include_groups=False).mean()
        log.info("[%d/%d] %s -> %s  train=%d rows  IC=%.4f", i, len(starts) - 1, a.date(), b.date(), tr.sum(), ic)
        out.append(seg)
        if "gbm" in ens.members:
            importance = ens.members["gbm"].importance()
    return pd.concat(out, ignore_index=True), importance


@dataclass
class SimResult:
    equity: pd.Series
    returns: pd.Series
    turnover: pd.Series
    n_holdings: pd.Series
    trades: list = field(default_factory=list)
    weights: dict = field(default_factory=dict)


def simulate(cfg: Config, scores: pd.DataFrame, panel: pd.DataFrame, prices: pd.DataFrame,
             macro: pd.DataFrame, score_col: str = "score") -> SimResult:
    o = total_return_ohlc(prices)
    first = scores["date"].min()
    dates = o["close"].index[(o["close"].index >= first) & (o["close"].index <= scores["date"].max())]
    tickers = list(o["close"].columns)
    col = {t: j for j, t in enumerate(tickers)}
    O = o["open"].reindex(dates).to_numpy()
    C = o["close"].reindex(dates).to_numpy()
    prevC = o["close"].shift(1).reindex(dates).to_numpy()

    S = scores.pivot(index="date", columns="ticker", values=score_col).reindex(index=dates, columns=tickers)
    S = smooth_scores(S, cfg.portfolio.get("score_halflife", 0))
    mcap = (panel.pivot(index="date", columns="ticker", values="raw_mcap").reindex(index=dates, columns=tickers)
            if "raw_mcap" in panel else None)
    vol = panel.pivot(index="date", columns="ticker", values="raw_vol_63").reindex(index=dates, columns=tickers)
    price = panel.pivot(index="date", columns="ticker", values="raw_price").reindex(index=dates, columns=tickers)
    spy = o["close"][universe.BENCHMARK]
    risk_off = (spy < spy.rolling(200).mean()).reindex(dates).fillna(False).to_numpy()
    if not macro.empty and "DGS3MO" in macro:
        rf = (macro["DGS3MO"].reindex(dates.union(macro.index)).ffill().reindex(dates).fillna(0) / 100 / 252).to_numpy()
    else:
        rf = np.zeros(len(dates))

    cost_rate = (cfg.costs.commission_bps + cfg.costs.slippage_bps) / 1e4
    w = np.zeros(len(tickers))
    pending: np.ndarray | None = None
    V = 1.0
    eq, rets, tos, nh = [], [], [], []
    weights_log = {}
    last_rebalance = -10**9

    for i, t in enumerate(dates):
        v0 = V
        # overnight: previous close -> today's open
        on = np.where(w > 0, O[i] / prevC[i] - 1, 0.0)
        dead = (w > 0) & ~np.isfinite(on)  # no price today (delisted/halted): exit flat at last close
        on[~np.isfinite(on)] = 0.0
        forced = w[dead].sum()
        w[dead] = 0.0
        V *= 1 - forced * cost_rate  # still pay to get out
        cash = 1.0 - w.sum()
        g = 1.0 + (w * on).sum() + cash * rf[i]
        V *= g
        w = w * (1 + on) / g

        turnover = 0.0
        if pending is not None:
            tradable = np.isfinite(O[i])
            target = np.where(tradable, pending, w)  # can't trade names without a price today
            turnover = np.abs(target - w).sum()
            V *= 1 - turnover * cost_rate
            w = target
            pending = None

        # intraday: open -> close
        ir = np.where(w > 0, C[i] / O[i] - 1, 0.0)
        ir[~np.isfinite(ir)] = 0.0
        g = 1.0 + (w * ir).sum()
        V *= g
        w = w * (1 + ir) / g

        eq.append(V)
        rets.append(V / v0 - 1)
        tos.append(turnover)
        nh.append(int((w > 1e-6).sum()))

        # decision at the close for tomorrow's open
        if i - last_rebalance >= cfg.portfolio.rebalance_days and S.iloc[i].notna().sum() > cfg.portfolio.top_k:
            current = {tickers[j] for j in np.nonzero(w > 1e-6)[0]}
            tw = select_targets(cfg, S.iloc[i], current, vol.iloc[i], price.iloc[i], bool(risk_off[i]),
                                None if mcap is None else mcap.iloc[i])
            pending = np.zeros(len(tickers))
            for tk, wt in tw.items():
                pending[col[tk]] = wt
            pending = apply_trade_band(pending, w, cfg.portfolio.get("trade_band", 0.0))
            weights_log[t] = tw
            last_rebalance = i

    idx = pd.DatetimeIndex(dates)
    return SimResult(pd.Series(eq, idx), pd.Series(rets, idx), pd.Series(tos, idx), pd.Series(nh, idx),
                     weights=weights_log)


def metrics(returns: pd.Series, rf: pd.Series | None = None, turnover: pd.Series | None = None) -> dict:
    r = returns.dropna()
    if r.empty:
        return {}
    years = len(r) / 252
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    ex = r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)
    down = r[r < 0].std() * np.sqrt(252)
    out = {
        "cagr": eq.iloc[-1] ** (1 / years) - 1,
        "total_return": eq.iloc[-1] - 1,
        "vol": r.std() * np.sqrt(252),
        "sharpe": ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else np.nan,
        "sortino": ex.mean() * 252 / down if down > 0 else np.nan,
        "max_drawdown": dd.min(),
        "calmar": (eq.iloc[-1] ** (1 / years) - 1) / abs(dd.min()) if dd.min() < 0 else np.nan,
        "best_day": r.max(),
        "worst_day": r.min(),
        "pct_months_up": (r.resample("ME").apply(lambda x: (1 + x).prod() - 1) > 0).mean(),
        "years": years,
    }
    if turnover is not None:
        out["turnover_per_year"] = turnover.sum() / years
    return {k: float(v) for k, v in out.items()}


def ic_stats(scores: pd.DataFrame, col: str = "score") -> dict:
    s = scores.dropna(subset=[col, "target"])
    ic = s.groupby("date").apply(lambda g: g[col].corr(g["target"]), include_groups=False)
    dec = s.groupby("date").apply(
        lambda g: g.loc[g[col] >= g[col].quantile(0.9), "fwd_ret"].mean()
        - g.loc[g[col] <= g[col].quantile(0.1), "fwd_ret"].mean(), include_groups=False)
    return {"ic_mean": float(ic.mean()), "ic_ir": float(ic.mean() / ic.std() * np.sqrt(252)),
            "ic_pct_positive": float((ic > 0).mean()), "decile_spread_per_period": float(dec.mean()),
            "_ic_series": ic}


def run_backtest(cfg: Config, panel: pd.DataFrame, prices: pd.DataFrame, macro: pd.DataFrame,
                 use_nn: bool = True, out_dir: Path | None = None) -> dict:
    scores, importance = walk_forward(cfg, panel, use_nn)
    o = total_return_ohlc(prices)
    rf = None
    if not macro.empty and "DGS3MO" in macro:
        rf = macro["DGS3MO"].reindex(o["close"].index.union(macro.index)).ffill() / 100 / 252

    report: dict = {"config": {k: cfg[k] for k in ("label", "model", "portfolio", "costs", "backtest")}}
    sims = {}
    for col in [c for c in scores.columns if c.startswith("score")]:
        sim = simulate(cfg, scores, panel, prices, macro, score_col=col)
        sims[col] = sim
        stats = ic_stats(scores, col)
        stats.pop("_ic_series")
        report[col] = {**metrics(sim.returns, rf, sim.turnover), **stats,
                       "avg_holdings": float(sim.n_holdings[sim.n_holdings > 0].mean())}

    main = sims["score"]
    spy = o["close"][universe.BENCHMARK].pct_change(fill_method=None).reindex(main.returns.index)
    report["benchmark_spy"] = metrics(spy, rf)
    uni = scores.pivot(index="date", columns="ticker", values="score").notna()
    daily = o["close"].pct_change(fill_method=None).shift(-1).reindex(index=uni.index, columns=uni.columns)
    ew = daily.where(uni).mean(axis=1).shift(1).reindex(main.returns.index)
    report["benchmark_equal_weight"] = metrics(ew, rf)
    yearly = pd.DataFrame({
        "strategy": main.returns.groupby(main.returns.index.year).apply(lambda x: (1 + x).prod() - 1),
        "spy": spy.groupby(spy.index.year).apply(lambda x: (1 + x).prod() - 1),
    })
    report["yearly"] = {int(y): {k: float(v) for k, v in row.items()} for y, row in yearly.iterrows()}

    out_dir = out_dir or cfg.path("reports", f"backtest-{datetime.now():%Y%m%d-%H%M}", "x").parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"strategy": main.equity, "spy": (1 + spy.fillna(0)).cumprod(),
                  "equal_weight": (1 + ew.fillna(0)).cumprod(), "turnover": main.turnover,
                  "n_holdings": main.n_holdings}).to_csv(out_dir / "equity.csv")
    scores.to_parquet(out_dir / "scores.parquet", index=False)
    if importance is not None:
        importance.to_csv(out_dir / "importance.csv", header=["gain"])
        report["top_features"] = {k: float(v) for k, v in importance.head(20).items()}
    last = max(main.weights) if main.weights else None
    if last is not None:
        report["latest_portfolio"] = {"date": str(last.date()), "weights": main.weights[last].round(4).to_dict()}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    _plot(out_dir, main, spy, ew, ic_stats(scores)["_ic_series"])
    report["out_dir"] = str(out_dir)
    return report


def _plot(out_dir: Path, sim: SimResult, spy: pd.Series, ew: pd.Series, ic: pd.Series) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
    ax[0].plot(sim.equity.index, sim.equity, label="strategy", lw=1.6)
    ax[0].plot(spy.index, (1 + spy.fillna(0)).cumprod(), label="SPY", lw=1)
    ax[0].plot(ew.index, (1 + ew.fillna(0)).cumprod(), label="equal-weight universe", lw=1, alpha=0.7)
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[0].set_title("Walk-forward backtest (out-of-sample, after costs)")
    dd = sim.equity / sim.equity.cummax() - 1
    ax[1].fill_between(dd.index, dd, 0, color="tab:red", alpha=0.4)
    ax[1].set_ylabel("drawdown")
    ax[2].plot(ic.index, ic.rolling(63).mean(), color="tab:green")
    ax[2].axhline(0, color="grey", lw=0.5)
    ax[2].set_ylabel("rank IC (63d)")
    fig.tight_layout()
    fig.savefig(out_dir / "backtest.png", dpi=110)
    plt.close(fig)
