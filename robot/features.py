"""Feature engineering: wide (date x ticker) computations -> long training panel.

Stock-level features are converted to cross-sectional percentile ranks in
[-0.5, 0.5] each day, which makes them comparable across years and regimes.
Market-level features (prefixed ``mkt_``) are left raw so the model can learn
regime-dependent behaviour.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from robot.config import Config
from robot.data import fundamentals, universe
from robot.data.prices import total_return_ohlc, wide

log = logging.getLogger(__name__)

META = ["date", "ticker", "sector", "target", "fwd_ret", "raw_vol_63", "raw_price", "raw_adv_21"]


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def technical_features(o: dict[str, pd.DataFrame], spy: str = universe.BENCHMARK) -> dict[str, pd.DataFrame]:
    c, op, hi, lo = o["close"], o["open"], o["high"], o["low"]
    r = c.pct_change(fill_method=None)
    f: dict[str, pd.DataFrame] = {}
    for n in (1, 5, 21, 63, 126, 252):
        f[f"ret_{n}"] = c / c.shift(n) - 1
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    f["mom_6_1"] = c.shift(21) / c.shift(126) - 1
    f["vol_21"] = r.rolling(21).std()
    f["vol_63"] = r.rolling(63).std()
    f["vol_ratio"] = f["vol_21"] / f["vol_63"]
    f["downvol_63"] = np.sqrt((r.clip(upper=0) ** 2).rolling(63).mean())
    f["skew_63"] = r.rolling(63).skew()
    f["max_ret_21"] = r.rolling(21).max()
    prev = c.shift(1)
    tr = np.fmax(hi - lo, np.fmax((hi - prev).abs(), (lo - prev).abs()))
    f["atr_14"] = tr.rolling(14).mean() / c
    f["rsi_14"] = _rsi(c, 14)
    f["rsi_2"] = _rsi(c, 2)
    f["dist_high_252"] = c / c.rolling(252).max() - 1
    f["dist_low_252"] = c / c.rolling(252).min() - 1
    sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    f["px_sma20"] = c / sma20 - 1
    f["px_sma50"] = c / sma50 - 1
    f["px_sma200"] = c / sma200 - 1
    f["sma50_sma200"] = sma50 / sma200 - 1
    gap = op / prev - 1
    intraday = c / op - 1
    f["gap_1"] = gap
    f["intraday_1"] = intraday
    f["overnight_21"] = gap.rolling(21).mean()
    f["intraday_21"] = intraday.rolling(21).mean()
    dv = o["dollar_volume"]
    adv21 = dv.rolling(21).mean()
    f["log_adv_21"] = np.log1p(adv21)
    f["volume_ratio"] = o["volume"].rolling(5).mean() / o["volume"].rolling(63).mean()
    f["amihud_21"] = np.log1p((r.abs() / dv.replace(0, np.nan)).rolling(21).mean() * 1e9)
    if spy in c:
        m = r[spy]
        cov = r.mul(m, axis=0).rolling(252).mean() - r.rolling(252).mean().mul(m.rolling(252).mean(), axis=0)
        beta = cov.div(m.rolling(252).var(ddof=0), axis=0)
        f["beta_252"] = beta
        idio_var = r.rolling(63).var(ddof=0) - beta**2 * m.rolling(63).var(ddof=0).to_numpy()[:, None]
        f["idio_vol_63"] = np.sqrt(idio_var.clip(lower=0))
        f["resid_ret_21"] = f["ret_21"] - beta * f["ret_21"][spy].to_numpy()[:, None]
    return f


def _split_factor_after(splits: pd.Series, when: pd.Series) -> np.ndarray:
    """Product of split ratios strictly after each date in `when` (for share counts)."""
    s = splits[splits > 0]
    if s.empty:
        return np.ones(len(when))
    logs = np.log(s.to_numpy())
    after = np.concatenate([np.cumsum(logs[::-1])[::-1], [0.0]])  # after[i] = sum logs[i:]
    idx = np.searchsorted(s.index.to_numpy(dtype="datetime64[ns]"), when.to_numpy(dtype="datetime64[ns]"), side="right")
    out = np.exp(after[idx])
    out[pd.isna(when).to_numpy()] = np.nan
    return out


def fundamental_features(cfg: Config, prices: pd.DataFrame, o: dict[str, pd.DataFrame], tickers: list[str]) -> dict[str, pd.DataFrame]:
    snaps = fundamentals.load_snapshots(cfg, tickers)
    if not snaps:
        log.warning("no fundamentals available - run `robot data` with EDGAR access")
        return {}
    dates = o["close"].index
    splits = wide(prices, "splits").reindex(dates).fillna(0.0)
    names = ["ey", "sy", "bm", "fcfy", "roe", "roa", "gross_margin", "op_margin", "leverage",
             "accruals", "rev_growth", "ni_growth", "days_since_filing", "log_mcap", "cash_to_assets"]
    cols: dict[str, dict[str, pd.Series]] = {n: {} for n in names}
    day = pd.DataFrame({"date": dates.astype("datetime64[ns]")})
    for t, snap in snaps.items():
        if t not in o["close"]:
            continue
        s = snap.copy()
        s.index = s.index + pd.Timedelta(days=1)  # tradable the session after filing
        s = s.reset_index(names="avail").sort_values("avail")
        s["avail"] = s["avail"].astype("datetime64[ns]")
        d = pd.merge_asof(day, s, left_on="date", right_on="avail").set_index("date")
        d.index = dates  # merge_asof keeps the left order
        px = o["split_close"][t]
        shares = d.get("shares")
        if shares is None:
            continue
        factor = _split_factor_after(splits[t], d["shares_date"]) if "shares_date" in d else 1.0
        mcap = (px * shares * factor).where(lambda x: x > 0)
        get = lambda k: d[k] if k in d else pd.Series(np.nan, index=dates)  # noqa: E731
        equity, assets = get("equity"), get("assets")
        ni, rev = get("net_income_ttm"), get("revenue_ttm")
        vals = {
            "ey": ni / mcap,
            "sy": rev / mcap,
            "bm": equity / mcap,
            "fcfy": (get("cfo_ttm") - get("capex_ttm").fillna(0)) / mcap,
            "roe": ni / equity.where(equity > 0),
            "roa": ni / assets,
            "gross_margin": get("gross_profit_ttm") / rev,
            "op_margin": get("op_income_ttm") / rev,
            "leverage": get("liabilities") / assets,
            "accruals": (ni - get("cfo_ttm")) / assets,
            "rev_growth": get("revenue_growth"),
            "ni_growth": get("net_income_growth"),
            "days_since_filing": (dates.to_series() - pd.to_datetime(d["filing_date"])).dt.days,
            "log_mcap": np.log(mcap),
            "cash_to_assets": get("cash") / assets,
        }
        for n, v in vals.items():
            cols[n][t] = v.astype("float64")
    return {n: pd.DataFrame(c, index=dates).reindex(columns=tickers).replace([np.inf, -np.inf], np.nan)
            for n, c in cols.items()}


def market_features(cfg: Config, o: dict[str, pd.DataFrame], mask: pd.DataFrame, macro: pd.DataFrame,
                    f: dict[str, pd.DataFrame]) -> pd.DataFrame:
    c = o["close"]
    dates = c.index
    m = pd.DataFrame(index=dates)
    spy = c[universe.BENCHMARK]
    spy_r = spy.pct_change(fill_method=None)
    m["mkt_ret_5"] = spy / spy.shift(5) - 1
    m["mkt_ret_21"] = spy / spy.shift(21) - 1
    m["mkt_ret_63"] = spy / spy.shift(63) - 1
    m["mkt_vol_21"] = spy_r.rolling(21).std() * np.sqrt(252)
    m["mkt_trend"] = spy / spy.rolling(200).mean() - 1
    m["mkt_breadth"] = (f["px_sma200"] > 0).where(mask).mean(axis=1)
    m["mkt_dispersion"] = f["ret_21"].where(mask).std(axis=1)
    if not macro.empty:
        mac = macro.reindex(dates.union(macro.index)).ffill().shift(1).reindex(dates)  # published with a lag
        for col in mac:
            m[f"mkt_{col.lower()}"] = mac[col]
        if "VIXCLS" in mac:
            m["mkt_vix_chg_21"] = mac["VIXCLS"] - mac["VIXCLS"].shift(21)
    return m


def _xs_rank(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    return df.where(mask).rank(axis=1, pct=True) - 0.5


def build_panel(cfg: Config, prices: pd.DataFrame, macro: pd.DataFrame, *, labels: bool = True,
                since: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Long panel with one row per (date, in-universe ticker)."""
    o = total_return_ohlc(prices)
    dates = o["close"].index
    tickers = [t for t in o["close"].columns if t != universe.BENCHMARK]
    mask = universe.membership_mask(cfg, dates, list(o["close"].columns))
    mask[universe.BENCHMARK] = False
    mask &= o["close"].notna() & o["open"].notna()
    # Only names we can map to SEC filings. Otherwise "no fundamentals" would mostly flag
    # companies that later left the index (their tickers vanish from today's SEC map) - a leak.
    covered = fundamentals.tickers_with_filings(cfg)
    if covered:
        mask.loc[:, [t for t in mask.columns if t not in covered]] = False

    log.info("computing technical features (%d dates x %d tickers)", len(dates), len(tickers))
    f = technical_features(o)
    f.update(fundamental_features(cfg, prices, o, tickers))
    mkt = market_features(cfg, o, mask, macro, f)

    # sector-relative momentum
    sic = fundamentals.load_sic(cfg)
    sector_of = {t: fundamentals.sic_sector(sic.get(t, 0)) for t in o["close"].columns}
    sectors = sorted(set(sector_of.values()))
    sector_code = {s: i for i, s in enumerate(sectors)}
    for base in ("ret_21", "ret_63", "mom_12_1"):
        rel = pd.DataFrame(np.nan, index=dates, columns=o["close"].columns)
        masked = f[base].where(mask)
        for s in sectors:
            cols = [t for t in o["close"].columns if sector_of[t] == s]
            rel[cols] = masked[cols].sub(masked[cols].median(axis=1), axis=0)
        f[f"{base}_vs_sector"] = rel

    raw = {
        "raw_vol_63": f["vol_63"],
        "raw_price": o["raw_close"],
        "raw_adv_21": o["dollar_volume"].rolling(21).mean(),
    }
    ranked = {k: _xs_rank(v, mask) for k, v in f.items()}

    if labels:
        h = cfg.label.horizon
        fwd = o["open"].shift(-(h + 1)) / o["open"].shift(-1) - 1  # decide at close t, trade at open t+1
        raw["fwd_ret"] = fwd
        raw["target"] = _xs_rank(fwd, mask)

    rows_mask = mask.copy()
    if since is not None:
        rows_mask.loc[rows_mask.index < pd.Timestamp(since)] = False
    di, ti = np.nonzero(rows_mask.to_numpy())
    cols = rows_mask.columns
    panel = pd.DataFrame({"date": dates[di], "ticker": cols[ti]})
    panel["sector"] = np.array([sector_code[sector_of[t]] for t in cols])[ti].astype("int16")
    for name, frame in {**ranked, **raw}.items():
        panel[name] = frame.reindex(columns=cols).to_numpy(dtype="float32")[di, ti]
    mk = mkt.to_numpy(dtype="float32")[di]
    for j, name in enumerate(mkt.columns):
        panel[name] = mk[:, j]
    log.info("panel: %d rows x %d cols", *panel.shape)
    return panel


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in META]
