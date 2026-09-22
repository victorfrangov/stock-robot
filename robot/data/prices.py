"""Daily OHLCV from Yahoo Finance, cached in one long parquet file.

Columns: date, ticker, open, high, low, close (split-adjusted), adj_close
(split+dividend adjusted), volume (split-adjusted), splits.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from robot.config import Config

log = logging.getLogger(__name__)

_FIELDS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Stock Splits": "splits",
}
COLUMNS = ["date", "ticker", *_FIELDS.values()]
_CHUNK = 100


def _download(tickers: list[str], start: str | pd.Timestamp) -> pd.DataFrame:
    frames = []
    for i in range(0, len(tickers), _CHUNK):
        chunk = tickers[i : i + _CHUNK]
        log.info("yahoo: %d-%d of %d tickers from %s", i + 1, i + len(chunk), len(tickers), pd.Timestamp(start).date())
        raw = yf.download(
            chunk, start=pd.Timestamp(start).strftime("%Y-%m-%d"), auto_adjust=False, actions=True,
            group_by="ticker", threads=True, progress=False,
        )
        if raw.empty:
            continue
        if not isinstance(raw.columns, pd.MultiIndex):  # single ticker
            raw.columns = pd.MultiIndex.from_product([chunk, raw.columns])
        long = raw.stack(level=0, future_stack=True).reset_index()
        long.columns = ["date", "ticker", *long.columns[2:]]
        long = long.rename(columns=_FIELDS)
        for c in _FIELDS.values():
            if c not in long:
                long[c] = 0.0 if c == "splits" else np.nan
        long = long.dropna(subset=["close", "adj_close"])
        frames.append(long[COLUMNS])
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    # A download during the session returns today's partial bar; never keep it.
    from robot.calendar import last_completed_session

    df = df[df["date"] <= last_completed_session()]
    df["splits"] = df["splits"].fillna(0.0)
    for c in ["open", "high", "low", "close", "adj_close", "volume", "splits"]:
        df[c] = df[c].astype("float64")
    return df


def validate_feed(df: pd.DataFrame, windows: pd.DataFrame | None = None, since: str = "2011-01-01") -> pd.DataFrame:
    """Drop tickers whose Yahoo series is not a real S&P 500 stock.

    A recycled symbol can point at some other instrument (PARA became a micro-cap
    with closes near 100,000 and a third of its days flat), and a few dead names come
    back as stale stubs. No index member trades with zero volume on 5% of days.
    Only rows inside the ticker's index membership `windows` (ticker, start_date,
    end_date) are judged: Yahoo often has thin pre-listing history for real members.
    """
    recent = df[df["date"] >= since]
    if windows is not None:
        w = recent.merge(windows[["ticker", "start_date", "end_date"]], on="ticker")
        recent = w[(w["date"] >= w["start_date"]) & (w["end_date"].isna() | (w["date"] <= w["end_date"]))]
    if recent.empty:
        return df
    g = recent.groupby("ticker")
    zero_vol = g["volume"].apply(lambda v: (v <= 0).mean())
    flat = g.apply(lambda x: ((x["open"] == x["high"]) & (x["high"] == x["low"]) & (x["low"] == x["close"])).mean(),
                   include_groups=False)
    med_dv = g.apply(lambda x: (x["close"] * x["volume"]).median(), include_groups=False)
    bad = sorted(set(zero_vol[zero_vol > 0.05].index) | set(flat[flat > 0.05].index) | set(med_dv[med_dv < 1e6].index))
    if bad:
        log.warning("dropping %d tickers with a junk price feed: %s", len(bad), bad)
        df = df[~df["ticker"].isin(bad)]
    return df


def prices_path(cfg: Config):
    return cfg.path("prices", "daily.parquet")


def load_prices(cfg: Config) -> pd.DataFrame:
    path = prices_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `robot data` first")
    return pd.read_parquet(path)


def update_prices(cfg: Config, tickers: list[str], full: bool = False) -> pd.DataFrame:
    """Download or incrementally refresh prices for `tickers`.

    Incremental mode refetches the last ~15 sessions for every ticker. If a
    ticker's overlapping history no longer matches (a split or dividend changed
    the adjustments), its whole history is downloaded again.
    """
    path = prices_path(cfg)
    start = cfg.data.start_date
    from robot.data.universe import membership

    windows = membership(cfg)
    if full or not path.exists():
        df = validate_feed(_download(tickers, start), windows)
        df.to_parquet(path, index=False)
        return df

    old = pd.read_parquet(path)
    known = set(old["ticker"].unique())
    new_tickers = [t for t in tickers if t not in known]
    refresh = sorted(set(tickers) & known)
    since = old["date"].max() - pd.Timedelta(days=21)
    recent = _download(refresh, since) if refresh else pd.DataFrame(columns=COLUMNS)

    # Detect adjustment changes on the overlap.
    merged = recent.merge(old[old["date"] >= since], on=["date", "ticker"], suffixes=("", "_old"))
    drift = (merged["adj_close"] / merged["adj_close_old"] - 1).abs()
    stale = sorted(set(merged.loc[drift > 1e-3, "ticker"]))
    if stale:
        log.info("adjustments changed for %d tickers, full reload: %s", len(stale), stale[:20])
    reload = new_tickers + stale
    full_hist = _download(reload, start) if reload else pd.DataFrame(columns=COLUMNS)

    keep = old[~old["ticker"].isin(reload)]
    keep = keep[~(keep["ticker"].isin(recent["ticker"].unique()) & (keep["date"] >= since))]
    recent = recent[~recent["ticker"].isin(reload)]
    df = pd.concat([keep, recent, full_hist], ignore_index=True)
    df = df.drop_duplicates(["date", "ticker"], keep="last").sort_values(["ticker", "date"])
    df = validate_feed(df, windows)
    df.to_parquet(path, index=False)
    log.info("prices: %d rows, %d tickers, last date %s", len(df), df["ticker"].nunique(), df["date"].max().date())
    return df


def wide(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    return prices.pivot(index="date", columns="ticker", values=field).sort_index()


def total_return_ohlc(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Wide dividend-and-split adjusted OHLC plus raw dollar volume."""
    close = wide(prices, "close")
    adj = wide(prices, "adj_close")
    factor = adj / close
    out = {
        "open": wide(prices, "open") * factor,
        "high": wide(prices, "high") * factor,
        "low": wide(prices, "low") * factor,
        "close": adj,
        "split_close": close,
        "volume": wide(prices, "volume"),
    }
    out["dollar_volume"] = close * out["volume"]
    # Unadjusted price as traded that day: undo the splits that happened later.
    ratio = wide(prices, "splits").reindex_like(close).fillna(0.0).replace(0.0, 1.0)
    after = ratio[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    out["raw_close"] = close * after
    return out
