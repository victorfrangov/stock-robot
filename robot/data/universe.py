"""S&P 500 universe with historical membership (reduces survivorship bias)."""

from __future__ import annotations

import io
import logging
import time

import numpy as np
import pandas as pd
import requests

from robot.config import Config

log = logging.getLogger(__name__)

MEMBERSHIP_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"

# Ticker changes: the membership file records them as one ticker ending and another
# starting the same day, but Yahoo only serves the history under the new symbol, so
# without this map the company vanishes from the universe for its whole earlier window
# (Meta was missing 2012-2021). Derived from same-day swaps cross-checked against SEC
# former-name changes (research/renames.py); the last three were verified by hand.
RENAMES: dict[str, str] = {
    "HRS": "LHX", "BLL": "BALL", "ABC": "COR", "NLOK": "GEN", "SYMC": "NLOK", "MMC": "MRSH",
    "FLT": "CPAY", "WLTW": "WTW", "FI": "FISV", "DWDP": "DD", "BK": "BNY", "JEC": "J",
    "DISCA": "WBD", "DISCK": "WBD", "PKI": "RVTY", "ARNC": "HWM", "CTL": "LUMN", "TMK": "GL",
    "BHGE": "BKR", "ANTM": "ELV", "UTX": "RTX", "RE": "EG", "FB": "META",
}


def resolve_rename(ticker: str) -> str:
    seen = set()
    while ticker in RENAMES and ticker not in seen:
        seen.add(ticker)
        ticker = RENAMES[ticker]
    return ticker
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BENCHMARK = "SPY"
_MAX_AGE_S = 7 * 86400


def yahoo_symbol(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def _current_wikipedia() -> pd.DataFrame:
    html = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    added = pd.to_datetime(table.get("Date added"), errors="coerce")
    return pd.DataFrame({"ticker": table["Symbol"].map(yahoo_symbol), "start_date": added})


def membership(cfg: Config, refresh: bool = False) -> pd.DataFrame:
    """Rows of (ticker, start_date, end_date); end_date NaT = still a member."""
    path = cfg.path("universe", "membership.parquet")
    if path.exists() and not refresh and time.time() - path.stat().st_mtime < _MAX_AGE_S:
        return pd.read_parquet(path)

    log.info("downloading historical S&P 500 membership")
    df = pd.read_csv(io.StringIO(requests.get(MEMBERSHIP_URL, timeout=30).text))
    df["ticker"] = df["ticker"].map(yahoo_symbol)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])

    # Top up with the live Wikipedia list in case the GitHub dataset lags recent additions.
    try:
        wiki = _current_wikipedia()
        current = set(df.loc[df["end_date"].isna(), "ticker"])
        missing = wiki[~wiki["ticker"].isin(current)].copy()
        if len(missing):
            log.info("adding %d current members missing from dataset: %s", len(missing), list(missing["ticker"]))
            missing["start_date"] = missing["start_date"].fillna(pd.Timestamp.today().normalize())
            missing["end_date"] = pd.NaT
            df = pd.concat([df, missing], ignore_index=True)
    except Exception as e:  # Wikipedia is a convenience, never a hard dependency
        log.warning("wikipedia check skipped: %s", e)

    df["ticker"] = df["ticker"].map(resolve_rename)
    df.to_parquet(path)
    return df


def tickers_since(cfg: Config, since: str | pd.Timestamp) -> list[str]:
    m = membership(cfg)
    since = pd.Timestamp(since)
    alive = m["end_date"].isna() | (m["end_date"] >= since)
    return sorted(m.loc[alive, "ticker"].unique())


def current_members(cfg: Config) -> list[str]:
    m = membership(cfg)
    return sorted(m.loc[m["end_date"].isna(), "ticker"].unique())


def membership_mask(cfg: Config, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """Boolean frame (dates x tickers): was the ticker in the index on that date."""
    m = membership(cfg)
    col = {t: i for i, t in enumerate(tickers)}
    values = np.zeros((len(dates), len(tickers)), dtype=bool)
    d = dates.to_numpy()
    for row in m.itertuples(index=False):
        j = col.get(row.ticker)
        if j is None:
            continue
        end = np.datetime64("2262-01-01") if pd.isna(row.end_date) else row.end_date.to_datetime64()
        values[(d >= row.start_date.to_datetime64()) & (d <= end), j] = True
    return pd.DataFrame(values, index=dates, columns=tickers)
