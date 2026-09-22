"""Point-in-time earnings-announcement dates from SEC EDGAR (free), plus daily features.

Source: the EDGAR submissions API. Every 8-K / 8-K/A tagged with item 2.02
("Results of Operations and Financial Condition") is treated as an earnings
release. Item tagging in the submissions index starts around Aug 2004, so
earlier releases are not visible.

Timing convention
-----------------
``acceptanceDateTime`` is a true UTC timestamp (AAPL's 16:30 ET releases show
as 20:30Z in summer, 21:30Z in winter). It is converted to US/Eastern and the
*reaction day* (``event_date``) is derived as:

* accepted before 09:30 ET on a weekday -> that same day's session;
* accepted at/after 09:30 ET, or on a weekend -> the next business day.

Intraday releases are pushed to the next day on purpose (conservative), and a
company that issues its press release pre-market but files the 8-K after the
open gets a reaction day one session late. Exchange holidays are ignored here
(plain business days); ``earnings_features`` maps each ``event_date`` onto the
first actual trading session on or after it, which absorbs holidays.

The model decides at the close of day t, so an event whose reaction day is t
is usable in features at the close of t (that close already reflects the news).
In other words the reaction day *is* the first tradable close date.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from robot.config import Config
from robot.data.fundamentals import SUBMISSIONS_URL, EdgarClient, ticker_cik_map

log = logging.getLogger(__name__)

PAGE_URL = "https://data.sec.gov/submissions/{}"
EARN_FORMS = ("8-K", "8-K/A")
CACHE_COLUMNS = ["accepted_utc", "event_date", "form"]
_OPEN = pd.Timedelta(hours=9, minutes=30)


# ---------------------------------------------------------------- download


def reaction_day(accepted_utc: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First session whose close follows the announcement (see module docstring)."""
    ts = pd.DatetimeIndex(pd.to_datetime(accepted_utc, utc=True))
    et = ts.tz_convert("America/New_York")
    day = et.normalize().tz_localize(None)
    same_day = (et.weekday < 5) & ((et - et.normalize()) < _OPEN)
    nxt = day + pd.offsets.BDay(1)
    return pd.DatetimeIndex(np.where(same_day, day, nxt)).astype("datetime64[ns]")


def _earn_rows(block: dict) -> list[tuple[str, str]]:
    forms = block.get("form", [])
    acc = block.get("acceptanceDateTime", [])
    items = block.get("items", [""] * len(forms))
    return [(a, f) for f, a, i in zip(forms, acc, items) if f in EARN_FORMS and "2.02" in (i or "")]


def parse_submissions(sub: dict, pages: list[dict]) -> pd.DataFrame:
    rows = _earn_rows(sub.get("filings", {}).get("recent", {}))
    for p in pages:
        rows += _earn_rows(p or {})
    if not rows:
        return pd.DataFrame({"accepted_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                             "event_date": pd.Series(dtype="datetime64[ns]"),
                             "form": pd.Series(dtype=object)})
    df = pd.DataFrame(rows, columns=["accepted_utc", "form"])
    df["accepted_utc"] = pd.to_datetime(df["accepted_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["accepted_utc"]).drop_duplicates().sort_values("accepted_utc")
    df["event_date"] = reaction_day(df["accepted_utc"])
    return df[CACHE_COLUMNS].reset_index(drop=True)


def update_earnings(cfg: Config, tickers: list[str], max_age_days: float = 6.0) -> None:
    client = EdgarClient(cfg.data.sec_user_agent)
    cik_of = ticker_cik_map(cfg, client)
    todo, seen = [], set()
    for t in tickers:
        cik = cik_of.get(t)
        if cik is None or cik in seen:
            continue
        seen.add(cik)
        path = cfg.path("earnings", f"{cik}.parquet")
        if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
            continue
        todo.append((cik, path))
    log.info("earnings: refreshing %d companies (%d tickers without a CIK)", len(todo),
             sum(t not in cik_of for t in tickers))

    def work(job):
        cik, path = job
        sub = client.get_json(SUBMISSIONS_URL.format(cik))
        if not sub:
            return
        pages = [client.get_json(PAGE_URL.format(f["name"])) for f in sub.get("filings", {}).get("files", [])]
        parse_submissions(sub, pages).to_parquet(path, index=False)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(work, todo), 1):
            if i % 50 == 0:
                log.info("earnings: %d/%d", i, len(todo))


def run_update(cfg: Config, max_age_days: float = 6.0) -> None:
    """Refresh earnings dates for every priced ticker that has a CIK."""
    from robot.data.prices import prices_path

    tmap = pd.read_parquet(cfg.path("fundamentals", "tickers.parquet"))
    priced = set(pd.read_parquet(prices_path(cfg), columns=["ticker"])["ticker"].unique())
    tickers = sorted(set(tmap["ticker"]) & priced)
    update_earnings(cfg, tickers, max_age_days=max_age_days)


# ---------------------------------------------------------------- events


def clean_events(df: pd.DataFrame, gap_days: int = 5) -> pd.DatetimeIndex:
    """Reaction days, sorted and deduplicated.

    8-K/As within ``gap_days`` calendar days of an original 8-K are dropped, and
    any remaining event within ``gap_days`` of the previously kept one is
    treated as part of the same announcement (the earliest wins, which keeps it
    point-in-time).
    """
    if df.empty:
        return pd.DatetimeIndex([])
    d = df.sort_values("event_date")
    orig = pd.DatetimeIndex(d.loc[d["form"] == "8-K", "event_date"]).unique().sort_values()
    amend = d["form"] != "8-K"
    if amend.any() and len(orig):
        ev = d["event_date"].to_numpy()
        pos = np.searchsorted(orig.to_numpy(), ev)
        near = np.full(len(d), np.inf)
        tol = np.timedelta64(1, "D")
        for shift in (0, 1):
            j = np.clip(pos - shift, 0, len(orig) - 1)
            near = np.minimum(near, np.abs(ev - orig.to_numpy()[j]) / tol)
        d = d[~(amend.to_numpy() & (near <= gap_days))]
    dates = pd.DatetimeIndex(d["event_date"]).unique().sort_values()
    keep, last = [], None
    for x in dates:
        if last is None or (x - last).days > gap_days:
            keep.append(x)
            last = x
    return pd.DatetimeIndex(keep)


def load_events(cfg: Config, tickers: list[str]) -> dict[str, pd.DatetimeIndex]:
    tpath = cfg.path("fundamentals", "tickers.parquet")
    if not tpath.exists():
        return {}
    tmap = pd.read_parquet(tpath)
    cik_of = dict(zip(tmap["ticker"], tmap["cik"].astype(int)))
    out, cache = {}, {}
    for t in tickers:
        cik = cik_of.get(t)
        if cik is None:
            continue
        if cik not in cache:
            path = cfg.root / "earnings" / f"{cik}.parquet"
            cache[cik] = clean_events(pd.read_parquet(path)) if path.exists() else None
        if cache[cik] is not None and len(cache[cik]):
            out[t] = cache[cik]
    return out


# ---------------------------------------------------------------- features

FEATURES = ["days_since_earn", "earn_ret_3", "earn_gap", "earn_vol_spike", "pre_earn_window"]


def _ticker_features(p: np.ndarray, n: int, c: np.ndarray, op: np.ndarray, v: np.ndarray,
                     sc: np.ndarray, so: np.ndarray, pre_days: int = 5, overdue: int = 10) -> dict[str, np.ndarray]:
    """Features for one ticker. p = sorted unique row positions of reaction days."""
    t = np.arange(n)
    k = np.searchsorted(p, t, side="right") - 1          # latest event known at close of t
    has = k >= 0
    kk = np.where(has, k, 0)
    last = p[kk]
    nan = np.full(n, np.nan)

    days = np.where(has, t - last, np.nan)

    prev = p - 1
    ok_prev = prev >= 0
    pv = np.where(ok_prev, prev, 0)
    end = np.minimum(p + 2, n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret3 = (c[end] / c[pv]) / (sc[end] / sc[pv]) - 1
        ret3[~ok_prev | (p + 2 >= n)] = np.nan
        gap = (op[p] / c[pv]) / (so[p] / sc[pv]) - 1
        gap[~ok_prev] = np.nan
        cv = np.concatenate([[0.0], np.cumsum(np.nan_to_num(v))])
        cn = np.concatenate([[0], np.cumsum(~np.isnan(v))])
        lo = np.maximum(p - 21, 0)
        cnt = cn[p] - cn[lo]
        mean_v = (cv[p] - cv[lo]) / cnt
        spike = np.where(cnt >= 10, v[p] / mean_v, np.nan)

    earn_ret_3 = np.where(has & (t >= last + 2), ret3[kk], nan)
    earn_gap = np.where(has, gap[kk], nan)
    vol_spike = np.where(has, spike[kk], nan)

    # expected next event: last + median of the previous up-to-4 gaps (past events only)
    gaps = np.diff(p).astype(float)
    exp_next = np.full(len(p), np.nan)
    if len(gaps):
        m = len(gaps)
        win = np.full((m, 4), np.nan)
        for j in range(4):
            win[j:, j] = gaps[: m - j]
        exp_next[1:] = p[1:] + np.nanmedian(win, axis=1)
    remaining = exp_next[kk] - t
    pre = np.where(has & (k >= 1), ((remaining <= pre_days) & (remaining >= -overdue)).astype(float), nan)
    return {"days_since_earn": days, "earn_ret_3": earn_ret_3, "earn_gap": earn_gap,
            "earn_vol_spike": vol_spike, "pre_earn_window": pre}


def compute_features(events: dict[str, pd.DatetimeIndex], o: dict[str, pd.DataFrame],
                     tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Wide earnings features from in-memory events (see ``earnings_features``)."""
    close = o["close"]
    idx = close.index
    cols = [t for t in tickers if t in close.columns]
    n = len(idx)
    out = {f: np.full((n, len(cols)), np.nan) for f in FEATURES}
    sc = close["SPY"].to_numpy(float) if "SPY" in close else np.ones(n)
    so = o["open"]["SPY"].to_numpy(float) if "SPY" in o["open"] else np.ones(n)
    dates = idx.to_numpy()
    for j, t in enumerate(cols):
        ev = events.get(t)
        if ev is None or not len(ev):
            continue
        ev = pd.DatetimeIndex(ev).to_numpy()
        ev = ev[(ev >= dates[0]) & (ev <= dates[-1])]
        if not len(ev):
            continue
        p = np.unique(np.searchsorted(dates, ev, side="left"))  # first session on/after event_date
        c = close[t].to_numpy(float)
        feats = _ticker_features(p, n, c, o["open"][t].to_numpy(float), o["volume"][t].to_numpy(float), sc, so)
        alive = ~np.isnan(c)
        for f, arr in feats.items():
            out[f][:, j] = np.where(alive, arr, np.nan)
    return {f: pd.DataFrame(a, index=idx, columns=cols) for f, a in out.items()}


def earnings_features(cfg: Config, o: dict[str, pd.DataFrame], tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Daily point-in-time earnings features, each a wide frame shaped like o["close"][tickers].

    Every value at row t uses only information public at the close of t:

    * days_since_earn - trading sessions since the latest reaction day (0 on it).
    * earn_ret_3 - SPY-relative total return over reaction day + 2 sessions,
      known from the close of the 3rd session, carried forward to the next event.
    * earn_gap - SPY-relative open / prior-close gap on the reaction day.
    * earn_vol_spike - reaction-day volume / mean volume of the prior 21 sessions.
    * pre_earn_window - 1 if the next event, projected as last event + median of
      the previous up-to-4 inter-event gaps, falls within 5 sessions (or is up
      to 10 sessions overdue), else 0; NaN with fewer than 2 past events.

    Values are NaN where the stock has no close.
    """
    return compute_features(load_events(cfg, tickers), o, tickers)
