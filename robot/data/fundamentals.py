"""Point-in-time fundamentals from SEC EDGAR XBRL "companyfacts" (free).

Every value is keyed by the date it was *filed*, so the model only ever sees
numbers that were public at the time - no look-ahead bias. Flow items
(revenue, income, cash flow) are converted to trailing-twelve-months using the
standard "last annual + current YTD - prior YTD" identity.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from robot.config import Config

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{:010d}.json"

FLOW = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "op_income": ["OperatingIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}
INSTANT = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "lt_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
}


class _RateLimiter:
    """SEC allows 10 requests/second; stay under it across threads."""

    def __init__(self, per_second: float = 8.0):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.next - now
            self.next = max(now, self.next) + self.interval
        if delay > 0:
            time.sleep(delay)


class EdgarClient:
    def __init__(self, user_agent: str):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.limiter = _RateLimiter()

    def get_json(self, url: str) -> dict | None:
        for attempt in range(4):
            self.limiter.wait()
            try:
                r = self.session.get(url, timeout=60)
            except requests.RequestException as e:  # timeouts/resets: back off and retry
                log.debug("EDGAR %s: %s", url, e)
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                return None
            if r.status_code in (403, 429, 503):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        log.warning("EDGAR gave up on %s", url)
        return None


def ticker_cik_map(cfg: Config, client: EdgarClient) -> dict[str, int]:
    path = cfg.path("fundamentals", "tickers.parquet")
    if path.exists() and time.time() - path.stat().st_mtime < 7 * 86400:
        df = pd.read_parquet(path)
    else:
        raw = client.get_json(TICKERS_URL) or {}
        df = pd.DataFrame(raw.values()).rename(columns={"cik_str": "cik"})
        df["ticker"] = df["ticker"].str.upper().str.replace(".", "-", regex=False)
        df.to_parquet(path)
    return dict(zip(df["ticker"], df["cik"].astype(int)))


def _records(facts: dict, taxonomy: str, tags: list[str], unit: str) -> pd.DataFrame:
    rows = []
    for priority, tag in enumerate(tags):
        node = facts.get("facts", {}).get(taxonomy, {}).get(tag)
        if not node:
            continue
        for u, items in node.get("units", {}).items():
            if u != unit:
                continue
            for it in items:
                if it.get("form", "").startswith(("10-K", "10-Q", "20-F", "40-F")) or it.get("form") in ("8-K",):
                    rows.append((priority, it.get("start"), it["end"], it["val"], it["filed"]))
    if not rows:
        return pd.DataFrame(columns=["start", "end", "val", "filed"])
    df = pd.DataFrame(rows, columns=["priority", "start", "end", "val", "filed"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce").astype("datetime64[ns]")
    df = df.dropna(subset=["end", "filed"])
    # One value per period: the earliest filing (what the market saw first), best tag first.
    df = df.sort_values(["start", "end", "filed", "priority"])
    first = df.groupby(["start", "end"], dropna=False, sort=False).first().reset_index()
    return first[["start", "end", "val", "filed"]]


def extract_company(facts: dict) -> pd.DataFrame:
    """Long frame (item, start, end, val, filed) for one company."""
    out = []
    for item, tags in {**FLOW, **INSTANT}.items():
        df = _records(facts, "us-gaap", tags, "USD")
        df.insert(0, "item", item)
        out.append(df)
    shares = _records(facts, "dei", ["EntityCommonStockSharesOutstanding"], "shares")
    if shares.empty:
        shares = _records(facts, "us-gaap", ["CommonStockSharesOutstanding"], "shares")
    shares.insert(0, "item", "shares")
    out.append(shares)
    return pd.concat(out, ignore_index=True)


def update_fundamentals(cfg: Config, tickers: list[str], max_age_days: float = 6.0) -> None:
    client = EdgarClient(cfg.data.sec_user_agent)
    cik_of = ticker_cik_map(cfg, client)
    todo = []
    for t in tickers:
        cik = cik_of.get(t)
        if cik is None:
            continue
        path = cfg.path("fundamentals", "facts", f"{cik}.parquet")
        if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
            continue
        todo.append((t, cik, path))
    log.info("EDGAR: refreshing %d companies (%d tickers without a CIK)", len(todo),
             sum(t not in cik_of for t in tickers))

    sic_path = cfg.path("fundamentals", "sic.parquet")
    sic = pd.read_parquet(sic_path) if sic_path.exists() else pd.DataFrame(columns=["cik", "sic"])
    known_sic = set(sic["cik"].astype(int))
    new_sic: list[tuple[int, int]] = []

    def work(job):
        t, cik, path = job
        facts = client.get_json(FACTS_URL.format(cik))
        if facts:
            extract_company(facts).to_parquet(path, index=False)
        if cik not in known_sic:
            sub = client.get_json(SUBMISSIONS_URL.format(cik)) or {}
            try:
                new_sic.append((cik, int(sub.get("sic") or 0)))
            except ValueError:
                pass

    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(work, todo), 1):
            if i % 50 == 0:
                log.info("EDGAR: %d/%d", i, len(todo))

    if new_sic:
        sic = pd.concat([sic, pd.DataFrame(new_sic, columns=["cik", "sic"])]).drop_duplicates("cik", keep="last")
        sic.to_parquet(sic_path, index=False)


# ---------------------------------------------------------------- point-in-time


def _ttm(flow: pd.DataFrame) -> pd.DataFrame:
    """Trailing-twelve-month values: columns end, val, avail."""
    f = flow.dropna(subset=["start"]).copy()
    if f.empty:
        return pd.DataFrame(columns=["end", "val", "avail"])
    f["dur"] = (f["end"] - f["start"]).dt.days
    annual = f[f["dur"].between(350, 380)]
    ytd = f[f["dur"].between(80, 290)]
    quarters = f[f["dur"].between(80, 100)].sort_values("end")
    out = [pd.DataFrame({"end": annual["end"], "val": annual["val"], "avail": annual["filed"]})]

    tol = pd.Timedelta(days=12)
    ann_end = annual["end"].to_numpy()
    rows = []
    for r in ytd.itertuples(index=False):
        # prior fiscal year ending the day before this YTD period started
        a = annual[np.abs(ann_end - (r.start - pd.Timedelta(days=1)).to_datetime64()) <= tol.to_timedelta64()]
        prior = ytd[
            ((ytd["end"] - (r.end - pd.Timedelta(days=365))).abs() <= tol)
            & ((ytd["dur"] - r.dur).abs() <= 15)
        ]
        if len(a) and len(prior):
            val = a["val"].iloc[0] + r.val - prior["val"].iloc[0]
            avail = max(r.filed, a["filed"].iloc[0], prior["filed"].iloc[0])
            rows.append((r.end, val, avail))
            continue
        # fallback: four discrete quarters ending at r.end
        q = quarters[(quarters["end"] <= r.end) & (quarters["end"] > r.end - pd.Timedelta(days=370))]
        if r.dur <= 100 and len(q) == 4:
            rows.append((r.end, q["val"].sum(), q["filed"].max()))
    if rows:
        out.append(pd.DataFrame(rows, columns=["end", "val", "avail"]))
    ttm = pd.concat(out, ignore_index=True).sort_values(["end", "avail"])
    return ttm.drop_duplicates("end", keep="first")


def _drop_scale_errors(r: pd.DataFrame, tol: float = 20.0) -> pd.DataFrame:
    """Drop share counts that are >tol x away from the median of the previous three filings.

    Some filers report shares in thousands or with a wrong `decimals` attribute (ORCL 2012-09,
    YUM 2016-10, AJG 2020: 1e6 x too large). No real split is 20:1 in a single quarter... except
    a handful, which the market-cap guard in features catches instead.
    """
    r = r.sort_values(["end", "avail"]).reset_index(drop=True)
    vals = r["val"].to_numpy(dtype="float64")
    keep = np.ones(len(r), dtype=bool)
    good: list[float] = []
    for i, v in enumerate(vals):
        if v <= 0:
            keep[i] = False
            continue
        if len(good) >= 2:
            ref = float(np.median(good[-3:]))
            if v > ref * tol or v < ref / tol:
                keep[i] = False
                continue
        good.append(v)
    return r[keep]


def _latest_known(series: pd.DataFrame) -> pd.DataFrame:
    """Keep, at each availability date, only the newest period known so far."""
    s = series.sort_values(["avail", "end"])
    s = s[s["end"] >= s["end"].cummax()]
    return s.drop_duplicates("avail", keep="last")


def company_snapshots(records: pd.DataFrame) -> pd.DataFrame:
    """Wide point-in-time table indexed by availability date."""
    parts = []
    for item in FLOW:
        ttm = _ttm(records[records["item"] == item])
        if ttm.empty:
            continue
        # year-over-year growth of the TTM value
        prev = ttm[["end", "val"]].rename(columns={"end": "prev_end", "val": "prev_val"})
        ttm = ttm.sort_values("end")
        g = pd.merge_asof(
            ttm.assign(target=ttm["end"] - pd.Timedelta(days=365)).sort_values("target"),
            prev.sort_values("prev_end"), left_on="target", right_on="prev_end",
            direction="nearest", tolerance=pd.Timedelta(days=20),
        )
        g["growth"] = g["val"] / g["prev_val"].abs() - np.sign(g["prev_val"])
        known = _latest_known(g)
        parts.append(known.set_index("avail")["val"].rename(f"{item}_ttm"))
        if item in ("revenue", "net_income"):
            parts.append(known.set_index("avail")["growth"].rename(f"{item}_growth"))
    for item in [*INSTANT, "shares"]:
        r = records[records["item"] == item].rename(columns={"filed": "avail"})
        if item == "shares":
            r = _drop_scale_errors(r)
        if r.empty:
            continue
        known = _latest_known(r[["end", "val", "avail"]])
        parts.append(known.set_index("avail")["val"].rename(item))
        if item == "shares":
            # The cover-page count is as of a date just before filing, so it already reflects
            # any split between the period end and the filing. Splits are applied from here on.
            parts.append(known.set_index("avail").index.to_series().rename("shares_date"))
    if not parts:
        return pd.DataFrame()
    parts = [p[~p.index.duplicated(keep="last")] for p in parts]
    snap = pd.concat(parts, axis=1).sort_index().ffill()
    snap["filing_date"] = snap.index
    return snap


def load_snapshots(cfg: Config, tickers: list[str]) -> dict[str, pd.DataFrame]:
    tpath = cfg.path("fundamentals", "tickers.parquet")
    if not tpath.exists():
        return {}
    tmap = pd.read_parquet(tpath)
    cik_of = dict(zip(tmap["ticker"], tmap["cik"].astype(int)))
    out = {}
    for t in tickers:
        cik = cik_of.get(t)
        path = cfg.path("fundamentals", "facts", f"{cik}.parquet")
        if cik is None or not path.exists():
            continue
        snap = company_snapshots(pd.read_parquet(path))
        if not snap.empty:
            out[t] = snap
    return out


def tickers_with_filings(cfg: Config) -> set[str]:
    tpath = cfg.path("fundamentals", "tickers.parquet")
    if not tpath.exists():
        return set()
    tmap = pd.read_parquet(tpath)
    facts = cfg.root / "fundamentals" / "facts"
    have = {int(p.stem) for p in facts.glob("*.parquet")}
    return {t for t, c in zip(tmap["ticker"], tmap["cik"]) if int(c) in have}


def load_sic(cfg: Config) -> dict[str, int]:
    tpath, spath = cfg.path("fundamentals", "tickers.parquet"), cfg.path("fundamentals", "sic.parquet")
    if not (tpath.exists() and spath.exists()):
        return {}
    tmap = pd.read_parquet(tpath)
    sic = dict(zip(pd.read_parquet(spath)["cik"].astype(int), pd.read_parquet(spath)["sic"].astype(int)))
    return {t: sic.get(int(c), 0) for t, c in zip(tmap["ticker"], tmap["cik"])}


def sic_sector(sic: int) -> str:
    """Coarse GICS-like sector from a 4-digit SIC code."""
    if not sic:
        return "Unknown"
    s2 = sic // 100
    if s2 in (13, 29):
        return "Energy"
    if 2830 <= sic <= 2836 or 3841 <= sic <= 3851 or s2 == 80 or sic == 5122 or sic == 8731:
        return "HealthCare"
    if s2 in (10, 12, 14, 24, 26, 28, 30, 32, 33):
        return "Materials"
    if 3570 <= sic <= 3579 or s2 == 36 or s2 == 38 or 7370 <= sic <= 7379:
        return "Technology"
    if s2 == 48 or s2 in (78, 79) or sic in (2711, 2721, 7812):
        return "Communication"
    if s2 == 49:
        return "Utilities"
    if sic == 6798 or s2 == 65:
        return "RealEstate"
    if 60 <= s2 <= 67:
        return "Financials"
    if s2 in (20, 21, 54) or sic in (2080, 2086, 5411, 5912):
        return "ConsumerStaples"
    if s2 in (22, 23, 25, 31, 39, 52, 53, 55, 56, 57, 58, 59, 70, 72) or sic in (3711, 3714, 3751):
        return "ConsumerDiscretionary"
    return "Industrials"
