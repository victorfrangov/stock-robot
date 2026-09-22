import numpy as np
import pandas as pd
import pytest

from robot.data.earnings import clean_events, compute_features, parse_submissions, reaction_day


def test_reaction_day_before_open_vs_after_close():
    acc = pd.Series([
        "2024-08-01T20:30:26.000Z",  # Thu 16:30 EDT -> Fri
        "2024-05-02T11:00:00.000Z",  # Thu 07:00 EDT -> same day
        "2024-05-02T13:30:00.000Z",  # Thu 09:30 EDT exactly -> next day
        "2024-02-01T14:29:00.000Z",  # Thu 09:29 EST (winter) -> same day
        "2024-08-02T21:00:00.000Z",  # Fri after close -> Mon
        "2024-08-03T12:00:00.000Z",  # Saturday -> Mon
    ])
    got = reaction_day(acc)
    want = pd.DatetimeIndex(["2024-08-02", "2024-05-02", "2024-05-03", "2024-02-01", "2024-08-05", "2024-08-05"])
    assert list(got) == list(want)


def test_parse_submissions_filters_item_202():
    sub = {"filings": {"recent": {
        "form": ["8-K", "8-K", "10-Q", "8-K/A"],
        "acceptanceDateTime": ["2024-08-01T20:30:26.000Z", "2024-07-01T20:30:00.000Z",
                               "2024-08-02T20:30:00.000Z", "2024-08-05T20:30:00.000Z"],
        "items": ["2.02,9.01", "5.02", "", "2.02"],
    }}}
    page = {"form": ["8-K"], "acceptanceDateTime": ["2010-01-25T21:35:00.000Z"], "items": ["2.02"]}
    df = parse_submissions(sub, [page])
    assert list(df["event_date"]) == [pd.Timestamp("2010-01-26"), pd.Timestamp("2024-08-02"), pd.Timestamp("2024-08-06")]
    # amendment within 5 days of the original collapses
    assert list(clean_events(df)) == [pd.Timestamp("2010-01-26"), pd.Timestamp("2024-08-02")]


def _market(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    r = rng.normal(0, 0.01, (n, 2))
    close = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=["AAA", "SPY"])
    opn = close.shift(1) * (1 + rng.normal(0, 0.005, (n, 2)))
    opn.iloc[0] = close.iloc[0]
    vol = pd.DataFrame(1e6, index=idx, columns=close.columns)
    return {"close": close, "open": opn, "volume": vol}


def test_earn_ret_3_timing_and_carry_forward():
    o = _market()
    idx = o["close"].index
    e = 30
    f = compute_features({"AAA": pd.DatetimeIndex([idx[e]])}, o, ["AAA"])
    r3 = f["earn_ret_3"]["AAA"]
    assert r3.iloc[: e + 2].isna().all()
    c, s = o["close"]["AAA"], o["close"]["SPY"]
    want = (c.iloc[e + 2] / c.iloc[e - 1]) / (s.iloc[e + 2] / s.iloc[e - 1]) - 1
    rel = ((1 + c.pct_change()) / (1 + s.pct_change())).iloc[e:e + 3].prod() - 1
    assert r3.iloc[e + 2] == pytest.approx(want)
    assert r3.iloc[e + 2] == pytest.approx(rel)
    assert (r3.iloc[e + 2:] == r3.iloc[e + 2]).all()  # carried forward

    ds = f["days_since_earn"]["AAA"]
    assert ds.iloc[:e].isna().all() and ds.iloc[e] == 0 and ds.iloc[e + 7] == 7

    gap = f["earn_gap"]["AAA"]
    want_gap = (o["open"]["AAA"].iloc[e] / c.iloc[e - 1]) / (o["open"]["SPY"].iloc[e] / s.iloc[e - 1]) - 1
    assert gap.iloc[:e].isna().all() and gap.iloc[e] == pytest.approx(want_gap)
    assert (gap.iloc[e:] == gap.iloc[e]).all()


def test_vol_spike_and_new_event_resets():
    o = _market()
    idx = o["close"].index
    o["volume"].iloc[30, 0] = 3e6
    f = compute_features({"AAA": pd.DatetimeIndex([idx[30], idx[45]])}, o, ["AAA"])
    vs = f["earn_vol_spike"]["AAA"]
    assert vs.iloc[30] == pytest.approx(3.0) and vs.iloc[44] == pytest.approx(3.0)
    assert vs.iloc[45] == pytest.approx(21 / 23)  # prior 21 sessions include the 3e6 day
    r3 = f["earn_ret_3"]["AAA"]
    assert r3.iloc[45:47].isna().all() and not np.isnan(r3.iloc[47])


def test_event_on_holiday_maps_to_next_session():
    o = _market()
    idx = o["close"].index
    o = {k: v.drop(idx[20]) for k, v in o.items()}
    f = compute_features({"AAA": pd.DatetimeIndex([idx[20]])}, o, ["AAA"])
    assert f["days_since_earn"]["AAA"].loc[idx[21]] == 0


def test_pre_earn_window_uses_only_past_events():
    o = _market(n=120)
    idx = o["close"].index
    ev = [10, 30, 50, 70]
    events = pd.DatetimeIndex(idx[ev])
    full = compute_features({"AAA": events}, o, ["AAA"])["pre_earn_window"]["AAA"]
    assert full.iloc[:30].isna().all()          # fewer than 2 past events
    assert full.iloc[30:45].eq(0).all()          # expected next = 50
    assert full.iloc[45:50].eq(1).all()
    assert full.iloc[50] == 0                    # event happened; next expected 70
    # future events must not change past values
    trunc = compute_features({"AAA": events[:2]}, o, ["AAA"])["pre_earn_window"]["AAA"]
    pd.testing.assert_series_equal(full.iloc[:50], trunc.iloc[:50])
    # overdue grace then back to 0 when the company goes silent
    assert trunc.iloc[55:61].eq(1).all() and trunc.iloc[61:].eq(0).all()
