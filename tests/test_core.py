import numpy as np
import pandas as pd
import pytest

from robot.backtest import metrics, simulate, train_rows
from robot.broker import Broker, session_phase
from robot.config import load_config
from robot.data.fundamentals import _ttm, company_snapshots
from robot.live import plan_orders
from robot.portfolio import capped_weights, select_targets
from robot.risk import TradingHalted, is_paper_account, validate_order


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOT_DATA_DIR", str(tmp_path))
    return load_config()


# ------------------------------------------------------------------ fundamentals

def _rec(item, start, end, val, filed):
    return {"item": item, "start": pd.Timestamp(start) if start else pd.NaT, "end": pd.Timestamp(end),
            "val": val, "filed": pd.Timestamp(filed)}


def test_ttm_from_ytd_identity():
    recs = pd.DataFrame([
        _rec("revenue", "2020-01-01", "2020-12-31", 100.0, "2021-02-15"),  # FY2020
        _rec("revenue", "2020-01-01", "2020-03-31", 20.0, "2020-05-01"),   # Q1 2020
        _rec("revenue", "2021-01-01", "2021-03-31", 30.0, "2021-05-01"),   # Q1 2021
    ])
    ttm = _ttm(recs).set_index("end")
    assert ttm.loc["2020-12-31", "val"] == 100.0
    assert ttm.loc["2021-03-31", "val"] == pytest.approx(110.0)  # 100 + 30 - 20
    assert ttm.loc["2021-03-31", "avail"] == pd.Timestamp("2021-05-01")


def test_snapshots_are_point_in_time():
    recs = pd.DataFrame([
        _rec("assets", None, "2020-12-31", 1000.0, "2021-02-15"),
        _rec("assets", None, "2021-03-31", 1100.0, "2021-05-01"),
        # an old period re-reported later must not overwrite the newer value
        _rec("assets", None, "2019-12-31", 900.0, "2021-06-01"),
    ])
    snap = company_snapshots(recs)
    assert snap.loc[:"2021-04-30", "assets"].iloc[-1] == 1000.0
    assert snap["assets"].iloc[-1] == 1100.0
    assert pd.Timestamp("2021-02-14") not in snap.index  # nothing known before the first filing


# ------------------------------------------------------------------ leakage

def test_train_rows_purges_label_overlap(cfg):
    dates = pd.bdate_range("2020-01-01", periods=100)
    panel = pd.DataFrame({"date": np.repeat(dates, 2), "ticker": ["A", "B"] * 100, "target": 0.1})
    cfg["model"]["train_sample_every"] = 1
    deploy = dates[60]
    rows = train_rows(panel, cfg, deploy, dates)
    last_train = panel.loc[rows, "date"].max()
    h = cfg.label.horizon
    # the label for day t uses the open of day t+h+1, which must be before deployment
    assert dates.get_loc(last_train) + h + 1 < dates.get_loc(deploy)


# ------------------------------------------------------------------ portfolio

def test_capped_weights_respects_cap_and_total():
    w = capped_weights(pd.Series([10.0, 1, 1, 1, 1]), total=0.9, cap=0.3)
    assert w.max() <= 0.3 + 1e-9
    assert w.sum() == pytest.approx(0.9)


def test_capped_weights_leaves_cash_when_too_few_names():
    w = capped_weights(pd.Series([1.0, 1.0]), total=1.0, cap=0.1)
    assert w.sum() == pytest.approx(0.2)


def test_select_targets_hysteresis(cfg):
    cfg["portfolio"].update(top_k=2, hold_buffer=3, weighting="equal", max_weight=1.0, cash_buffer=0.0)
    scores = pd.Series({"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}, dtype=float)
    px = pd.Series(50.0, index=scores.index)
    vol = pd.Series(0.02, index=scores.index)
    # C is held and still inside the buffer (rank 3) -> kept instead of B
    w = select_targets(cfg, scores, {"C"}, vol, px, regime_risk_off=False)
    assert set(w.index) == {"C", "A"}
    # D (rank 4) falls outside the buffer -> replaced
    w = select_targets(cfg, scores, {"D"}, vol, px, regime_risk_off=False)
    assert set(w.index) == {"A", "B"}
    assert w.sum() == pytest.approx(1.0)


def test_regime_filter_scales_exposure(cfg):
    cfg["portfolio"].update(top_k=2, weighting="equal", max_weight=1.0, cash_buffer=0.0, regime_exposure=0.5)
    scores = pd.Series({"A": 2.0, "B": 1.0})
    w = select_targets(cfg, scores, set(), scores * 0 + 0.02, scores * 0 + 50, regime_risk_off=True)
    assert w.sum() == pytest.approx(0.5)


# ------------------------------------------------------------------ simulation

def _synthetic(n=60):
    dates = pd.bdate_range("2021-01-01", periods=n)
    rows = []
    for t, drift in [("UP", 0.01), ("FLAT", 0.0), ("DOWN", -0.01), ("SPY", 0.001)]:
        close = 100 * (1 + drift) ** np.arange(n)
        for d, c in zip(dates, close):
            rows.append({"date": d, "ticker": t, "open": c, "high": c, "low": c, "close": c,
                         "adj_close": c, "volume": 1e6, "splits": 0.0})
    return dates, pd.DataFrame(rows)


def test_simulate_picks_winner_and_charges_costs(cfg):
    dates, prices = _synthetic()
    cfg["portfolio"].update(top_k=1, hold_buffer=1, weighting="equal", max_weight=1.0, cash_buffer=0.0,
                            regime_filter=False, rebalance_days=1)
    names = ["UP", "FLAT", "DOWN"]
    scores = pd.DataFrame([{"date": d, "ticker": t, "score": {"UP": 1, "FLAT": 0, "DOWN": -1}[t]}
                           for d in dates for t in names])
    panel = scores.assign(raw_vol_63=0.02, raw_price=100.0)
    sim = simulate(cfg, scores, panel, prices, pd.DataFrame())
    # decided at close of day 0, bought at open of day 1 (open == close here), so it earns
    # UP's close-to-close return from day 1 -> 2 onward, minus one round of entry costs.
    cost = (cfg.costs.commission_bps + cfg.costs.slippage_bps) / 1e4
    expected = (1 - cost) * 1.01 ** (len(dates) - 2)
    assert sim.equity.iloc[-1] == pytest.approx(expected, rel=1e-6)
    assert sim.turnover.sum() == pytest.approx(1.0)


def test_metrics_basic():
    r = pd.Series([0.01, -0.005] * 126, index=pd.bdate_range("2020-01-01", periods=252))
    m = metrics(r)
    assert m["max_drawdown"] < 0 and m["cagr"] > 0 and m["sharpe"] > 0


# ------------------------------------------------------------------ safety

def test_paper_account_detection():
    assert is_paper_account("DU1234567")
    assert is_paper_account("DF999")
    assert not is_paper_account("U1234567")


def test_broker_refuses_live_port(cfg):
    cfg["broker"]["port"] = 7496
    with pytest.raises(TradingHalted):
        Broker(cfg).connect()


def test_validate_order(cfg):
    assert validate_order(cfg, "A", 10, 100.0, allow_buys=True) is None
    assert "disabled" in validate_order(cfg, "A", 10, 100.0, allow_buys=False)
    assert validate_order(cfg, "A", -10, 100.0, allow_buys=False) is None  # sells always allowed
    assert "max_order_value" in validate_order(cfg, "A", 10_000, 100.0, allow_buys=True)
    assert validate_order(cfg, "A", -10_000, 100.0, allow_buys=True) is None
    assert validate_order(cfg, "A", -5, float("nan"), allow_buys=True) is None


def test_plan_orders(cfg):
    targets = pd.Series({"A": 0.5, "B": 0.5})
    positions = {"A": 10.0, "C": 7.0}
    prices = pd.Series({"A": 100.0, "B": 50.0, "C": 20.0})
    orders = {o["ticker"]: o for o in plan_orders(cfg, targets, positions, prices, 10_000, True)}
    assert orders["C"]["qty"] == -7          # exit names not in the target
    assert orders["A"]["qty"] == 40          # 50 shares wanted, 10 held
    assert orders["B"]["qty"] == 100
    assert list(orders)[0] == "C"            # sells first


def test_dry_run_does_not_touch_risk_state(cfg):
    from robot.journal import Journal
    from robot.risk import evaluate

    j = Journal(cfg)
    j.record_equity("2026-01-01", 100_000)
    st = evaluate(cfg, j, 70_000, "2026-01-02", persist=False)
    assert st.flatten
    assert j.get("halted") is None and len(j.equity()) == 1


def test_resume_resets_drawdown_peak(cfg):
    from robot.journal import Journal
    from robot.risk import evaluate

    j = Journal(cfg)
    j.record_equity("2026-01-01", 100_000)
    assert evaluate(cfg, j, 70_000, "2026-01-02").flatten
    assert j.get("halted") == "2026-01-02"
    j.set("halted", None)
    j.set("peak_reset", "2026-01-03")  # what `robot resume` does
    st = evaluate(cfg, j, 70_500, "2026-01-05")
    assert not st.flatten and st.allow_buys and j.get("halted") is None


def test_last_completed_session_handles_holidays():
    from robot.broker import ET
    from robot.calendar import last_completed_session

    assert last_completed_session(pd.Timestamp("2026-11-27 09:00", tz=ET)).date().isoformat() == "2026-11-25"
    assert last_completed_session(pd.Timestamp("2026-09-22 16:20", tz=ET)).date().isoformat() == "2026-09-22"


def test_session_phase():
    from robot.broker import ET
    d = pd.Timestamp("2026-09-22 09:00", tz=ET).to_pydatetime()
    assert session_phase(d) == "pre_open"
    assert session_phase(d.replace(hour=11)) == "rth"
    assert session_phase(d.replace(hour=17)) == "closed"
    assert session_phase(pd.Timestamp("2026-09-26 11:00", tz=ET).to_pydatetime()) == "closed"  # Saturday
