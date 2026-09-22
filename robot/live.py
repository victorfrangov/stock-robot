"""The daily trading job: refresh data -> score -> target portfolio -> orders."""

from __future__ import annotations

import logging
import math
from datetime import datetime

import pandas as pd

from robot import calendar
from robot.broker import ET, Broker
from robot.config import Config
from robot.data import universe
from robot.data.prices import load_prices
from robot.data.update import update_all
from robot.journal import Journal
from robot.pipeline import load_model, score_latest
from robot.portfolio import select_targets
from robot.risk import TradingHalted, check_kill_switch, evaluate, validate_order

log = logging.getLogger(__name__)
MIN_ORDER_VALUE = 100.0
SIZING_BUFFER = 0.01


def _trading_days_since(prices: pd.DataFrame, since: str | None) -> int:
    if not since:
        return 10**6
    days = prices.loc[prices["ticker"] == universe.BENCHMARK, "date"]
    return int((days > pd.Timestamp(since)).sum())


def _risk_off(prices: pd.DataFrame) -> bool:
    spy = prices[prices["ticker"] == universe.BENCHMARK].set_index("date")["adj_close"].sort_index()
    return bool(spy.iloc[-1] < spy.rolling(200).mean().iloc[-1])


def plan_orders(cfg: Config, targets: pd.Series, positions: dict[str, float], prices: pd.Series,
                net_liq: float, allow_buys: bool) -> list[dict]:
    """Share deltas needed to move from `positions` to `targets` (sells first)."""
    orders = []
    for t in sorted(set(targets.index) | set(positions)):
        px = float(prices.get(t, float("nan")))
        has_px = math.isfinite(px) and px > 0
        cur = positions.get(t, 0.0)
        if t in targets:
            if not has_px:
                continue  # can't size a position without a price
            # Sized from yesterday's close but filled at the open: leave 1% for a gap up so a
            # fully invested book never dips into margin.
            want = math.floor(targets[t] * net_liq * (1 - SIZING_BUFFER) / px)
        else:
            want = 0
        qty = int(want - cur)
        band = cfg.portfolio.get("trade_band", 0.0) * net_liq
        if qty == 0 or (want != 0 and abs(qty) * px < max(MIN_ORDER_VALUE, band if cur else 0)):
            continue  # skip tiny re-weightings (same no-trade band as the backtest); exits always go
        reason = validate_order(cfg, t, qty, px, allow_buys, net_liq)
        orders.append({"ticker": t, "qty": qty, "price": px, "current": cur, "target": want, "reject": reason})
    return sorted(orders, key=lambda o: o["qty"])  # sells (negative) first


def _managed(cfg: Config, journal: Journal, positions: dict[str, float]) -> dict[str, float]:
    """Positions the robot is allowed to touch: its own, unless manage_all_positions is on."""
    if cfg.broker.get("manage_all_positions", False):
        return positions
    owned = set(journal.get("owned", []))
    return {t: q for t, q in positions.items() if t in owned}


def _price_mismatches(broker: Broker, prices: pd.Series, tol: float = 0.25) -> set[str]:
    """Held names whose IBKR price disagrees with our data by >25% (a split not yet in our data)."""
    bad = set()
    for item in broker.portfolio():
        t = item.contract.symbol.replace(" ", "-")
        ours, theirs = float(prices.get(t, float("nan"))), float(item.marketPrice or float("nan"))
        if math.isfinite(ours) and math.isfinite(theirs) and theirs > 0 and abs(theirs / ours - 1) > tol:
            bad.add(t)
    return bad


def run_trade(cfg: Config, dry_run: bool = False, force_rebalance: bool = False, refresh_data: bool = True) -> dict:
    journal = Journal(cfg)
    run_id = journal.start_run("trade-dry" if dry_run else "trade")
    try:
        check_kill_switch(cfg)
        now = datetime.now(ET)
        if not dry_run and not force_rebalance and not calendar.is_session(now.date()) and now.hour < 16:
            msg = f"{now.date()} is not an NYSE session - nothing to do"
            journal.end_run(run_id, "ok", msg)
            return {"asof": None, "risk_off": None, "orders": [], "note": msg}
        expected = calendar.last_completed_session(now)
        if refresh_data:
            update_all(cfg, current_only=True, skip_fundamentals=True)
        prices = load_prices(cfg)
        prices = prices[prices["date"] <= expected]  # never use a partial intraday bar
        model = load_model(cfg)
        asof, scored = score_latest(cfg, model, cutoff=expected)
        if asof != expected:
            raise TradingHalted(f"latest price data is from {asof.date()} but the last session was "
                                f"{expected.date()} - refusing to trade on stale data")

        last_px = prices[prices["date"] == asof].set_index("ticker")["close"]
        risk_off = _risk_off(prices)
        summary = {"asof": str(asof.date()), "model": model.meta.get("trained_at"), "risk_off": risk_off}

        with Broker(cfg) as broker:
            if not dry_run:
                cancelled = broker.cancel_robot_orders()  # before reading positions: no double orders
                if cancelled:
                    log.info("cancelled %d stale robot orders", cancelled)
            net_liq = broker.net_liquidation()
            all_positions = broker.positions()
            # Ownership is reconciled against what is actually held: a buy that never filled
            # drops out, a sell that never filled stays owned and is retried next rebalance.
            if not dry_run and not cfg.broker.get("manage_all_positions", False):
                journal.set("owned", sorted(t for t in journal.get("owned", []) if t in all_positions))
            positions = _managed(cfg, journal, all_positions)
            unmanaged = sorted(set(all_positions) - set(positions))
            risk = evaluate(cfg, journal, net_liq, str(now.date()), persist=not dry_run)
            summary.update(account=broker.account, net_liq=net_liq, positions=len(positions),
                           unmanaged=unmanaged, drawdown_pct=round(risk.drawdown_pct, 2), risk=risk.reasons)
            if unmanaged:
                log.info("leaving %d positions the robot didn't open: %s", len(unmanaged), unmanaged)
            # A held name with no bar for `asof` would look like it left the universe and be sold.
            no_bar = sorted(t for t in positions if t not in set(scored["ticker"]))
            if no_bar and not risk.flatten:
                log.warning("no price bar today for held %s - keeping them untouched", no_bar)
                positions = {t: q for t, q in positions.items() if t not in no_bar}
                summary["no_bar"] = no_bar

            since = _trading_days_since(prices, journal.get("last_rebalance"))
            due = force_rebalance or not positions or since >= cfg.portfolio.rebalance_days or risk.flatten
            if due and not risk.allow_buys and not risk.flatten:
                # Selling without buying would leave a lopsided book the backtest never models;
                # wait for the next session instead (the halt/daily-loss reason is logged).
                msg = f"rebalance postponed: {'; '.join(risk.reasons)}"
                log.warning(msg)
                journal.end_run(run_id, "ok", msg)
                return {**summary, "orders": [], "note": msg}
            scores = scored.set_index("ticker")["score"]
            if risk.flatten:
                targets = pd.Series(dtype=float)
            else:
                sc = scored.set_index("ticker")
                targets = select_targets(cfg, scores, set(positions), sc["raw_vol_63"], sc["raw_price"], risk_off,
                                         sc["raw_mcap"] if "raw_mcap" in sc else None)
            sig = scored[["ticker", "score", "rank"]].copy()
            sig["target_weight"] = sig["ticker"].map(targets).fillna(0.0)
            if not dry_run:
                journal.record_signals(str(asof.date()), sig)
            summary["targets"] = targets.round(4).to_dict()

            if not due:
                msg = f"no rebalance: {since}/{cfg.portfolio.rebalance_days} trading days since last"
                log.info(msg)
                journal.end_run(run_id, "ok", msg)
                return {**summary, "orders": [], "note": msg}

            orders = plan_orders(cfg, targets, positions, last_px, net_liq, risk.allow_buys)
            suspect = _price_mismatches(broker, last_px)
            for o in orders:
                if o["ticker"] in suspect and not o["reject"]:
                    o["reject"] = "IBKR price differs >25% from our data (split?) - skipped this run"
            summary["orders"] = orders
            if dry_run:
                journal.end_run(run_id, "ok", f"dry run: {len(orders)} orders planned")
                return summary

            style = broker.order_style()
            placed = []
            owned = set(journal.get("owned", []))
            for o in orders:
                if o["reject"]:
                    log.warning("skip %s %+d: %s", o["ticker"], o["qty"], o["reject"])
                    journal.record_order(run_id, ticker=o["ticker"], action="BUY" if o["qty"] > 0 else "SELL",
                                         qty=o["qty"], order_type=style, est_price=o["price"],
                                         status=f"rejected: {o['reject']}")
                    continue
                trade = broker.place(o["ticker"], o["qty"], style)
                row = journal.record_order(run_id, ticker=o["ticker"], action=trade.order.action, qty=o["qty"],
                                           order_type=style, est_price=o["price"], status="submitted",
                                           ib_order_id=trade.order.orderId)
                placed.append((row, trade))
                if o["target"] > 0:
                    owned.add(o["ticker"])  # exits leave `owned` at the next run's reconciliation
            journal.set("owned", sorted(owned))
            journal.set("last_rebalance", str(asof.date()))
            broker.ib.sleep(2)
            fills = broker.wait([t for _, t in placed], seconds=90 if style == "adaptive" else 5)
            for (row, _), f in zip(placed, fills):
                journal.update_order(row, f.status, f.filled, f.avg_price)
            summary["style"] = style
            summary["fills"] = [f.__dict__ for f in fills]
        journal.end_run(run_id, "ok", f"{len(placed)} orders placed ({style})")
        return summary
    except Exception as e:
        journal.end_run(run_id, "error", repr(e))
        raise


def flatten(cfg: Config) -> list:
    journal = Journal(cfg)
    run_id = journal.start_run("flatten")
    with Broker(cfg) as broker:
        broker.cancel_robot_orders()
        style = broker.order_style()
        held = _managed(cfg, journal, broker.positions())
        trades = [broker.place(t, -int(q), style) for t, q in held.items() if int(q) != 0]
        journal.set("owned", [])
        fills = broker.wait(trades, 60 if style == "adaptive" else 5)
    journal.end_run(run_id, "ok", f"flatten: {len(trades)} orders")
    return fills
