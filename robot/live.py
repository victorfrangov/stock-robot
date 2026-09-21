"""The daily trading job: refresh data -> score -> target portfolio -> orders."""

from __future__ import annotations

import logging
import math
from datetime import datetime

import pandas as pd

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
            want = math.floor(targets[t] * net_liq / px)
        else:
            want = 0
        qty = int(want - cur)
        if qty == 0 or (want != 0 and abs(qty) * px < MIN_ORDER_VALUE):
            continue  # skip tiny re-weightings; full exits always go through
        reason = validate_order(cfg, t, qty, px, allow_buys)
        orders.append({"ticker": t, "qty": qty, "price": px, "current": cur, "target": want, "reject": reason})
    return sorted(orders, key=lambda o: o["qty"])  # sells (negative) first


def run_trade(cfg: Config, dry_run: bool = False, force_rebalance: bool = False, refresh_data: bool = True) -> dict:
    journal = Journal(cfg)
    run_id = journal.start_run("trade-dry" if dry_run else "trade")
    try:
        check_kill_switch(cfg)
        if refresh_data:
            update_all(cfg, current_only=True, skip_fundamentals=True)
        prices = load_prices(cfg)
        model = load_model(cfg)
        asof, scored = score_latest(cfg, model)
        today = datetime.now(ET).date()
        if (pd.Timestamp(today) - asof).days > 4:
            raise TradingHalted(f"latest price data is from {asof.date()} - refusing to trade on stale data")

        last_px = prices[prices["date"] == asof].set_index("ticker")["close"]
        risk_off = _risk_off(prices)
        summary = {"asof": str(asof.date()), "model": model.meta.get("trained_at"), "risk_off": risk_off}

        with Broker(cfg) as broker:
            net_liq = broker.net_liquidation()
            positions = broker.positions()
            risk = evaluate(cfg, journal, net_liq, str(today))
            summary.update(account=broker.account, net_liq=net_liq, positions=len(positions),
                           drawdown_pct=round(risk.drawdown_pct, 2), risk=risk.reasons)

            since = _trading_days_since(prices, journal.get("last_rebalance"))
            due = force_rebalance or not positions or since >= cfg.portfolio.rebalance_days or risk.flatten
            scores = scored.set_index("ticker")["score"]
            if risk.flatten:
                targets = pd.Series(dtype=float)
            else:
                targets = select_targets(cfg, scores, set(positions), scored.set_index("ticker")["raw_vol_63"],
                                         scored.set_index("ticker")["raw_price"], risk_off)
            sig = scored[["ticker", "score", "rank"]].copy()
            sig["target_weight"] = sig["ticker"].map(targets).fillna(0.0)
            journal.record_signals(str(asof.date()), sig)
            summary["targets"] = targets.round(4).to_dict()

            if not due:
                msg = f"no rebalance: {since}/{cfg.portfolio.rebalance_days} trading days since last"
                log.info(msg)
                journal.end_run(run_id, "ok", msg)
                return {**summary, "orders": [], "note": msg}

            orders = plan_orders(cfg, targets, positions, last_px, net_liq, risk.allow_buys)
            summary["orders"] = orders
            if dry_run:
                journal.end_run(run_id, "ok", f"dry run: {len(orders)} orders planned")
                return summary

            style = broker.order_style()
            cancelled = broker.cancel_robot_orders()
            if cancelled:
                log.info("cancelled %d stale robot orders", cancelled)
            placed = []
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
            broker.ib.sleep(2)
            fills = broker.wait([t for _, t in placed], seconds=90 if style == "adaptive" else 5)
            for (row, _), f in zip(placed, fills):
                journal.update_order(row, f.status, f.filled, f.avg_price)
            journal.set("last_rebalance", str(asof.date()))
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
        trades = [broker.place(t, -int(q), style) for t, q in broker.positions().items() if int(q) != 0]
        fills = broker.wait(trades, 60 if style == "adaptive" else 5)
    journal.end_run(run_id, "ok", f"flatten: {len(trades)} orders")
    return fills
