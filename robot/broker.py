"""Interactive Brokers execution via ib_async (TWS or IB Gateway must be running)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

from ib_async import IB, MarketOrder, Stock, TagValue

from robot.config import Config
from robot.risk import TradingHalted, is_paper_account

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
ORDER_REF = "stock-robot"
LIVE_PORTS = {7496, 4001}


def to_ib(ticker: str) -> str:
    return ticker.replace("-", " ")


def from_ib(symbol: str) -> str:
    return symbol.replace(" ", "-")


def session_phase(now: datetime | None = None) -> str:
    now = (now or datetime.now(ET)).astimezone(ET)
    if now.weekday() >= 5:
        return "closed"
    t = now.time()
    if t < time(9, 28):
        return "pre_open"
    if time(9, 30) <= t < time(15, 58):
        return "rth"
    if t >= time(16, 0):
        return "closed"
    return "auction"  # 09:28-09:30 and 15:58-16:00: too late for MOO, too close for safe MKT


@dataclass
class Fill:
    ticker: str
    status: str
    filled: float
    avg_price: float


class Broker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ib = IB()
        self.account = ""

    def connect(self) -> "Broker":
        b = self.cfg.broker
        if b.paper_only and int(b.port) in LIVE_PORTS:
            raise TradingHalted(f"port {b.port} is a LIVE trading port; paper_only is on (use 4002 or 7497)")
        self.ib.connect(b.host, int(b.port), clientId=int(b.client_id), timeout=20)
        accounts = self.ib.managedAccounts()
        if not accounts:
            raise TradingHalted("no accounts reported by IB")
        self.account = accounts[0]
        if b.paper_only and not is_paper_account(self.account):
            self.ib.disconnect()
            raise TradingHalted(f"account {self.account} is not a paper account - refusing to trade")
        log.info("connected to IB account %s on port %s", self.account, b.port)
        return self

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.disconnect()

    def net_liquidation(self) -> float:
        for v in self.ib.accountSummary(self.account):
            if v.tag == "NetLiquidation" and v.currency in ("USD", "BASE", ""):
                return float(v.value)
        raise RuntimeError("NetLiquidation not available")

    def positions(self) -> dict[str, float]:
        return {from_ib(p.contract.symbol): float(p.position) for p in self.ib.positions(self.account)
                if p.contract.secType == "STK" and p.position != 0}

    def portfolio(self):
        return [p for p in self.ib.portfolio(self.account) if p.contract.secType == "STK"]

    def cancel_robot_orders(self, timeout: float = 15) -> int:
        """Cancel every open stock-robot order, from any client id, and wait until they're gone."""
        self.ib.reqAllOpenOrders()
        self.ib.sleep(1)
        mine = [t for t in self.ib.openTrades() if t.order.orderRef == ORDER_REF]
        for trade in mine:
            self.ib.cancelOrder(trade.order)
        waited = 0.0
        while mine and waited < timeout and not all(t.isDone() for t in mine):
            self.ib.sleep(0.5)
            waited += 0.5
        still = [t for t in mine if not t.isDone()]
        if still:
            raise TradingHalted(f"{len(still)} old robot orders would not cancel - not placing new ones")
        return len(mine)

    def order_style(self) -> str:
        style = self.cfg.broker.order_type
        if style != "auto":
            return style
        phase = session_phase()
        if phase == "rth":
            return "adaptive"
        if phase == "auction":
            raise TradingHalted("inside the opening/closing auction window - try again in a few minutes")
        return "moo"

    def place(self, ticker: str, qty: int, style: str):
        contract = Stock(to_ib(ticker), "SMART", "USD")
        self.ib.qualifyContracts(contract)
        order = MarketOrder("BUY" if qty > 0 else "SELL", abs(qty))
        order.account = self.account
        order.orderRef = ORDER_REF
        if style == "moo":
            order.tif = "OPG"
        elif style == "adaptive":
            order.algoStrategy = "Adaptive"
            order.algoParams = [TagValue("adaptivePriority", "Normal")]
            order.tif = "DAY"
        return self.ib.placeOrder(contract, order)

    def wait(self, trades, seconds: float = 60) -> list[Fill]:
        waited = 0.0
        while waited < seconds and not all(t.isDone() for t in trades):
            self.ib.sleep(1)
            waited += 1
        return [Fill(from_ib(t.contract.symbol), t.orderStatus.status, float(t.orderStatus.filled),
                     float(t.orderStatus.avgFillPrice)) for t in trades]
