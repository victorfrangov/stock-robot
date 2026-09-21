"""Hard risk limits that sit between the model and the broker."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from robot.config import Config
from robot.journal import Journal

log = logging.getLogger(__name__)


class TradingHalted(RuntimeError):
    pass


def check_kill_switch(cfg: Config) -> None:
    path = cfg.root / cfg.risk.kill_switch_file
    if path.exists():
        raise TradingHalted(f"kill switch present ({path}); delete it to resume trading")


def is_paper_account(account: str) -> bool:
    # IBKR paper accounts are prefixed DU (individual) or DF (advisor); live ones are U/F...
    return account.upper().startswith(("DU", "DF"))


@dataclass
class RiskState:
    net_liq: float
    peak: float
    prev: float | None
    drawdown_pct: float = 0.0
    daily_loss_pct: float = 0.0
    allow_buys: bool = True
    flatten: bool = False
    reasons: list[str] = field(default_factory=list)


def evaluate(cfg: Config, journal: Journal, net_liq: float, today: str) -> RiskState:
    eq = journal.equity()
    history = eq[eq.index < today]
    prev = float(history.iloc[-1]) if len(history) else None
    peak = max([net_liq, *eq.tolist()])
    st = RiskState(net_liq=net_liq, peak=peak, prev=prev)
    st.drawdown_pct = (1 - net_liq / peak) * 100 if peak > 0 else 0.0
    if prev:
        st.daily_loss_pct = (1 - net_liq / prev) * 100

    if journal.get("halted"):
        st.allow_buys = False
        st.reasons.append(f"halted since {journal.get('halted')} - run `robot resume` to re-enable")
    if st.drawdown_pct >= cfg.risk.max_drawdown_pct:
        st.flatten, st.allow_buys = True, False
        st.reasons.append(f"drawdown {st.drawdown_pct:.1f}% >= {cfg.risk.max_drawdown_pct}% - flattening")
        journal.set("halted", today)
    if st.daily_loss_pct >= cfg.risk.max_daily_loss_pct:
        st.allow_buys = False
        st.reasons.append(f"daily loss {st.daily_loss_pct:.1f}% >= {cfg.risk.max_daily_loss_pct}% - no buys today")
    journal.record_equity(today, net_liq)
    return st


def validate_order(cfg: Config, ticker: str, qty: int, price: float, allow_buys: bool) -> str | None:
    """Return a rejection reason, or None if the order is fine."""
    if qty == 0:
        return "zero quantity"
    if qty > 0 and not allow_buys:
        return "buys disabled by risk state"
    if not (price and price > 0):
        return None if qty < 0 else "no price"  # never block an exit for lack of a quote
    if qty > 0 and qty * price > cfg.risk.max_order_value:  # sells only ever reduce risk
        return f"notional ${abs(qty) * price:,.0f} > max_order_value ${cfg.risk.max_order_value:,.0f}"
    return None
