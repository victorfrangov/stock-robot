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


def evaluate(cfg: Config, journal: Journal, net_liq: float, today: str, persist: bool = True) -> RiskState:
    """Risk state for today. `persist=False` (dry runs) leaves the journal untouched."""
    eq = journal.equity()
    history = eq[eq.index < today]
    prev = float(history.iloc[-1]) if len(history) else None
    reset = journal.get("peak_reset")  # set by `robot resume` so an old peak can't re-trigger the halt
    since = eq[eq.index >= reset] if reset else eq
    peak = max([net_liq, *since.tolist()])
    st = RiskState(net_liq=net_liq, peak=peak, prev=prev)
    st.drawdown_pct = (1 - net_liq / peak) * 100 if peak > 0 else 0.0
    if prev:
        st.daily_loss_pct = (1 - net_liq / prev) * 100

    halted = journal.get("halted")
    if halted:
        st.allow_buys = False
        st.reasons.append(f"halted since {halted} - run `robot resume` to re-enable")
    if st.drawdown_pct >= cfg.risk.max_drawdown_pct and not halted:
        st.flatten, st.allow_buys = True, False
        st.reasons.append(f"drawdown {st.drawdown_pct:.1f}% >= {cfg.risk.max_drawdown_pct}% - flattening")
        if persist:
            journal.set("halted", today)
    if st.daily_loss_pct >= cfg.risk.max_daily_loss_pct:
        st.allow_buys = False
        st.reasons.append(f"daily loss {st.daily_loss_pct:.1f}% >= {cfg.risk.max_daily_loss_pct}% - no buys today")
    if persist:
        journal.record_equity(today, net_liq)
    return st


def validate_order(cfg: Config, ticker: str, qty: int, price: float, allow_buys: bool,
                   net_liq: float | None = None) -> str | None:
    """Return a rejection reason, or None if the order is fine."""
    if qty == 0:
        return "zero quantity"
    if qty > 0 and not allow_buys:
        return "buys disabled by risk state"
    if not (price and price > 0):
        return None if qty < 0 else "no price"  # never block an exit for lack of a quote
    if qty < 0:
        return None  # sells only ever reduce risk
    notional = qty * price
    cap_abs = cfg.risk.get("max_order_value")
    if cap_abs and notional > cap_abs:
        return f"notional ${notional:,.0f} > max_order_value ${cap_abs:,.0f}"
    cap_pct = cfg.risk.get("max_order_pct")
    if cap_pct and net_liq and notional > cap_pct * net_liq:
        return f"notional ${notional:,.0f} > {cap_pct:.0%} of equity (${cap_pct * net_liq:,.0f})"
    return None
