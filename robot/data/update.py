"""One call to refresh every data source."""

from __future__ import annotations

import logging

from robot.config import Config
from robot.data import earnings, fundamentals, macro, prices, universe

log = logging.getLogger(__name__)


def update_all(cfg: Config, full: bool = False, skip_fundamentals: bool = False, current_only: bool = False) -> None:
    universe.membership(cfg, refresh=full)
    if current_only:  # daily live run: only names that can actually be traded today
        tickers = universe.current_members(cfg)
    else:
        tickers = universe.tickers_since(cfg, cfg.data.start_date)
    tickers = sorted(set(tickers) | {universe.BENCHMARK})
    log.info("updating %d tickers", len(tickers))
    prices.update_prices(cfg, tickers, full=full)
    macro.update_macro(cfg)
    if not skip_fundamentals:
        stocks = [t for t in tickers if t != universe.BENCHMARK]
        fundamentals.update_fundamentals(cfg, stocks)
        earnings.update_earnings(cfg, stocks)
