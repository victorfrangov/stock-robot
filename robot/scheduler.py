"""Long-running scheduler: daily trade before the open, weekly data refresh + retrain."""

from __future__ import annotations

import logging
import time as _time
import traceback
from datetime import datetime, timedelta

from robot.broker import ET
from robot.config import Config

log = logging.getLogger(__name__)


def _next_at(now: datetime, hhmm: str, weekdays: set[int]) -> datetime:
    h, m = map(int, hhmm.split(":"))
    cand = now.replace(hour=h, minute=m, second=0, microsecond=0)
    while cand <= now or cand.weekday() not in weekdays:
        cand = (cand + timedelta(days=1)).replace(hour=h, minute=m)
    return cand


def run_scheduler(cfg: Config) -> None:
    from robot.data.update import update_all
    from robot.live import run_trade
    from robot.pipeline import get_panel, train_production

    s = cfg.schedule
    jobs = {
        "trade": (s.trade_time, set(range(5))),
        "retrain": (s.retrain_time, {int(s.retrain_weekday)}),
    }
    log.info("scheduler started (US/Eastern): %s", {k: v[0] for k, v in jobs.items()})
    while True:
        now = datetime.now(ET)
        name, when = min(((k, _next_at(now, *v)) for k, v in jobs.items()), key=lambda x: x[1])
        log.info("next job: %s at %s", name, when.strftime("%a %Y-%m-%d %H:%M %Z"))
        while datetime.now(ET) < when:
            _time.sleep(min(60, max(1, (when - datetime.now(ET)).total_seconds())))
        try:
            if name == "trade":
                summary = run_trade(cfg)
                log.info("trade done: %s orders, net_liq=%s", len(summary.get("orders", [])), summary.get("net_liq"))
            else:
                update_all(cfg)
                train_production(cfg, panel=get_panel(cfg, rebuild=True))
        except Exception:
            log.error("job %s failed:\n%s", name, traceback.format_exc())
