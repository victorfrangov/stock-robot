"""NYSE trading calendar helpers."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache

import pandas as pd

from robot.broker import ET


@lru_cache(maxsize=1)
def _nyse():
    import exchange_calendars as xc

    return xc.get_calendar("XNYS")


def is_session(day) -> bool:
    return bool(_nyse().is_session(pd.Timestamp(day).normalize()))


def last_completed_session(now: datetime | None = None) -> pd.Timestamp:
    """Most recent session whose close (+15 min for data to settle) is in the past."""
    now = pd.Timestamp(now or datetime.now(ET))
    cal = _nyse()
    sessions = cal.sessions_in_range(now.normalize().tz_localize(None) - pd.Timedelta(days=14),
                                     now.normalize().tz_localize(None))
    done = [s for s in sessions if cal.session_close(s) + pd.Timedelta(minutes=15) <= now]
    return pd.Timestamp(done[-1]).tz_localize(None)


def next_session(day) -> pd.Timestamp:
    cal = _nyse()
    d = pd.Timestamp(day).normalize()
    return pd.Timestamp(cal.date_to_session(d, direction="next") if not cal.is_session(d) else cal.next_session(d))
