"""SQLite journal: runs, equity, orders, signals and persistent state."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import pandas as pd

from robot.config import Config

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, ts TEXT, kind TEXT, status TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS equity (date TEXT PRIMARY KEY, net_liq REAL, ts TEXT);
CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, run_id INTEGER, ts TEXT, ticker TEXT, action TEXT,
    qty REAL, order_type TEXT, est_price REAL, status TEXT, filled REAL, avg_price REAL, ib_order_id INTEGER);
CREATE TABLE IF NOT EXISTS signals (date TEXT, ticker TEXT, score REAL, rank INTEGER, target_weight REAL,
    PRIMARY KEY (date, ticker));
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
"""


class Journal:
    def __init__(self, cfg: Config):
        self.db = sqlite3.connect(cfg.path("state", "journal.sqlite"))
        self.db.executescript(SCHEMA)

    def start_run(self, kind: str) -> int:
        cur = self.db.execute("INSERT INTO runs (ts, kind, status) VALUES (?, ?, 'running')",
                              (datetime.now().isoformat(timespec="seconds"), kind))
        self.db.commit()
        return cur.lastrowid

    def end_run(self, run_id: int, status: str, message: str = "") -> None:
        self.db.execute("UPDATE runs SET status=?, message=? WHERE id=?", (status, message[:2000], run_id))
        self.db.commit()

    def record_equity(self, date: str, net_liq: float) -> None:
        self.db.execute("INSERT OR REPLACE INTO equity VALUES (?, ?, ?)",
                        (date, net_liq, datetime.now().isoformat(timespec="seconds")))
        self.db.commit()

    def equity(self) -> pd.Series:
        df = pd.read_sql("SELECT date, net_liq FROM equity ORDER BY date", self.db)
        return pd.Series(df["net_liq"].to_numpy(), index=pd.to_datetime(df["date"]), dtype=float)

    def record_order(self, run_id: int, **kw) -> int:
        cols = ["ticker", "action", "qty", "order_type", "est_price", "status", "filled", "avg_price", "ib_order_id"]
        cur = self.db.execute(
            f"INSERT INTO orders (run_id, ts, {', '.join(cols)}) VALUES (?, ?, {', '.join('?' * len(cols))})",
            (run_id, datetime.now().isoformat(timespec="seconds"), *[kw.get(c) for c in cols]))
        self.db.commit()
        return cur.lastrowid

    def update_order(self, row_id: int, status: str, filled: float, avg_price: float) -> None:
        self.db.execute("UPDATE orders SET status=?, filled=?, avg_price=? WHERE id=?",
                        (status, filled, avg_price, row_id))
        self.db.commit()

    def record_signals(self, date: str, df: pd.DataFrame) -> None:
        self.db.execute("DELETE FROM signals WHERE date=?", (date,))
        self.db.executemany("INSERT INTO signals VALUES (?, ?, ?, ?, ?)",
                            [(date, r.ticker, float(r.score), int(r.rank), float(r.target_weight))
                             for r in df.itertuples()])
        self.db.commit()

    def get(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key: str, value) -> None:
        self.db.execute("INSERT OR REPLACE INTO state VALUES (?, ?)", (key, json.dumps(value, default=str)))
        self.db.commit()

    def recent(self, table: str, n: int = 20) -> pd.DataFrame:
        order = "date" if table in ("equity", "signals") else "id"
        return pd.read_sql(f"SELECT * FROM {table} ORDER BY {order} DESC LIMIT {int(n)}", self.db)
