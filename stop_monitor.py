"""Fresh-quote stop-loss, take-profit and trailing-stop monitor for paper trading."""
from __future__ import annotations

from datetime import datetime
import sqlite3
from contextlib import closing
from pathlib import Path

from execution_engine import TZ, taiwan_regular_session


class StopMonitor:
    """Persist per-symbol high-water marks and emit risk-exit signals only.

    This module never places orders. Callers must submit a separate guarded
    limit order through PaperExecutionEngine after receiving a signal.
    """

    def __init__(self, db_path: str, *, stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.10, trailing_stop_pct: float = 0.05,
                 max_age_seconds: int = 120):
        for name, value in (("stop_loss_pct", stop_loss_pct),
                            ("take_profit_pct", take_profit_pct),
                            ("trailing_stop_pct", trailing_stop_pct)):
            if not 0 < float(value) < 1:
                raise ValueError(f"{name} 必須介於 0 與 1 之間")
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds 必須大於零")
        self.db_path = str(db_path)
        self.stop_loss_pct = float(stop_loss_pct)
        self.take_profit_pct = float(take_profit_pct)
        self.trailing_stop_pct = float(trailing_stop_pct)
        self.max_age_seconds = int(max_age_seconds)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS paper_stop_state (
                symbol TEXT PRIMARY KEY,
                high_water REAL NOT NULL,
                updated_at TEXT NOT NULL
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS paper_stop_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                reason TEXT NOT NULL,
                price REAL NOT NULL,
                trigger_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                quote_source TEXT NOT NULL,
                quote_time TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(symbol, reason, quote_time)
            )""")
            conn.commit()

    def evaluate_quote(self, symbol: str, price: float, quote_time: str,
                       source: str, *, now: datetime | None = None) -> dict | None:
        current = now or datetime.now(TZ)
        if current.tzinfo is None:
            current = current.replace(tzinfo=TZ)
        current = current.astimezone(TZ)
        try:
            quoted = datetime.fromisoformat(str(quote_time))
            if quoted.tzinfo is None:
                quoted = quoted.replace(tzinfo=TZ)
            quoted = quoted.astimezone(TZ)
        except (TypeError, ValueError):
            raise ValueError("報價時間格式無效，停損監控拒絕執行")
        age = (current - quoted).total_seconds()
        if not source or age < -5 or age > self.max_age_seconds:
            raise ValueError("報價來源缺失或報價已過期，停損監控拒絕執行")
        if not isinstance(price, (int, float)) or price <= 0:
            raise ValueError("報價必須大於零")
        if not taiwan_regular_session(current):
            raise ValueError("非台股一般盤中時段，停損監控拒絕執行")

        symbol = str(symbol).strip().upper()
        with closing(sqlite3.connect(self.db_path, timeout=15)) as conn:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                pos = conn.execute(
                    "SELECT quantity,average_cost FROM paper_positions WHERE symbol=?",
                    (symbol,),
                ).fetchone()
                if not pos or int(pos[0]) <= 0 or float(pos[1]) <= 0:
                    conn.execute("COMMIT")
                    return None
                quantity, avg_cost = int(pos[0]), float(pos[1])
                state = conn.execute(
                    "SELECT high_water FROM paper_stop_state WHERE symbol=?", (symbol,)
                ).fetchone()
                high_water = max(float(state[0]), float(price)) if state else max(avg_cost, float(price))
                conn.execute(
                    "INSERT INTO paper_stop_state(symbol,high_water,updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(symbol) DO UPDATE SET high_water=excluded.high_water,updated_at=excluded.updated_at",
                    (symbol, high_water, current.isoformat()),
                )
                stop_loss = avg_cost * (1 - self.stop_loss_pct)
                trailing_stop = high_water * (1 - self.trailing_stop_pct)
                take_profit = avg_cost * (1 + self.take_profit_pct)
                trigger = None
                if float(price) <= stop_loss:
                    trigger = ("STOP_LOSS", stop_loss)
                elif float(price) <= trailing_stop and high_water > avg_cost:
                    trigger = ("TRAILING_STOP", trailing_stop)
                elif float(price) >= take_profit:
                    trigger = ("TAKE_PROFIT", take_profit)
                if trigger is None:
                    conn.execute("COMMIT")
                    return None
                reason, trigger_price = trigger
                conn.execute(
                    "INSERT OR IGNORE INTO paper_stop_events "
                    "(symbol,reason,price,trigger_price,quantity,quote_source,quote_time,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (symbol, reason, float(price), trigger_price, quantity, source,
                     quoted.isoformat(), current.isoformat()),
                )
                inserted = conn.execute("SELECT changes()").fetchone()[0]
                conn.execute("COMMIT")
                if not inserted:
                    return None
                return {
                    "symbol": symbol, "side": "SELL", "quantity": quantity,
                    "price": float(price), "reason": reason,
                    "trigger_price": trigger_price, "quote_time": quoted.isoformat(),
                    "quote_source": source,
                }
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def events(self, limit: int = 100) -> list[dict]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT symbol,reason,price,trigger_price,quantity,quote_source,quote_time,created_at "
                "FROM paper_stop_events ORDER BY event_id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        keys = ["symbol","reason","price","trigger_price","quantity",
                "quote_source","quote_time","created_at"]
        return [dict(zip(keys, row)) for row in rows]
