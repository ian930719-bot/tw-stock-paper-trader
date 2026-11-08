"""Persistent adaptive exit monitor inspired by O'Neil, Minervini and ATR risk control.

The monitor emits paper exit signals only; it never submits orders. Stops use
completed-session ATR/structure supplied by the caller and fresh intraday quotes.
There is no fixed take-profit: winners are managed with protective trailing stops.
"""
from __future__ import annotations

from datetime import datetime
import math
import sqlite3
from contextlib import closing
from pathlib import Path

from execution_engine import TZ, taiwan_regular_session
from strategy_exits import initial_stop, trailing_stop


PROFILES = {"oneil_growth", "minervini_sepa", "atr_dynamic"}


class AdaptiveExitMonitor:
    def __init__(self, db_path: str, *, max_age_seconds: int = 60):
        if isinstance(max_age_seconds, bool) or int(max_age_seconds) <= 0:
            raise ValueError("max_age_seconds 必須大於零")
        self.db_path = str(db_path)
        self.max_age_seconds = int(max_age_seconds)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS adaptive_exit_state (
                    symbol TEXT PRIMARY KEY,
                    quantity INTEGER NOT NULL,
                    position_version INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    initial_risk REAL NOT NULL,
                    high_water REAL NOT NULL,
                    atr REAL NOT NULL,
                    atr_multiple REAL NOT NULL,
                    profile TEXT NOT NULL,
                    stop_price REAL NOT NULL,
                    profit_review_triggered INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS adaptive_exit_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    quote_source TEXT NOT NULL,
                    quote_time TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    position_version INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(symbol, position_version, quote_time)
                )
            """)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(adaptive_exit_events)")}
            if "position_version" not in columns:
                conn.execute("ALTER TABLE adaptive_exit_events ADD COLUMN position_version INTEGER NOT NULL DEFAULT 1")
            conn.commit()

    def configure_position(
        self, symbol: str, quantity: int, entry_price: float, atr: float, *,
        position_version: int, profile: str = "atr_dynamic",
        atr_multiple: float = 2.0, technical_invalidation: float | None = None,
        now: datetime | None = None,
    ) -> dict:
        code = str(symbol).strip().upper()
        if not code.isalnum() or not 4 <= len(code) <= 8:
            raise ValueError("證券代號格式不合法")
        if isinstance(quantity, bool) or int(quantity) <= 0:
            raise ValueError("持股股數必須大於零")
        if isinstance(position_version, bool) or int(position_version) <= 0:
            raise ValueError("持倉版本必須大於零")
        if profile not in PROFILES:
            raise ValueError("未知的大師出場策略")
        entry, volatility, multiple = float(entry_price), float(atr), float(atr_multiple)
        if not all(math.isfinite(x) and x > 0 for x in (entry, volatility, multiple)):
            raise ValueError("進場價、ATR 與 ATR 倍數必須是有限正數")
        stop_info = initial_stop(
            entry, profile, technical_invalidation=technical_invalidation,
            atr=volatility, atr_multiple=multiple,
        )
        stop = float(stop_info["stop_price"])
        if stop >= entry:
            raise ValueError("初始停損必須低於進場價")
        stamp = (now or datetime.now(TZ)).astimezone(TZ).isoformat()
        with closing(sqlite3.connect(self.db_path)) as conn:
            existing = conn.execute(
                "SELECT position_version,stop_price,high_water FROM adaptive_exit_state WHERE symbol=?",
                (code,),
            ).fetchone()
            if existing and int(existing[0]) == int(position_version):
                # Refresh ATR and quantity without loosening an already-protective stop.
                stop = max(stop, float(existing[1]))
                high = max(entry, float(existing[2]))
            else:
                high = entry
            conn.execute("""
                INSERT INTO adaptive_exit_state(
                    symbol,quantity,position_version,entry_price,initial_risk,
                    high_water,atr,atr_multiple,profile,stop_price,
                    profit_review_triggered,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,0,?)
                ON CONFLICT(symbol) DO UPDATE SET
                    quantity=excluded.quantity,
                    position_version=excluded.position_version,
                    entry_price=excluded.entry_price,
                    initial_risk=excluded.initial_risk,
                    high_water=excluded.high_water,
                    atr=excluded.atr,
                    atr_multiple=excluded.atr_multiple,
                    profile=excluded.profile,
                    stop_price=excluded.stop_price,
                    profit_review_triggered=CASE
                        WHEN adaptive_exit_state.position_version=excluded.position_version
                        THEN adaptive_exit_state.profit_review_triggered ELSE 0 END,
                    updated_at=excluded.updated_at
            """, (code, int(quantity), int(position_version), entry,
                  max(entry - stop, 0.01), high, volatility, multiple,
                  profile, stop, stamp))
            conn.commit()
        return {"symbol": code, "profile": profile, "stop_price": stop,
                "high_water": high, "atr": volatility}

    def evaluate_quote(
        self, symbol: str, price: float, quote_time: str, source: str, *,
        now: datetime | None = None,
    ) -> dict | None:
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
            raise ValueError("報價時間格式無效，拒絕執行出場監控")
        age = (current - quoted).total_seconds()
        if not source or age < -5 or age > self.max_age_seconds:
            raise ValueError("報價來源缺失或報價已過期，拒絕執行出場監控")
        if not math.isfinite(float(price)) or float(price) <= 0:
            raise ValueError("報價必須是有限正數")
        if not taiwan_regular_session(current):
            raise ValueError("非台股一般盤中時段，拒絕執行出場監控")
        code = str(symbol).strip().upper()
        with closing(sqlite3.connect(self.db_path, timeout=15)) as conn:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("""
                    SELECT quantity,position_version,entry_price,initial_risk,
                           high_water,atr,atr_multiple,profile,stop_price,
                           profit_review_triggered
                    FROM adaptive_exit_state WHERE symbol=?
                """, (code,)).fetchone()
                if not row:
                    conn.execute("COMMIT")
                    return None
                qty, version, entry, initial_risk, high, atr, multiple, profile, stop, reviewed = row
                px = float(price)
                high = max(float(high), px)
                risk = max(float(initial_risk), 0.01)
                r_multiple = (high - float(entry)) / risk
                reviewed = bool(reviewed)
                if profile == "oneil_growth":
                    if px >= float(entry) * 1.20:
                        reviewed = True
                    if reviewed:
                        stop = max(float(stop), trailing_stop(high, float(atr), 3.0, float(stop)))
                elif profile == "minervini_sepa":
                    if r_multiple >= 2.0:
                        stop = max(float(stop), float(entry))
                    if r_multiple >= 3.0:
                        stop = max(float(stop), trailing_stop(high, float(atr), 2.5, float(stop)))
                else:
                    stop = max(float(stop), trailing_stop(high, float(atr), float(multiple), float(stop)))
                stamp = current.isoformat()
                conn.execute("""
                    UPDATE adaptive_exit_state SET high_water=?,stop_price=?,
                        profit_review_triggered=?,updated_at=? WHERE symbol=?
                """, (high, stop, int(reviewed), stamp, code))
                if px > stop:
                    conn.execute("COMMIT")
                    return None
                conn.execute("""
                    INSERT OR IGNORE INTO adaptive_exit_events(
                        symbol,reason,price,stop_price,quantity,quote_source,
                        quote_time,created_at,position_version
                    ) VALUES(?,?,?,?,?,?,?,?,?)
                """, (code, "ADAPTIVE_STOP", px, stop, int(qty), source,
                      quoted.isoformat(), stamp, int(version)))
                inserted = conn.execute("SELECT changes()").fetchone()[0]
                conn.execute("COMMIT")
                if not inserted:
                    return None
                return {
                    "symbol": code, "side": "SELL", "quantity": int(qty),
                    "price": px, "trigger_price": float(stop),
                    "reason": f"{profile} 動態保護停損",
                    "r_multiple_at_high": round(r_multiple, 2),
                    "quote_time": quoted.isoformat(), "quote_source": source,
                }
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def events(self, limit: int = 100) -> list[dict]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute("""
                SELECT symbol,reason,price,stop_price,quantity,quote_source,
                       quote_time,created_at,position_version
                FROM adaptive_exit_events ORDER BY event_id DESC LIMIT ?
            """, (int(limit),)).fetchall()
        keys = ["symbol","reason","price","stop_price","quantity","quote_source",
                "quote_time","created_at","position_version"]
        return [dict(zip(keys, row)) for row in rows]
