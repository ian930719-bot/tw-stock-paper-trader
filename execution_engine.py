"""Guarded order-intent and paper execution engine.

This module intentionally has no live broker side effects. Live routing is a
separate future adapter and must not be enabled by this paper engine.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, time
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
import os
import tempfile
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Taipei")


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    REJECTED = "REJECTED"
    ACCEPTED = "ACCEPTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class RiskLimits:
    max_order_notional: float = 100_000
    max_symbol_exposure: float = 200_000
    max_portfolio_exposure: float = 800_000
    max_daily_loss: float = 20_000
    max_open_orders: int = 5
    max_order_count_per_day: int = 20
    allow_short: bool = False
    require_regular_session: bool = True


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: Side
    quantity: int
    limit_price: float
    reason: str
    idempotency_key: str
    instrument_type: str = "stock"


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    quantity: int
    limit_price: float
    status: str
    message: str
    filled_quantity: int = 0
    average_fill_price: float | None = None
    fees: float = 0.0
    created_at: str = ""


def realized_pnl_from_trades(trades: list[dict], trading_date: str) -> float | None:
    """FIFO realized P&L from the complete paper trade history.

    Returns None if sells cannot be matched to recorded buys (for example,
    imported starting positions without cost basis); callers must fail closed.
    """
    lots: dict[str, list[list[float]]] = {}
    realized = 0.0
    for trade in sorted(trades, key=lambda item: str(item.get("時間", ""))):
        symbol = str(trade.get("代號", "")).upper()
        side = trade.get("方向")
        quantity = int(trade.get("股數", 0))
        amount = float(trade.get("估算金額", 0))
        if not symbol or quantity <= 0 or amount < 0:
            return None
        if side == "買進":
            lots.setdefault(symbol, []).append([float(quantity), amount / quantity])
        elif side == "賣出":
            remaining = float(quantity)
            cost = 0.0
            inventory = lots.setdefault(symbol, [])
            while remaining > 0 and inventory:
                take = min(remaining, inventory[0][0])
                cost += take * inventory[0][1]
                inventory[0][0] -= take
                remaining -= take
                if inventory[0][0] <= 1e-9:
                    inventory.pop(0)
            if remaining > 1e-9:
                return None
            if str(trade.get("時間", "")).startswith(trading_date):
                realized += amount - cost
        else:
            return None
    return realized


def taiwan_regular_session(now: datetime | None = None) -> bool:
    current = now or datetime.now(TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=TZ)
    current = current.astimezone(TZ)
    return current.weekday() < 5 and time(9, 0) <= current.time().replace(tzinfo=None) <= time(13, 25)


def _positive_decimal(value, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"{label}必須是有效數字")
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{label}必須大於零")
    return result


def validate_order(intent: OrderIntent, *, cash: float, positions: dict[str, int],
                   equity: float, daily_realized_pnl: float,
                   open_order_count: int, daily_order_count: int,
                   limits: RiskLimits, now: datetime | None = None) -> None:
    symbol = str(intent.symbol).strip().upper()
    if not symbol.isalnum() or not 4 <= len(symbol) <= 8:
        raise ValueError("證券代號格式不合法")
    if intent.side not in (Side.BUY, Side.SELL):
        raise ValueError("不支援的買賣方向")
    if isinstance(intent.quantity, bool) or not isinstance(intent.quantity, int) or intent.quantity <= 0:
        raise ValueError("委託股數必須是正整數")
    price = _positive_decimal(intent.limit_price, "限價")
    from taiwan_market_rules import is_valid_stock_tick
    if not is_valid_stock_tick(price, intent.instrument_type):
        raise ValueError("限價不符合台股普通股票升降單位")
    if intent.quantity % 1000 != 0:
        raise ValueError("普通交易股數須為 1,000 股整數倍；零股需使用獨立零股撮合")
    notional = price * intent.quantity
    if notional > Decimal(str(limits.max_order_notional)):
        raise ValueError("超過單筆委託金額上限")
    if open_order_count >= limits.max_open_orders:
        raise ValueError("未完成委託數已達上限")
    if daily_order_count >= limits.max_order_count_per_day:
        raise ValueError("今日委託次數已達上限")
    if daily_realized_pnl is None or not float(daily_realized_pnl) == float(daily_realized_pnl):
        raise ValueError("每日已實現損益無法核對，為安全起見停止自動委託")
    if daily_realized_pnl <= -abs(limits.max_daily_loss):
        raise ValueError("已觸及每日虧損停手機制")
    if limits.require_regular_session and not taiwan_regular_session(now):
        raise ValueError("目前不在台股一般盤中交易時段")
    held = int(positions.get(symbol, 0))
    if intent.side == Side.SELL:
        if intent.quantity > held:
            raise ValueError("賣出股數超過可用持股；目前不允許放空")
    elif not limits.allow_short:
        if notional * Decimal("1.001425") > Decimal(str(max(0.0, cash))):
            raise ValueError("可用現金不足（含估算買進手續費）")
        projected = Decimal(str(held)) * price + notional
        if projected > Decimal(str(limits.max_symbol_exposure)):
            raise ValueError("單一標的曝險超過上限")
        # Equity minus cash is a conservative proxy for existing invested
        # exposure; avoid valuing every holding at the new order's price.
        current_exposure = max(0.0, float(equity) - max(0.0, float(cash)))
        if Decimal(str(current_exposure)) + notional > Decimal(str(limits.max_portfolio_exposure)):
            raise ValueError("投資組合曝險超過上限")
    if not intent.idempotency_key or len(intent.idempotency_key) > 128:
        raise ValueError("缺少有效的防重複委託識別碼")


class PaperExecutionEngine:
    """SQLite-backed paper order log; never connects to a broker."""

    def __init__(self, db_path: str | None = None, limits: RiskLimits | None = None):
        self.db_path = db_path or os.getenv(
            "EXECUTION_DB_PATH",
            str(Path(tempfile.gettempdir()) / "tw_stock_execution.sqlite3"),
        )
        self.limits = limits or RiskLimits()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_orders (
                    order_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    limit_price REAL NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    filled_quantity INTEGER NOT NULL DEFAULT 0,
                    average_fill_price REAL,
                    fees REAL NOT NULL DEFAULT 0,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def _existing(self, key: str):
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT order_id,symbol,side,quantity,limit_price,status,message,"
                "filled_quantity,average_fill_price,fees,created_at "
                "FROM paper_orders WHERE idempotency_key=?", (key,)
            ).fetchone()
        return row

    def submit(self, intent: OrderIntent, *, cash: float, positions: dict[str, int],
               equity: float, daily_realized_pnl: float = 0.0,
               open_order_count: int = 0, daily_order_count: int = 0,
               now: datetime | None = None) -> OrderResult:
        existing = self._existing(intent.idempotency_key)
        if existing:
            return OrderResult(*existing[:7], *existing[7:10], created_at=existing[10])
        try:
            validate_order(
                intent, cash=cash, positions=positions, equity=equity,
                daily_realized_pnl=daily_realized_pnl,
                open_order_count=open_order_count, daily_order_count=daily_order_count,
                limits=self.limits, now=now,
            )
            status, message = OrderStatus.ACCEPTED.value, "模擬委託已接受，等待價格觸發"
        except ValueError as exc:
            status, message = OrderStatus.REJECTED.value, str(exc)

        created = (now or datetime.now(TZ)).astimezone(TZ).isoformat()
        digest = hashlib.sha256(intent.idempotency_key.encode("utf-8")).hexdigest()[:20]
        order_id = "PAPER-" + digest
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO paper_orders(
                    order_id,idempotency_key,symbol,side,quantity,limit_price,status,
                    message,reason,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """, (order_id, intent.idempotency_key, intent.symbol.upper(),
                  intent.side.value, intent.quantity, float(intent.limit_price),
                  status, message, intent.reason, created, created))
            conn.commit()
        row = self._existing(intent.idempotency_key)
        return OrderResult(*row[:7], *row[7:10], created_at=row[10])

    def match_limit(self, order_id: str, market_price: float, *,
                    buy_fee_rate: float = 0.001425,
                    sell_fee_rate: float = 0.001425,
                    sell_tax_rate: float = 0.003,
                    now: datetime | None = None) -> OrderResult:
        market = float(_positive_decimal(market_price, "市場價格"))
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT symbol,side,quantity,limit_price,status FROM paper_orders WHERE order_id=?",
                (order_id,),
            ).fetchone()
            if not row:
                raise KeyError("找不到模擬委託")
            symbol, side, quantity, limit_price, status = row
            if status != OrderStatus.ACCEPTED.value:
                raise ValueError("只有等待中的模擬委託可以撮合")
            crossed = market <= limit_price if side == Side.BUY.value else market >= limit_price
            if not crossed:
                raise ValueError("市場價格尚未觸及限價，委託維持等待")
            # A last-trade/one-minute close only proves that the limit was touched.
            # Queue position and executable depth are unavailable, so fill at the
            # submitted limit (conservative for both sides), never at a better
            # observed last-trade price.
            fill_price = float(limit_price)
            gross = fill_price * quantity
            fees = gross * (buy_fee_rate if side == Side.BUY.value else sell_fee_rate)
            if side == Side.SELL.value:
                fees += gross * sell_tax_rate
            stamp = (now or datetime.now(TZ)).astimezone(TZ).isoformat()
            conn.execute("""
                UPDATE paper_orders SET status=?,message=?,filled_quantity=?,
                    average_fill_price=?,fees=?,updated_at=? WHERE order_id=?
            """, (OrderStatus.FILLED.value, "報價觸及限價後，以委託限價保守模擬成交；非實際成交回報",
                  quantity, fill_price, fees, stamp, order_id))
            conn.commit()
        result = self.get_order(order_id)
        return result

    def cancel_order(self, order_id: str, now: datetime | None = None) -> OrderResult:
        stamp = (now or datetime.now(TZ)).astimezone(TZ).isoformat()
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                "UPDATE paper_orders SET status=?,message=?,updated_at=? "
                "WHERE order_id=? AND status=?",
                (OrderStatus.CANCELLED.value, "模擬委託已取消", stamp,
                 order_id, OrderStatus.ACCEPTED.value),
            )
            conn.commit()
            if cursor.rowcount == 0:
                exists = conn.execute(
                    "SELECT status FROM paper_orders WHERE order_id=?", (order_id,)
                ).fetchone()
                if not exists:
                    raise KeyError("找不到模擬委託")
                raise ValueError("只有等待中的模擬委託可以取消")
        return self.get_order(order_id)

    def get_order(self, order_id: str) -> OrderResult:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT order_id,symbol,side,quantity,limit_price,status,message,"
                "filled_quantity,average_fill_price,fees,created_at "
                "FROM paper_orders WHERE order_id=?", (order_id,)
            ).fetchone()
        if not row:
            raise KeyError("找不到模擬委託")
        return OrderResult(*row[:7], *row[7:10], created_at=row[10])

    def expire_day_orders(self, now: datetime | None = None) -> int:
        """Expire accepted day orders created before the current Taipei date."""
        current = now or datetime.now(TZ)
        if current.tzinfo is None:
            current = current.replace(tzinfo=TZ)
        today = current.astimezone(TZ).date().isoformat()
        stamp = current.astimezone(TZ).isoformat()
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                "UPDATE paper_orders SET status=?,message=?,updated_at=? "
                "WHERE status=? AND substr(created_at,1,10)<?",
                (OrderStatus.EXPIRED.value, "當日有效委託已於交易日結束後到期",
                 stamp, OrderStatus.ACCEPTED.value, today),
            )
            conn.commit()
            return int(cursor.rowcount)

    def list_orders(self, limit: int = 200) -> list[dict]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT order_id,symbol,side,quantity,limit_price,status,message,"
                "filled_quantity,average_fill_price,fees,reason,created_at "
                "FROM paper_orders ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        fields = ["order_id","symbol","side","quantity","limit_price","status",
                  "message","filled_quantity","average_fill_price","fees","reason","created_at"]
        return [dict(zip(fields, row)) for row in rows]
