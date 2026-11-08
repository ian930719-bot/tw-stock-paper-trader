"""Persistent SQLite paper account ledger; no broker connectivity or live orders."""
from __future__ import annotations

from datetime import datetime, timedelta
import sqlite3
import math
from contextlib import closing
from pathlib import Path
from zoneinfo import ZoneInfo

from execution_engine import OrderStatus, Side, TZ, taiwan_regular_session
from market_trading_calendar import market_trading_day_status


class PaperAccount:
    """Atomic paper fills and account ledger sharing execution_engine's database."""

    def __init__(self, db_path: str, initial_cash: float = 1_000_000, *, calendar_checker=None):
        if initial_cash <= 0:
            raise ValueError("初始資金必須大於零")
        self.db_path = str(db_path)
        self.calendar_checker = calendar_checker
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS paper_account (
                id INTEGER PRIMARY KEY CHECK(id=1),
                initial_cash REAL NOT NULL,
                cash REAL NOT NULL,
                created_at TEXT NOT NULL
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS paper_positions (
                symbol TEXT PRIMARY KEY,
                quantity INTEGER NOT NULL CHECK(quantity >= 0),
                average_cost REAL NOT NULL CHECK(average_cost >= 0),
                position_version INTEGER NOT NULL DEFAULT 1
            )""")
            position_columns = {row[1] for row in conn.execute("PRAGMA table_info(paper_positions)")}
            if "position_version" not in position_columns:
                conn.execute("ALTER TABLE paper_positions ADD COLUMN position_version INTEGER NOT NULL DEFAULT 1")
            conn.execute("""CREATE TABLE IF NOT EXISTS paper_fills (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL UNIQUE,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                gross REAL NOT NULL,
                fees REAL NOT NULL,
                tax REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                quote_source TEXT NOT NULL,
                quote_time TEXT NOT NULL,
                filled_at TEXT NOT NULL
            )""")
            conn.execute(
                "INSERT OR IGNORE INTO paper_account(id,initial_cash,cash,created_at) VALUES(1,?,?,?)",
                (float(initial_cash), float(initial_cash), datetime.now(TZ).isoformat()),
            )
            # Apply a changed starting-capital setting only to a genuinely unused
            # paper account. Never overwrite balances after any order, fill or position.
            existing_fills = conn.execute("SELECT COUNT(*) FROM paper_fills").fetchone()[0]
            existing_positions = conn.execute(
                "SELECT COUNT(*) FROM paper_positions WHERE quantity > 0"
            ).fetchone()[0]
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            existing_orders = 0
            if "paper_orders" in tables:
                existing_orders = conn.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0]
            if not existing_fills and not existing_positions and not existing_orders:
                conn.execute(
                    "UPDATE paper_account SET initial_cash=?,cash=? WHERE id=1",
                    (float(initial_cash), float(initial_cash)),
                )
            conn.commit()

    def snapshot(self) -> dict:
        with closing(sqlite3.connect(self.db_path)) as conn:
            account = conn.execute(
                "SELECT initial_cash,cash,created_at FROM paper_account WHERE id=1"
            ).fetchone()
            positions = conn.execute(
                "SELECT symbol,quantity,average_cost,position_version FROM paper_positions WHERE quantity>0"
            ).fetchall()
            fills = conn.execute(
                "SELECT fill_id,order_id,symbol,side,quantity,price,gross,fees,tax,realized_pnl,quote_source,quote_time,filled_at "
                "FROM paper_fills ORDER BY filled_at DESC"
            ).fetchall()
        fields = ["fill_id","order_id","symbol","side","quantity","price","gross","fees","tax","realized_pnl","quote_source","quote_time","filled_at"]
        return {
            "initial_cash": account[0], "cash": account[1], "created_at": account[2],
            "positions": [
                {"symbol": r[0], "quantity": r[1], "average_cost": r[2], "position_version": r[3]}
                for r in positions
            ],
            "fills": [dict(zip(fields, row)) for row in fills],
        }

    def integrity_report(self) -> dict:
        """Reconcile the SQLite cash/position ledger against immutable fill records."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            account = conn.execute(
                "SELECT initial_cash,cash FROM paper_account WHERE id=1"
            ).fetchone()
            if account is None:
                return {"ok": False, "status": "異常", "issues": ["找不到模擬帳戶主檔"]}
            initial_cash, actual_cash = map(float, account)
            if not math.isfinite(initial_cash) or not math.isfinite(actual_cash):
                return {"ok": False, "status": "異常", "issues": ["帳戶現金含非有限數值"]}
            fills = conn.execute(
                "SELECT symbol,side,quantity,price,gross,fees,tax FROM paper_fills ORDER BY filled_at,fill_id"
            ).fetchall()
            actual_positions = {
                str(symbol): int(quantity)
                for symbol, quantity in conn.execute(
                    "SELECT symbol,quantity FROM paper_positions WHERE quantity>0"
                ).fetchall()
            }
            broken_fill_links = int(conn.execute(
                "SELECT COUNT(*) FROM paper_fills f LEFT JOIN paper_orders o ON o.order_id=f.order_id "
                "WHERE o.order_id IS NULL OR o.status<>? OR o.filled_quantity<>f.quantity",
                (OrderStatus.FILLED.value,),
            ).fetchone()[0])
            missing_fill_rows = int(conn.execute(
                "SELECT COUNT(*) FROM paper_orders o LEFT JOIN paper_fills f ON f.order_id=o.order_id "
                "WHERE o.status=? AND f.order_id IS NULL",
                (OrderStatus.FILLED.value,),
            ).fetchone()[0])

        expected_cash = initial_cash
        expected_positions: dict[str, int] = {}
        for symbol, side, quantity, price, gross, fees, tax in fills:
            quantity = int(quantity)
            amounts = [float(price), float(gross), float(fees), float(tax)]
            if (
                quantity <= 0 or not all(math.isfinite(value) for value in amounts)
                or amounts[0] <= 0 or min(amounts[1:]) < 0
                or abs(amounts[1] - amounts[0] * quantity) > 0.02
            ):
                return {"ok": False, "status": "異常", "issues": ["成交紀錄含不合法數量、價格或金額"]}
            if side == Side.BUY.value:
                expected_cash -= float(gross) + float(fees)
                expected_positions[str(symbol)] = expected_positions.get(str(symbol), 0) + quantity
            elif side == Side.SELL.value:
                expected_cash += float(gross) - float(fees) - float(tax)
                expected_positions[str(symbol)] = expected_positions.get(str(symbol), 0) - quantity
            else:
                return {"ok": False, "status": "異常", "issues": [f"未知成交方向：{side}"]}

        expected_positions = {k: v for k, v in expected_positions.items() if v != 0}
        issues = []
        if integrity != "ok":
            issues.append(f"SQLite 完整性檢查失敗：{integrity}")
        cash_difference = actual_cash - expected_cash
        if abs(cash_difference) > 0.01:
            issues.append(f"現金帳差異 NT${cash_difference:,.2f}")
        if expected_positions != actual_positions:
            issues.append("持股數量與成交紀錄不一致")
        if broken_fill_links:
            issues.append(f"成交與委託狀態不一致：{broken_fill_links} 筆")
        if missing_fill_rows:
            issues.append(f"已成交委託缺少成交帳：{missing_fill_rows} 筆")
        return {
            "ok": not issues,
            "status": "通過" if not issues else "異常",
            "issues": issues,
            "sqlite_integrity": integrity,
            "actual_cash": round(actual_cash, 2),
            "expected_cash": round(expected_cash, 2),
            "cash_difference": round(cash_difference, 2),
            "actual_positions": actual_positions,
            "expected_positions": expected_positions,
            "fill_count": len(fills),
        }

    def process_quote(self, symbol: str, price: float, quote_time: str, source: str,
                      *, now: datetime | None = None, max_age_seconds: int = 120, market: str = "上市",
                      buy_fee_rate: float = 0.001425, sell_fee_rate: float = 0.001425,
                      sell_tax_rate: float = 0.003) -> list[dict]:
        """Fill crossed limit orders only from fresh, sourced quotes during regular hours."""
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
            raise ValueError("報價時間格式無效，拒絕撮合")
        age = (current - quoted).total_seconds()
        if not source or age < -5 or age > max_age_seconds:
            raise ValueError("報價來源缺失或報價已過期，拒絕撮合")
        if not math.isfinite(float(price)) or float(price) <= 0:
            raise ValueError("報價必須是有限正數")
        if isinstance(max_age_seconds, bool) or int(max_age_seconds) < 0:
            raise ValueError("報價有效秒數設定不合法")
        clock = current.time().replace(tzinfo=None)
        if current.weekday() >= 5 or not (datetime.strptime("09:00", "%H:%M").time() <= clock <= datetime.strptime("13:30", "%H:%M").time()):
            raise ValueError("非台股盤中交易時段，拒絕撮合")
        if self.calendar_checker is None:
            market_open, calendar_note = market_trading_day_status(current.date(), market)
        else:
            market_open, calendar_note = self.calendar_checker(current.date())
        if not market_open:
            raise ValueError(f"官方交易日曆確認休市或無法確認，拒絕撮合：{calendar_note}")

        # Verify the exchange calendar again before opening the fill transaction.
        # The price band is refreshed separately for each order immediately before
        # the account cash/position mutation.
        if self.calendar_checker is None:
            market_open, calendar_note = market_trading_day_status(current.date(), market)
        else:
            market_open, calendar_note = self.calendar_checker(current.date())
        if not market_open:
            raise ValueError(f"官方交易日曆再次確認失敗，拒絕撮合：{calendar_note}")
        from taiwan_price_limits import get_official_price_limit, validate_limit_price
        completed = []
        with closing(sqlite3.connect(self.db_path, timeout=15)) as conn:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                orders = conn.execute(
                    "SELECT order_id,side,quantity,limit_price,odd_lot FROM paper_orders "
                    "WHERE UPPER(symbol)=? AND status=? ORDER BY created_at",
                    (symbol.upper(), OrderStatus.ACCEPTED.value),
                ).fetchall()
                for order_id, side, quantity, limit_price, odd_lot in orders:
                    if (odd_lot and clock > datetime.strptime("13:30", "%H:%M").time()) or (not odd_lot and clock > datetime.strptime("13:25", "%H:%M").time()):
                        continue
                    crossed = price <= limit_price if side == Side.BUY.value else price >= limit_price
                    if not crossed:
                        continue
                    account = conn.execute(
                        "SELECT cash FROM paper_account WHERE id=1"
                    ).fetchone()
                    if not account:
                        raise ValueError("模擬帳戶尚未初始化")
                    cash = float(account[0])
                    pos = conn.execute(
                        "SELECT quantity,average_cost FROM paper_positions WHERE symbol=?",
                        (symbol.upper(),),
                    ).fetchone()
                    held, avg_cost = (int(pos[0]), float(pos[1])) if pos else (0, 0.0)
                    # Recheck each accepted order immediately before applying account
                    # cash/position changes; never let stale acceptance bypass today's band.
                    try:
                        band = get_official_price_limit(
                            symbol.upper(), market, expected_date=current.date(), force_refresh=True
                        )
                        validate_limit_price(float(limit_price), band)
                    except Exception as exc:
                        conn.execute(
                            "UPDATE paper_orders SET status=?,message=?,updated_at=? WHERE order_id=? AND status=?",
                            (OrderStatus.REJECTED.value,
                             f"成交前官方漲跌停驗證失敗，拒絕撮合：{exc}",
                             current.isoformat(), order_id, OrderStatus.ACCEPTED.value),
                        )
                        continue
                    # Last-trade data confirms only that a limit was touched; it does not
                    # reveal queue priority or executable depth. Use the submitted
                    # limit as the conservative simulated fill price.
                    fill_price = float(limit_price)
                    gross = fill_price * int(quantity)
                    fee_rate = buy_fee_rate if side == Side.BUY.value else sell_fee_rate
                    fees = gross * fee_rate
                    tax = gross * sell_tax_rate if side == Side.SELL.value else 0.0
                    realized = 0.0
                    if side == Side.BUY.value:
                        debit = gross + fees
                        if debit > cash:
                            conn.execute(
                                "UPDATE paper_orders SET status=?,message=?,updated_at=? WHERE order_id=? AND status=?",
                                (OrderStatus.REJECTED.value, "成交時可用現金不足，拒絕撮合", current.isoformat(), order_id, OrderStatus.ACCEPTED.value),
                            )
                            continue
                        new_qty = held + int(quantity)
                        new_avg = ((held * avg_cost) + gross + fees) / new_qty
                        conn.execute(
                            "INSERT INTO paper_positions(symbol,quantity,average_cost,position_version) VALUES(?,?,?,1) "
                            "ON CONFLICT(symbol) DO UPDATE SET quantity=excluded.quantity,average_cost=excluded.average_cost,position_version=paper_positions.position_version+1",
                            (symbol.upper(), new_qty, new_avg),
                        )
                        conn.execute("UPDATE paper_account SET cash=cash-? WHERE id=1", (debit,))
                    else:
                        if int(quantity) > held:
                            conn.execute(
                                "UPDATE paper_orders SET status=?,message=?,updated_at=? WHERE order_id=? AND status=?",
                                (OrderStatus.REJECTED.value, "成交時可用持股不足，拒絕撮合", current.isoformat(), order_id, OrderStatus.ACCEPTED.value),
                            )
                            continue
                        realized = gross - fees - tax - avg_cost * int(quantity)
                        new_qty = held - int(quantity)
                        conn.execute(
                            "UPDATE paper_positions SET quantity=?,average_cost=?,position_version=position_version+1 WHERE symbol=?",
                            (new_qty, avg_cost if new_qty else 0.0, symbol.upper()),
                        )
                        conn.execute("UPDATE paper_account SET cash=cash+? WHERE id=1", (gross - fees - tax,))
                    stamp = current.isoformat()
                    conn.execute(
                        "INSERT INTO paper_fills VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (order_id, order_id, symbol.upper(), side, int(quantity), fill_price,
                         gross, fees, tax, realized, source, quoted.isoformat(), stamp),
                    )
                    conn.execute(
                        "UPDATE paper_orders SET status=?,message=?,filled_quantity=?,average_fill_price=?,fees=?,updated_at=? "
                        "WHERE order_id=? AND status=?",
                        (OrderStatus.FILLED.value, "報價觸及限價後，以委託限價保守模擬成交；非實際成交回報",
                         int(quantity), fill_price, fees + tax, stamp, order_id, OrderStatus.ACCEPTED.value),
                    )
                    completed.append({"order_id": order_id, "symbol": symbol.upper(),
                                      "side": side, "quantity": int(quantity), "price": fill_price,
                                      "fees": fees, "tax": tax, "realized_pnl": realized})
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return completed
