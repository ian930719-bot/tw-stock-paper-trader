"""Persistent SQLite paper account ledger; no broker connectivity or live orders."""
from __future__ import annotations

from datetime import datetime, timedelta
import sqlite3
import math
from contextlib import closing
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo

from execution_engine import OrderStatus, Side, TZ, taiwan_regular_session
from market_trading_calendar import market_trading_day_status
from corporate_actions import validate_action_frame


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
            conn.execute("""CREATE TABLE IF NOT EXISTS paper_corporate_actions (
                event_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                action_type TEXT NOT NULL,
                effective_date TEXT NOT NULL,
                payment_date TEXT,
                share_multiplier REAL NOT NULL,
                cash_per_share REAL NOT NULL,
                reference_price REAL,
                entitled_quantity INTEGER NOT NULL DEFAULT 0,
                cash_amount REAL NOT NULL DEFAULT 0,
                cash_in_lieu REAL NOT NULL DEFAULT 0,
                applied_at TEXT NOT NULL,
                paid_at TEXT
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
            actions = conn.execute(
                "SELECT event_id,symbol,action_type,effective_date,payment_date,entitled_quantity,"
                "cash_amount,cash_in_lieu,applied_at,paid_at FROM paper_corporate_actions "
                "ORDER BY effective_date DESC,event_id"
            ).fetchall()
        fields = ["fill_id","order_id","symbol","side","quantity","price","gross","fees","tax","realized_pnl","quote_source","quote_time","filled_at"]
        return {
            "initial_cash": account[0], "cash": account[1], "created_at": account[2],
            "positions": [
                {"symbol": r[0], "quantity": r[1], "average_cost": r[2], "position_version": r[3]}
                for r in positions
            ],
            "fills": [dict(zip(fields, row)) for row in fills],
            "corporate_actions": [
                dict(zip(["event_id","symbol","action_type","effective_date","payment_date",
                          "entitled_quantity","cash_amount","cash_in_lieu","applied_at","paid_at"], row))
                for row in actions
            ],
        }

    def backup_bytes(self) -> bytes:
        """Return a transactionally consistent SQLite snapshot for offline backup."""
        with closing(sqlite3.connect(self.db_path, timeout=20)) as source:
            with closing(sqlite3.connect(":memory:")) as target:
                source.backup(target)
                return target.serialize()

    def position_start_date(self, symbol: str) -> str | None:
        """Return the start date of the currently open fill-derived position."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            fills = conn.execute(
                "SELECT side,quantity,filled_at FROM paper_fills WHERE symbol=? ORDER BY filled_at,fill_id",
                (str(symbol),),
            ).fetchall()
        quantity = 0
        opened_at = None
        for side, amount, filled_at in fills:
            amount = int(amount)
            if side == Side.BUY.value:
                if quantity <= 0:
                    opened_at = str(filled_at)
                quantity += amount
            elif side == Side.SELL.value:
                quantity -= amount
                if quantity <= 0:
                    quantity = 0
                    opened_at = None
        return opened_at

    def apply_corporate_actions(self, events, as_of: datetime | str) -> dict:
        """Apply ex-date actions exactly once and settle dividend receivables.

        This method only applies actions effective on the supplied Taiwan date.
        An unapplied older event is reported as missed; it is never silently
        applied late because current holdings may have changed since ex-date.
        """
        current = as_of if isinstance(as_of, datetime) else datetime.fromisoformat(str(as_of))
        action_day = current.date().isoformat()
        frame = validate_action_frame(
            events if isinstance(events, pd.DataFrame) else pd.DataFrame(events),
            "",
        )
        applied, paid, missed = [], [], []
        stamp = datetime.now(TZ).isoformat()
        with closing(sqlite3.connect(self.db_path, timeout=20)) as conn:
            conn.execute("PRAGMA busy_timeout=20000")
            # A missed ex-date must be reconciled manually; do not infer old entitlements.
            for event in frame.itertuples(index=False):
                effective = event.effective_date.date().isoformat()
                exists = conn.execute(
                    "SELECT 1 FROM paper_corporate_actions WHERE event_id=?",
                    (str(event.event_id),),
                ).fetchone()
                if exists:
                    continue
                if effective < action_day:
                    missed.append(str(event.event_id))
                    continue
                if effective > action_day:
                    continue
                pending = conn.execute(
                    "SELECT COUNT(*) FROM paper_orders WHERE symbol=? AND status=?",
                    (str(event.symbol), OrderStatus.ACCEPTED.value),
                ).fetchone()[0]
                if pending:
                    raise ValueError(
                        f"{event.symbol} 除權息日仍有未成交委託；請先取消委託再處理公司行動"
                    )
                position = conn.execute(
                    "SELECT quantity,average_cost FROM paper_positions WHERE symbol=?",
                    (str(event.symbol),),
                ).fetchone()
                quantity = int(position[0]) if position else 0
                average_cost = float(position[1]) if position else 0.0
                cash_amount = 0.0
                cash_in_lieu = 0.0
                new_quantity = quantity
                if event.action_type == "cash_dividend":
                    cash_amount = quantity * float(event.cash_per_share)
                elif quantity > 0:
                    multiplier = float(event.share_multiplier)
                    exact = quantity * multiplier
                    new_quantity = int(math.floor(exact + 1e-9))
                    fractional = max(0.0, exact - new_quantity)
                    if fractional > 1e-8:
                        reference = event.reference_price
                        if reference is None or not math.isfinite(float(reference)) or float(reference) <= 0:
                            raise ValueError(
                                f"{event.symbol} 公司行動產生零股權利，但缺少可用參考價；停止自動調整"
                            )
                        cash_in_lieu = fractional * float(reference)
                    if position:
                        conn.execute(
                            "UPDATE paper_positions SET quantity=?,average_cost=?,position_version=position_version+1 WHERE symbol=?",
                            (new_quantity, average_cost / multiplier, str(event.symbol)),
                        )
                if cash_in_lieu:
                    conn.execute(
                        "UPDATE paper_account SET cash=cash+? WHERE id=1",
                        (cash_in_lieu,),
                    )
                conn.execute("""INSERT INTO paper_corporate_actions(
                    event_id,symbol,action_type,effective_date,payment_date,
                    share_multiplier,cash_per_share,reference_price,entitled_quantity,
                    cash_amount,cash_in_lieu,applied_at,paid_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    str(event.event_id), str(event.symbol), str(event.action_type),
                    effective, event.payment_date.date().isoformat() if not pd.isna(event.payment_date) else None,
                    float(event.share_multiplier), float(event.cash_per_share),
                    float(event.reference_price) if event.reference_price is not None and math.isfinite(float(event.reference_price)) else None,
                    quantity, cash_amount, cash_in_lieu, stamp, None,
                ))
                applied.append(str(event.event_id))

            due = conn.execute(
                "SELECT event_id,cash_amount FROM paper_corporate_actions "
                "WHERE action_type='cash_dividend' AND cash_amount>0 AND paid_at IS NULL "
                "AND payment_date IS NOT NULL AND payment_date<=?",
                (action_day,),
            ).fetchall()
            total_paid = sum(float(row[1]) for row in due)
            if total_paid:
                conn.execute("UPDATE paper_account SET cash=cash+? WHERE id=1", (total_paid,))
                conn.executemany(
                    "UPDATE paper_corporate_actions SET paid_at=? WHERE event_id=?",
                    [(stamp, row[0]) for row in due],
                )
                paid = [row[0] for row in due]
            conn.commit()
        return {
            "applied": applied, "paid": paid, "missed": missed,
            "cash_dividends_paid": round(total_paid, 2),
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
                "SELECT symbol,side,quantity,price,gross,fees,tax,filled_at FROM paper_fills ORDER BY filled_at,fill_id"
            ).fetchall()
            action_rows = conn.execute(
                "SELECT symbol,action_type,effective_date,share_multiplier,entitled_quantity,"
                "cash_amount,cash_in_lieu,paid_at FROM paper_corporate_actions "
                "ORDER BY effective_date,applied_at,event_id"
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
        timeline = []
        for symbol, side, quantity, price, gross, fees, tax, filled_at in fills:
            quantity = int(quantity)
            amounts = [float(price), float(gross), float(fees), float(tax)]
            if (
                quantity <= 0 or not all(math.isfinite(value) for value in amounts)
                or amounts[0] <= 0 or min(amounts[1:]) < 0
                or abs(amounts[1] - amounts[0] * quantity) > 0.02
            ):
                return {"ok": False, "status": "異常", "issues": ["成交紀錄含不合法數量、價格或金額"]}
            timeline.append((datetime.fromisoformat(str(filled_at)), "fill", (
                str(symbol), side, quantity, float(gross), float(fees), float(tax)
            )))
        for symbol, action_type, effective_date, multiplier, entitled_quantity, cash_amount, cash_in_lieu, paid_at in action_rows:
            amounts = [float(multiplier), float(cash_amount), float(cash_in_lieu)]
            if not all(math.isfinite(x) for x in amounts) or amounts[0] <= 0 or min(amounts[1:]) < 0:
                return {"ok": False, "status": "異常", "issues": ["公司行動帳含不合法比例或金額"]}
            effective_stamp = datetime.fromisoformat(str(effective_date) + "T00:00:00+08:00")
            timeline.append((effective_stamp, "action", (
                str(symbol), str(action_type), amounts[0], int(entitled_quantity),
                amounts[1], amounts[2],
            )))
            if str(action_type) == "cash_dividend" and paid_at:
                timeline.append((datetime.fromisoformat(str(paid_at)), "dividend_payment", (
                    str(symbol), amounts[1],
                )))
        timeline.sort(key=lambda item: item[0])
        for _, event_type, payload in timeline:
            if event_type == "fill":
                symbol, side, quantity, gross, fees, tax = payload
                if side == Side.BUY.value:
                    expected_cash -= gross + fees
                    expected_positions[symbol] = expected_positions.get(symbol, 0) + quantity
                elif side == Side.SELL.value:
                    expected_cash += gross - fees - tax
                    expected_positions[symbol] = expected_positions.get(symbol, 0) - quantity
                else:
                    return {"ok": False, "status": "異常", "issues": [f"未知成交方向：{side}"]}
            elif event_type == "action":
                symbol, action_type, multiplier, entitled_quantity, cash_amount, cash_in_lieu = payload
                if action_type in {"stock_dividend", "stock_split", "capital_reduction", "par_value_change"}:
                    current_quantity = expected_positions.get(symbol, 0)
                    expected_positions[symbol] = int(math.floor(current_quantity * multiplier + 1e-9))
                expected_cash += cash_in_lieu
            elif event_type == "dividend_payment":
                expected_cash += float(payload[1])

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
