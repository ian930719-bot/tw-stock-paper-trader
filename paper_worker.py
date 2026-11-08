"""Standalone, persistent Taiwan-stock paper-trading worker.

This process is deliberately paper-only. It has no broker adapter and cannot
route real orders. Entries are disabled unless PAPER_AUTO_ENTRIES=1.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timedelta, time
from decimal import Decimal, ROUND_DOWN
import json
import logging
import os
from pathlib import Path
import sqlite3
import time as sleep_time
import uuid
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from adaptive_exit_monitor import AdaptiveExitMonitor
from execution_engine import (
    OrderIntent, PaperExecutionEngine, Side, TZ, taiwan_regular_session,
)
from historical_store import (
    get_incremental_history, read_daily_bars,
)
from market_trading_calendar import market_trading_day_status
from paper_account import PaperAccount
from research_engine import STRATEGIES, strategy_signals
from strategy_exits import initial_stop, risk_position_size
from taiwan_market_rules import floor_board_lot, stock_tick_size
from taiwan_price_limits import get_official_price_limit
from trading_config import TradingConfig
from corporate_actions import fetch_symbol_corporate_actions

LOG = logging.getLogger("paper_worker")
QUOTE_SOURCE = "Yahoo Finance 1-minute"
DEFAULT_DB = str(Path("data") / "trading.sqlite3")
DEFAULT_MARKET_DB = str(Path("data") / "market.sqlite3")
MAX_WATCHLIST = 30
QUOTE_MAX_AGE_SECONDS = 60
TAIPEI = ZoneInfo("Asia/Taipei")


def parse_watchlist(value: str) -> dict[str, str]:
    """Parse '2330:上市,6488:上櫃'; reject ambiguous or oversized lists."""
    result: dict[str, str] = {}
    if not str(value or "").strip():
        return result
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"觀察清單格式錯誤：{item}；應為代號:上市 或 代號:上櫃")
        symbol, market = (part.strip() for part in item.split(":", 1))
        symbol = symbol.upper()
        if not symbol.isalnum() or not 4 <= len(symbol) <= 8:
            raise ValueError(f"不合法的證券代號：{symbol}")
        if market not in {"上市", "上櫃"}:
            raise ValueError(f"{symbol} 的市場別必須是上市或上櫃")
        if symbol in result:
            raise ValueError(f"觀察清單重複代號：{symbol}")
        result[symbol] = market
    if len(result) > MAX_WATCHLIST:
        raise ValueError(f"單一 Worker 最多監控 {MAX_WATCHLIST} 檔，避免行情 API 超量")
    return result


def parse_timestamp(value, *, default_tz=TAIPEI) -> datetime:
    if isinstance(value, datetime):
        stamp = value
    else:
        stamp = datetime.fromisoformat(str(value))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=default_tz)
    return stamp.astimezone(TAIPEI)


def quote_is_fresh(quote: dict, now: datetime, max_age_seconds: int = QUOTE_MAX_AGE_SECONDS) -> bool:
    try:
        price = float(quote["price"])
        stamp = parse_timestamp(quote["timestamp"])
        current = parse_timestamp(now)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    age = (current - stamp).total_seconds()
    return bool(
        np.isfinite(price) and price > 0
        and quote.get("source")
        and -5 <= age <= max_age_seconds
    )


def round_price_down(price: float, instrument_type: str = "stock") -> float:
    value = Decimal(str(price))
    tick = stock_tick_size(value, instrument_type)
    rounded = (value / tick).to_integral_value(rounding=ROUND_DOWN) * tick
    if rounded <= 0:
        raise ValueError("價格依台股升降單位調整後不合法")
    return float(rounded)


def calculate_atr(frame: pd.DataFrame, period: int = 14) -> float:
    if frame is None or len(frame) < period + 1:
        raise ValueError(f"至少需要 {period + 1} 根已完成日線計算 ATR")
    z = frame.sort_values("date").reset_index(drop=True)
    previous_close = pd.to_numeric(z["close"], errors="coerce").shift(1)
    true_range = pd.concat([
        pd.to_numeric(z["high"], errors="coerce") - pd.to_numeric(z["low"], errors="coerce"),
        (pd.to_numeric(z["high"], errors="coerce") - previous_close).abs(),
        (pd.to_numeric(z["low"], errors="coerce") - previous_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().iloc[-1]
    if not np.isfinite(float(atr)) or float(atr) <= 0:
        raise ValueError("ATR 計算結果無效")
    return float(atr)


def completed_daily_bars(frame: pd.DataFrame, today) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    z = frame.copy()
    z["date"] = pd.to_datetime(z["date"], errors="coerce")
    z = z.dropna(subset=["date"])
    z = z[z["date"].dt.date < today].sort_values("date").reset_index(drop=True)
    return z


def max_shares_for_participation(frame: pd.DataFrame, price: float, rate: float = 0.01) -> int:
    """Cap an order using the last 20 completed daily bars' average dollar turnover."""
    if not np.isfinite(float(price)) or float(price) <= 0:
        return 0
    if not np.isfinite(float(rate)) or not 0 < float(rate) <= 0.10:
        raise ValueError("參與率必須介於 0 與 10%")
    if frame is None or frame.empty or not {"close", "volume"}.issubset(frame.columns):
        return 0
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    turnover = (close * volume).replace([np.inf, -np.inf], np.nan).dropna().tail(20)
    if len(turnover) < 20:
        return 0
    adv20 = float(turnover.mean())
    return max(0, int(adv20 * float(rate) / float(price)))


def rank_entry_candidates(frames: dict[str, pd.DataFrame], today) -> list[dict]:
    """Use completed daily bars only; rank by strategy agreement then momentum."""
    candidates = []
    for symbol, raw in frames.items():
        z = completed_daily_bars(raw, today)
        if len(z) < 260:
            continue
        signals = strategy_signals(z)
        active = [name for name in STRATEGIES if bool(signals[name][0].iloc[-1])]
        if not active:
            continue
        close = pd.to_numeric(z["close"], errors="coerce")
        momentum = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else -1.0
        if not np.isfinite(momentum):
            continue
        candidates.append({
            "symbol": symbol,
            "signal_date": z["date"].iloc[-1].date().isoformat(),
            "strategies": active,
            "agreement": len(active),
            "momentum_20d": momentum,
            "frame": z,
        })
    return sorted(
        candidates,
        key=lambda item: (item["agreement"], item["momentum_20d"]),
        reverse=True,
    )


def fetch_yahoo_intraday_quote(symbol: str, market: str) -> dict | None:
    suffix = ".TW" if market == "上市" else ".TWO"
    history = yf.Ticker(f"{symbol}{suffix}").history(
        period="1d", interval="1m", auto_adjust=False, prepost=False,
    )
    if history is None or history.empty or "Close" not in history:
        return None
    clean = history.dropna(subset=["Close"])
    if clean.empty:
        return None
    row = clean.iloc[-1]
    stamp = parse_timestamp(clean.index[-1].to_pydatetime())
    return {
        "price": float(row["Close"]),
        "timestamp": stamp.isoformat(),
        "source": QUOTE_SOURCE,
    }


class PaperWorker:
    def __init__(
        self,
        *,
        db_path: str | None = None,
        market_db_path: str | None = None,
        watchlist: dict[str, str] | None = None,
        config: TradingConfig | None = None,
        auto_entries: bool | None = None,
        exit_profile: str | None = None,
        quote_fetcher=None,
        history_loader=None,
        calendar_checker=None,
        price_limit_provider=None,
        now_provider=None,
    ):
        self.db_path = db_path or os.getenv("EXECUTION_DB_PATH", DEFAULT_DB)
        self.market_db_path = market_db_path or os.getenv("MARKET_DB_PATH", DEFAULT_MARKET_DB)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.market_db_path).parent.mkdir(parents=True, exist_ok=True)
        self.config = config or TradingConfig()
        raw_watchlist = os.getenv("PAPER_WATCHLIST", "") if watchlist is None else ""
        self.watchlist = parse_watchlist(raw_watchlist) if watchlist is None else dict(watchlist)
        if len(self.watchlist) > MAX_WATCHLIST:
            raise ValueError(f"觀察清單不可超過 {MAX_WATCHLIST} 檔")
        self.auto_entries = (
            os.getenv("PAPER_AUTO_ENTRIES", "0").strip().lower() in {"1", "true", "yes", "on"}
            if auto_entries is None else bool(auto_entries)
        )
        self.exit_profile = exit_profile or os.getenv("PAPER_EXIT_PROFILE", "atr_dynamic")
        if self.exit_profile not in {"atr_dynamic", "oneil_growth", "minervini_sepa"}:
            raise ValueError("PAPER_EXIT_PROFILE 不支援")
        self.quote_fetcher = quote_fetcher or fetch_yahoo_intraday_quote
        self.history_loader = history_loader or self._load_history
        self.calendar_checker = calendar_checker or market_trading_day_status
        self.price_limit_provider = price_limit_provider or self._price_limit
        self.now_provider = now_provider or (lambda: datetime.now(TZ))
        self.owner = str(uuid.uuid4())
        self.engine = PaperExecutionEngine(
            db_path=self.db_path,
            limits=self.config.risk_limits(),
            price_limit_provider=self.price_limit_provider,
            calendar_provider=self.calendar_checker,
        )
        self.account = PaperAccount(self.db_path, initial_cash=self.config.initial_capital)
        self.exit_monitor = AdaptiveExitMonitor(
            self.db_path, max_age_seconds=self.config.max_quote_age_seconds
        )
        self._init_worker_tables()

    def _price_limit(self, symbol: str, market: str):
        return get_official_price_limit(
            symbol, market, expected_date=self.now_provider().astimezone(TZ).date()
        )

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=20)
        conn.execute("PRAGMA busy_timeout=20000")
        return conn

    def _init_worker_tables(self):
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_worker_lock (
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    owner TEXT NOT NULL,
                    lease_until TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_worker_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    level TEXT NOT NULL,
                    event TEXT NOT NULL,
                    details TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_worker_history_refresh (
                    symbol TEXT PRIMARY KEY,
                    attempted_at TEXT NOT NULL,
                    status TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_worker_corporate_action_cache (
                    symbol TEXT PRIMARY KEY,
                    refreshed_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    events_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_worker_last_quote (
                    symbol TEXT PRIMARY KEY,
                    quote_time TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_worker_status (
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    heartbeat_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    last_summary TEXT NOT NULL
                )
            """)
            conn.commit()

    def _event(self, level: str, event: str, **details):
        stamp = self.now_provider().astimezone(TZ).isoformat()
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO paper_worker_events(created_at,level,event,details) VALUES(?,?,?,?)",
                (stamp, level, event, json.dumps(details, ensure_ascii=False, default=str)),
            )
            conn.commit()
        getattr(LOG, level.lower(), LOG.info)("%s %s", event, details)

    def acquire_lease(self, now: datetime | None = None, lease_seconds: int = 600) -> bool:
        current = parse_timestamp(now or self.now_provider())
        expiry = (current + timedelta(seconds=lease_seconds)).isoformat()
        with closing(self._connect()) as conn:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT owner,lease_until FROM paper_worker_lock WHERE id=1"
            ).fetchone()
            if row and row[0] != self.owner and parse_timestamp(row[1]) > current:
                conn.execute("ROLLBACK")
                return False
            conn.execute("""
                INSERT INTO paper_worker_lock(id,owner,lease_until) VALUES(1,?,?)
                ON CONFLICT(id) DO UPDATE SET owner=excluded.owner,lease_until=excluded.lease_until
            """, (self.owner, expiry))
            conn.execute("COMMIT")
        return True

    def renew_lease(self, now: datetime | None = None, lease_seconds: int = 600):
        current = parse_timestamp(now or self.now_provider())
        expiry = (current + timedelta(seconds=lease_seconds)).isoformat()
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "UPDATE paper_worker_lock SET lease_until=? WHERE id=1 AND owner=?",
                (expiry, self.owner),
            )
            conn.commit()
        if cursor.rowcount != 1:
            raise RuntimeError("Worker 執行租約遺失；停止本輪交易")

    def release_lease(self):
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM paper_worker_lock WHERE id=1 AND owner=?", (self.owner,))
            conn.commit()

    def _set_status(self, status: str, summary: dict):
        stamp = self.now_provider().astimezone(TZ).isoformat()
        with closing(self._connect()) as conn:
            conn.execute("""
                INSERT INTO paper_worker_status(id,heartbeat_at,status,last_summary)
                VALUES(1,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    heartbeat_at=excluded.heartbeat_at,
                    status=excluded.status,
                    last_summary=excluded.last_summary
            """, (stamp, status, json.dumps(summary, ensure_ascii=False, default=str)))
            conn.commit()

    def _last_quote_time(self, symbol: str):
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT quote_time FROM paper_worker_last_quote WHERE symbol=?", (symbol,)
            ).fetchone()
        return row[0] if row else None

    def _save_quote_time(self, symbol: str, quote_time: str):
        with closing(self._connect()) as conn:
            conn.execute("""
                INSERT INTO paper_worker_last_quote(symbol,quote_time) VALUES(?,?)
                ON CONFLICT(symbol) DO UPDATE SET quote_time=excluded.quote_time
            """, (symbol, quote_time))
            conn.commit()

    def _fallback_daily(self, symbol: str, period: str) -> pd.DataFrame:
        suffix = ".TW" if self.watchlist.get(symbol, "上市") == "上市" else ".TWO"
        history = yf.Ticker(f"{symbol}{suffix}").history(
            period=period, interval="1d", auto_adjust=False,
        )
        if history is None or history.empty:
            return pd.DataFrame()
        frame = history.reset_index().rename(columns={
            "Date": "date", "Datetime": "date",
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        return frame[["date", "open", "high", "low", "close", "volume"]]

    def _load_history(self, symbol: str, market: str, now: datetime):
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT attempted_at,status FROM paper_worker_history_refresh WHERE symbol=?",
                (symbol,),
            ).fetchone()
        should_refresh = row is None
        if row:
            last_attempt = parse_timestamp(row[0])
            retry_minutes = 30 if "失敗" in str(row[1]) else 360
            should_refresh = now - last_attempt >= timedelta(minutes=retry_minutes)
        status = row[1] if row else "尚未更新"
        if should_refresh:
            try:
                _, info = get_incremental_history(
                    symbol, "5y", token=os.getenv("FINMIND_TOKEN", ""),
                    fallback_fetcher=self._fallback_daily,
                    path=self.market_db_path,
                )
                status = str(info.get("來源狀態") or "歷史行情已更新")
                if info.get("更新錯誤"):
                    status += "；更新失敗：" + str(info["更新錯誤"])
            except Exception as exc:
                status = f"歷史行情更新失敗：{exc}"
            with closing(self._connect()) as conn:
                conn.execute("""
                    INSERT INTO paper_worker_history_refresh(symbol,attempted_at,status)
                    VALUES(?,?,?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        attempted_at=excluded.attempted_at,status=excluded.status
                """, (symbol, now.isoformat(), status))
                conn.commit()
        frame = read_daily_bars(symbol, path=self.market_db_path)
        return frame, status

    def _account_context(self, quotes: dict[str, dict], now: datetime):
        snapshot = self.account.snapshot()
        orders = self.engine.list_orders(limit=1000)
        today = now.date().isoformat()
        pending = [item for item in orders if item["status"] == "ACCEPTED"]
        reserved_buy_cash = sum(
            float(item["quantity"]) * float(item["limit_price"]) * 1.001425
            for item in pending if item["side"] == "BUY"
        )
        positions = {item["symbol"]: int(item["quantity"]) for item in snapshot["positions"]}
        for item in pending:
            if item["side"] == "SELL":
                positions[item["symbol"]] = max(
                    0, positions.get(item["symbol"], 0) - int(item["quantity"])
                )
        marked_value = sum(
            int(item["quantity"]) * float(
                quotes.get(item["symbol"], {}).get("price") or item["average_cost"]
            )
            for item in snapshot["positions"]
        )
        dividend_receivables = sum(
            float(item.get("cash_amount", 0.0))
            for item in snapshot.get("corporate_actions", [])
            if item.get("action_type") == "cash_dividend" and not item.get("paid_at")
        )
        equity = float(snapshot["cash"]) + marked_value + dividend_receivables
        daily_pnl = sum(
            float(fill["realized_pnl"])
            for fill in snapshot["fills"]
            if str(fill["filled_at"]).startswith(today)
        )
        daily_orders = sum(
            1 for item in orders if str(item["created_at"]).startswith(today)
        )
        return {
            "snapshot": snapshot,
            "orders": orders,
            "pending": pending,
            "positions": positions,
            "cash": max(0.0, float(snapshot["cash"]) - reserved_buy_cash),
            "equity": equity,
            "daily_pnl": daily_pnl,
            "open_count": len(pending),
            "daily_order_count": daily_orders,
        }

    def _submit(self, intent: OrderIntent, quotes: dict[str, dict], now: datetime):
        context = self._account_context(quotes, now)
        try:
            result = self.engine.submit(
                intent,
                cash=context["cash"],
                positions=context["positions"],
                equity=context["equity"],
                daily_realized_pnl=context["daily_pnl"],
                open_order_count=context["open_count"],
                daily_order_count=context["daily_order_count"],
                now=now,
            )
        except Exception as exc:
            self._event(
                "error", "模擬委託建立失敗",
                symbol=intent.symbol, side=intent.side.value, reason=str(exc),
            )
            return None
        self._event(
            "info" if result.status == "ACCEPTED" else "warning",
            "模擬委託結果",
            symbol=intent.symbol, side=intent.side.value,
            quantity=intent.quantity, limit_price=intent.limit_price,
            status=result.status, message=result.message, order_id=result.order_id,
        )
        return result

    def _submit_exit(self, signal: dict, position: dict, quote: dict,
                     market: str, now: datetime, quotes: dict[str, dict]):
        symbol = signal["symbol"]
        context = self._account_context(quotes, now)
        if any(o["symbol"] == symbol and o["side"] == "SELL" for o in context["pending"]):
            return None
        instrument = "etf" if symbol in self._etf_symbols() else "stock"
        raw = min(
            int(position["quantity"]),
            int(self.config.max_order_notional / float(quote["price"])),
        )
        quantity = floor_board_lot(raw) if raw >= 1000 else raw
        odd_lot = quantity < 1000
        if quantity <= 0:
            self._event("warning", "動態退出股數低於單筆委託上限", symbol=symbol)
            return None
        limit_price = round_price_down(float(quote["price"]), instrument)
        key = (
            f"worker-exit-{symbol}-{now.date().isoformat()}-"
            f"{signal['quote_time']}-{signal['trigger_price']}"
        )
        intent = OrderIntent(
            symbol=symbol, side=Side.SELL, quantity=quantity,
            limit_price=limit_price, reason=signal["reason"],
            idempotency_key=key, instrument_type=instrument,
            odd_lot=odd_lot, market=market,
        )
        return self._submit(intent, quotes, now)

    def _etf_symbols(self) -> set[str]:
        return {
            item.strip().upper()
            for item in os.getenv("PAPER_ETF_SYMBOLS", "").split(",")
            if item.strip()
        }

    def _manage_positions(self, quotes, frames, market_map, now):
        snapshot = self.account.snapshot()
        self.exit_monitor.clear_absent_positions(
            item["symbol"] for item in snapshot["positions"]
        )
        pending_sells = {
            item["symbol"] for item in self.engine.list_orders(limit=1000)
            if item["status"] == "ACCEPTED" and item["side"] == "SELL"
        }
        results = []
        for position in snapshot["positions"]:
            symbol = position["symbol"]
            quote = quotes.get(symbol)
            market = market_map.get(symbol)
            frame = frames.get(symbol)
            decision_now = parse_timestamp(self.now_provider())
            if (
                quote is None or market is None or frame is None
                or symbol in pending_sells
                or not quote_is_fresh(quote, decision_now, self.config.max_quote_age_seconds)
            ):
                continue
            try:
                z = completed_daily_bars(frame, decision_now.date())
                if len(z) < 16:
                    continue
                atr = calculate_atr(z)
                structure = float(z["low"].shift(1).tail(10).min())
                technical = structure if self.exit_profile == "minervini_sepa" else None
                self.exit_monitor.configure_position(
                    symbol, int(position["quantity"]), float(position["average_cost"]),
                    atr, position_version=int(position["position_version"]),
                    profile=self.exit_profile,
                    atr_multiple=float(os.getenv("PAPER_ATR_MULTIPLE", "2.0")),
                    technical_invalidation=technical, now=decision_now,
                )
                signal = self.exit_monitor.evaluate_quote(
                    symbol, float(quote["price"]), str(quote["timestamp"]),
                    str(quote["source"]), now=decision_now,
                )
                if signal:
                    result = self._submit_exit(
                        signal, position, quote, market, decision_now, quotes
                    )
                    if result:
                        results.append(result.status)
            except (ValueError, KeyError) as exc:
                self._event("warning", "持倉動態風控略過", symbol=symbol, reason=str(exc))
        return results

    def _submit_entries(self, quotes, frames, market_map, now):
        if not self.auto_entries or not taiwan_regular_session(now):
            return []
        candidates = rank_entry_candidates(frames, now.date())
        results = []
        for candidate in candidates:
            symbol = candidate["symbol"]
            quote = quotes.get(symbol)
            market = market_map.get(symbol)
            decision_now = parse_timestamp(self.now_provider())
            if (
                quote is None or market is None
                or not quote_is_fresh(quote, decision_now, self.config.max_quote_age_seconds)
                or not taiwan_regular_session(decision_now)
            ):
                continue
            context = self._account_context(quotes, decision_now)
            if context["open_count"] >= self.config.max_open_orders:
                break
            if context["daily_order_count"] >= self.config.max_order_count_per_day:
                break
            if context["daily_pnl"] <= -abs(self.config.max_daily_loss):
                self._event("warning", "每日虧損停手", pnl=context["daily_pnl"])
                break
            if context["positions"].get(symbol, 0) > 0:
                continue
            if any(o["symbol"] == symbol for o in context["pending"]):
                continue
            frame = candidate["frame"]
            try:
                atr = calculate_atr(frame)
                structure = float(frame["low"].shift(1).tail(10).min())
                instrument = "etf" if symbol in self._etf_symbols() else "stock"
                entry_price = round_price_down(float(quote["price"]), instrument)
                technical = structure if self.exit_profile == "minervini_sepa" else None
                stop = initial_stop(
                    entry_price, self.exit_profile,
                    technical_invalidation=technical, atr=atr,
                    atr_multiple=float(os.getenv("PAPER_ATR_MULTIPLE", "2.0")),
                )["stop_price"]
                risk_fraction = min(
                    self.config.risk_budget_per_trade / max(context["equity"], 1.0),
                    0.10,
                )
                sizing = risk_position_size(
                    context["equity"], risk_fraction, entry_price, stop,
                )
                held_value = 0.0
                snap_pos = next(
                    (p for p in context["snapshot"]["positions"] if p["symbol"] == symbol),
                    None,
                )
                if snap_pos:
                    held_value = int(snap_pos["quantity"]) * float(
                        quote["price"] or snap_pos["average_cost"]
                    )
                invested = sum(
                    int(p["quantity"]) * float(
                        quotes.get(p["symbol"], {}).get("price") or p["average_cost"]
                    ) for p in context["snapshot"]["positions"]
                )
                remaining = min(
                    self.config.max_order_notional,
                    self.config.max_symbol_exposure - held_value,
                    self.config.max_portfolio_exposure - invested,
                    context["cash"] / 1.001425,
                )
                max_by_value = max(0, int(remaining / entry_price))
                max_by_liquidity = max_shares_for_participation(
                    frame, entry_price, self.config.max_participation_rate,
                )
                quantity = min(int(sizing["shares"]), max_by_value, max_by_liquidity)
                if quantity >= 1000:
                    quantity = floor_board_lot(quantity)
                odd_lot = quantity < 1000
                if quantity <= 0:
                    continue
                key = (
                    f"worker-entry-{symbol}-{decision_now.date().isoformat()}-"
                    f"{candidate['signal_date']}-{candidate['agreement']}"
                )
                intent = OrderIntent(
                    symbol=symbol, side=Side.BUY, quantity=quantity,
                    limit_price=entry_price,
                    reason="多策略日線訊號：" + "、".join(candidate["strategies"]),
                    idempotency_key=key, instrument_type=instrument,
                    odd_lot=odd_lot, market=market,
                )
                result = self._submit(intent, quotes, decision_now)
                if result is not None:
                    results.append(result.status)
            except (ValueError, KeyError, TypeError) as exc:
                self._event("warning", "進場候選略過", symbol=symbol, reason=str(exc))
        return results

    def _load_corporate_actions(self, symbol: str, now: datetime) -> pd.DataFrame:
        """Refresh and load point-in-time events for one currently held symbol."""
        position_start = self.account.position_start_date(symbol)
        if not position_start:
            raise RuntimeError(f"{symbol} 持倉缺少可追溯的買進成交紀錄，無法確認公司行動權利")
        start_date = parse_timestamp(position_start).date().isoformat()
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT refreshed_at,status,events_json FROM paper_worker_corporate_action_cache WHERE symbol=?",
                (str(symbol),),
            ).fetchone()
        refresh = row is None
        if row:
            refresh = now - parse_timestamp(row[0]) >= timedelta(hours=6)
        if refresh:
            try:
                end_date = (now.date() + timedelta(days=365)).isoformat()
                events = fetch_symbol_corporate_actions(
                    symbol, start_date, end_date,
                    token=os.getenv("FINMIND_TOKEN", ""),
                )
                payload = events.to_json(orient="records", date_format="iso")
                status = "公司行動事件已更新"
            except Exception as exc:
                raise RuntimeError(f"{symbol} 公司行動資料更新失敗：{exc}") from exc
            with closing(self._connect()) as conn:
                conn.execute("""
                    INSERT INTO paper_worker_corporate_action_cache(symbol,refreshed_at,status,events_json)
                    VALUES(?,?,?,?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        refreshed_at=excluded.refreshed_at,
                        status=excluded.status,
                        events_json=excluded.events_json
                """, (str(symbol), now.isoformat(), status, payload))
                conn.commit()
        else:
            payload = row[2]
        try:
            events = pd.read_json(payload, orient="records")
        except ValueError as exc:
            raise RuntimeError(f"{symbol} 公司行動快取格式錯誤：{exc}") from exc
        if events.empty:
            return events
        events["effective_date"] = pd.to_datetime(events["effective_date"], errors="coerce")
        events = events.loc[
            events["effective_date"].dt.date >= parse_timestamp(position_start).date()
        ].copy()
        return events

    def _process_corporate_actions(self, now: datetime) -> dict:
        """Apply today's events and settle due dividends before matching orders."""
        snapshot = self.account.snapshot()
        held = [str(item["symbol"]) for item in snapshot["positions"]]
        batches = []
        for symbol in held:
            events = self._load_corporate_actions(symbol, now)
            if not events.empty:
                batches.append(events)
        combined = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()
        result = self.account.apply_corporate_actions(combined, now)
        if result["missed"]:
            raise RuntimeError(
                "偵測到未處理的過去除權息事件；為避免錯算持股權利，已停止模擬交易，"
                "需要人工核對帳本。事件：" + "、".join(result["missed"])
            )
        if result["applied"] or result["paid"]:
            self._event(
                "info", "公司行動已入帳",
                applied=result["applied"], paid=result["paid"],
                cash_dividends_paid=result["cash_dividends_paid"],
            )
        return result

    def run_cycle(self, now: datetime | None = None) -> dict:
        current = parse_timestamp(now or self.now_provider())
        self.renew_lease(current)
        self.engine.expire_day_orders(current)
        summary = {
            "time": current.isoformat(), "watchlist": len(self.watchlist),
            "fresh_quotes": 0, "fills": 0, "entries": 0, "exits": 0,
            "auto_entries_enabled": self.auto_entries,
        }
        ledger_check = self.account.integrity_report()
        if not ledger_check["ok"]:
            summary["status"] = "帳本完整性異常，已停止交易"
            summary["ledger_integrity"] = ledger_check
            self._event(
                "error", "帳本完整性檢查失敗，停止所有模擬交易",
                issues=ledger_check["issues"],
            )
            self._set_status("blocked", summary)
            return summary
        if not self.watchlist:
            summary["status"] = "等待設定 PAPER_WATCHLIST"
            self._set_status("idle", summary)
            self._event("warning", "Worker 未啟動交易：觀察清單為空")
            return summary
        try:
            summary["corporate_actions"] = self._process_corporate_actions(current)
        except Exception as exc:
            summary["status"] = "公司行動資料或帳務未確認，已停止交易"
            summary["corporate_action_error"] = str(exc)
            self._event("error", "公司行動驗證失敗，停止所有模擬交易", reason=str(exc))
            self._set_status("blocked", summary)
            return summary
        if current.weekday() >= 5 or not (
            time(9, 0) <= current.time().replace(tzinfo=None) <= time(13, 30)
        ):
            summary["status"] = "非盤中時段"
            self._set_status("sleeping", summary)
            return summary

        market_map = dict(self.watchlist)
        for market in set(market_map.values()):
            try:
                is_open, note = self.calendar_checker(current.date(), market)
            except Exception as exc:
                is_open, note = False, f"官方交易日曆錯誤：{exc}"
            if not is_open:
                self._event("warning", "市場交易日曆未確認，停止該市場委託",
                            market=market, note=note)
                market_map = {
                    symbol: item_market for symbol, item_market in market_map.items()
                    if item_market != market
                }
        if not market_map:
            summary["status"] = "官方交易日曆未確認"
            self._set_status("blocked", summary)
            return summary

        quotes: dict[str, dict] = {}
        for symbol, market in market_map.items():
            self.renew_lease()
            try:
                quote = self.quote_fetcher(symbol, market)
                quote_now = parse_timestamp(self.now_provider())
                if not quote or not quote_is_fresh(
                    quote, quote_now, self.config.max_quote_age_seconds
                ):
                    continue
                stamp = parse_timestamp(quote["timestamp"]).isoformat()
                last = self._last_quote_time(symbol)
                if last and parse_timestamp(stamp) <= parse_timestamp(last):
                    continue
                quote = dict(quote)
                quote["timestamp"] = stamp
                quotes[symbol] = quote
                summary["fresh_quotes"] += 1
            except Exception as exc:
                self._event("warning", "盤中報價取得失敗", symbol=symbol, reason=str(exc))
        if not quotes:
            summary["status"] = "沒有可用的新鮮報價"
            self._set_status("waiting_quote", summary)
            return summary

        # Match previously submitted orders before creating any new order. A quote
        # timestamp is consumed once, so the same minute bar cannot fill a new order.
        for symbol, quote in list(quotes.items()):
            market = market_map[symbol]
            processing_now = parse_timestamp(self.now_provider())
            if not quote_is_fresh(
                quote, processing_now, self.config.max_quote_age_seconds
            ):
                quotes.pop(symbol, None)
                continue
            try:
                fills = self.account.process_quote(
                    symbol, float(quote["price"]), quote["timestamp"],
                    str(quote["source"]), now=processing_now,
                    max_age_seconds=self.config.max_quote_age_seconds,
                    market=market,
                )
                summary["fills"] += len(fills)
                if fills:
                    self._event("info", "模擬委託成交", symbol=symbol, fills=fills)
            except Exception as exc:
                self._event("warning", "模擬撮合略過", symbol=symbol, reason=str(exc))
            self._save_quote_time(symbol, quote["timestamp"])

        frames = {}
        for symbol, market in market_map.items():
            self.renew_lease()
            try:
                frame, status = self.history_loader(symbol, market, current)
                if frame is not None and not frame.empty:
                    frame = completed_daily_bars(frame, current.date())
                    if not frame.empty:
                        latest = frame["date"].max().date()
                        if (current.date() - latest).days <= 7:
                            frames[symbol] = frame
                if symbol not in frames:
                    self._event("warning", "日線資料不足或過期", symbol=symbol, status=status)
            except Exception as exc:
                self._event("warning", "日線歷史載入失敗", symbol=symbol, reason=str(exc))

        decision_now = parse_timestamp(self.now_provider())
        quotes = {
            symbol: quote for symbol, quote in quotes.items()
            if quote_is_fresh(quote, decision_now, self.config.max_quote_age_seconds)
        }
        if not quotes:
            summary["status"] = "日線載入後盤中報價已過期"
            self._set_status("waiting_quote", summary)
            return summary
        exit_results = self._manage_positions(quotes, frames, market_map, decision_now)
        entry_results = self._submit_entries(quotes, frames, market_map, decision_now)
        summary["exits"] = sum(1 for item in exit_results if item == "ACCEPTED")
        summary["entries"] = sum(1 for item in entry_results if item == "ACCEPTED")
        summary["status"] = "執行完成"
        self._set_status("running", summary)
        self._event("info", "Worker 週期完成", **summary)
        return summary

    def run_once(self, now: datetime | None = None) -> dict:
        current = parse_timestamp(now or self.now_provider())
        if not self.acquire_lease(current):
            raise RuntimeError("已有另一個 Paper Worker 持有執行租約")
        try:
            return self.run_cycle(current)
        finally:
            self.release_lease()

    def run_forever(self, interval_seconds: int = 30):
        if interval_seconds < 10:
            raise ValueError("Worker 輪詢間隔不可低於 10 秒")
        if not self.acquire_lease():
            raise RuntimeError("已有另一個 Paper Worker 持有執行租約")
        self._event("info", "Paper Worker 常駐程序啟動",
                    auto_entries=self.auto_entries, watchlist=self.watchlist)
        try:
            while True:
                started = self.now_provider()
                try:
                    self.run_cycle(started)
                except Exception as exc:
                    LOG.exception("Worker 週期發生未預期錯誤")
                    self._event("error", "Worker 週期失敗", reason=str(exc))
                    self._set_status("error", {"error": str(exc)})
                sleep_time.sleep(interval_seconds)
                self.renew_lease()
        except KeyboardInterrupt:
            self._event("info", "Paper Worker 收到停止訊號")
        finally:
            self.release_lease()


def main():
    parser = argparse.ArgumentParser(description="Taiwan stock paper-trading worker (paper only)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="只執行一輪")
    mode.add_argument("--loop", action="store_true", help="常駐輪詢（預設）")
    parser.add_argument(
        "--interval", type=int,
        default=int(os.getenv("PAPER_WORKER_INTERVAL_SECONDS", "30")),
        help="輪詢秒數，最少 10 秒",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    worker = PaperWorker()
    if args.once:
        print(json.dumps(worker.run_once(), ensure_ascii=False, indent=2))
    else:
        worker.run_forever(args.interval)


if __name__ == "__main__":
    main()


def read_worker_status(db_path: str) -> dict | None:
    """Read the latest worker heartbeat from the shared paper ledger."""
    if not db_path or not Path(db_path).exists():
        return None
    try:
        with closing(sqlite3.connect(db_path, timeout=5)) as conn:
            row = conn.execute(
                "SELECT heartbeat_at,status,last_summary FROM paper_worker_status WHERE id=1"
            ).fetchone()
        if not row:
            return None
        return {
            "heartbeat_at": row[0],
            "status": row[1],
            "summary": json.loads(row[2]) if row[2] else {},
        }
    except (sqlite3.Error, ValueError, TypeError):
        return None
