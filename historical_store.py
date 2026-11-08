"""Incremental daily OHLCV storage for selected Taiwan securities.

The SQLite file is a runtime cache, not durable cloud storage. Streamlit
Community Cloud may discard it on restart/redeploy. A durable shared database
requires an external database service and user-provided credentials.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

API_URL = "https://api.finmindtrade.com/api/v4/data"
TZ = ZoneInfo("Asia/Taipei")
COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def database_path() -> str:
    return os.getenv("MARKET_DB_PATH", str(Path(tempfile.gettempdir()) / "tw_stock_market.sqlite3"))


def _connect(path: str | None = None):
    db_path = path or database_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=20)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_bars (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            source TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY(symbol, date)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_bars_symbol_date ON daily_bars(symbol, date)")
    return conn


def normalize_daily_bars(records, symbol: str) -> pd.DataFrame:
    """Validate FinMind/Yahoo-like records and return normalized OHLCV rows."""
    raw = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    aliases = {
        "max": "high", "min": "low", "Trading_Volume": "volume",
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Volume": "volume", "Date": "date", "Datetime": "date",
    }
    raw = raw.rename(columns=aliases)
    missing = set(COLUMNS).difference(raw.columns)
    if missing:
        raise ValueError("日線資料缺少欄位：" + ", ".join(sorted(missing)))
    out = raw[COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in COLUMNS[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=COLUMNS)
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1) & (out["volume"] >= 0)]
    valid_range = (
        (out["high"] >= out[["open", "close", "low"]].max(axis=1))
        & (out["low"] <= out[["open", "close", "high"]].min(axis=1))
    )
    out = out[valid_range].drop_duplicates("date", keep="last").sort_values("date")
    out = out.reset_index(drop=True)
    return out


def fetch_finmind_daily(symbol: str, start_date: str, end_date: str | None = None,
                        token: str = "", timeout: int = 20) -> pd.DataFrame:
    """Fetch one symbol's adjusted daily OHLCV from FinMind."""
    params = {
        "dataset": "TaiwanStockPriceAdj",
        "data_id": str(symbol),
        "start_date": start_date,
    }
    if end_date:
        params["end_date"] = end_date
    if token:
        params["token"] = token
    try:
        response = requests.get(API_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"FinMind 日線請求失敗：{exc}") from exc
    if not isinstance(payload, dict) or str(payload.get("status", "200")) not in {"200", "success", "Success"}:
        message = payload.get("msg", "未知 API 錯誤") if isinstance(payload, dict) else "回傳格式錯誤"
        raise RuntimeError(f"FinMind 日線 API 錯誤：{message}")
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        raise RuntimeError("FinMind 日線資料格式錯誤")
    return normalize_daily_bars(rows, symbol)


def upsert_daily_bars(symbol: str, frame: pd.DataFrame, source: str,
                      path: str | None = None) -> int:
    clean = normalize_daily_bars(frame, symbol)
    if clean.empty:
        return 0
    fetched_at = datetime.now(TZ).isoformat()
    values = [
        (str(symbol), row.date, float(row.open), float(row.high), float(row.low),
         float(row.close), float(row.volume), str(source), fetched_at)
        for row in clean.itertuples(index=False)
    ]
    with closing(_connect(path)) as conn:
        conn.executemany("""
            INSERT INTO daily_bars(symbol,date,open,high,low,close,volume,source,fetched_at)
            VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(symbol,date) DO UPDATE SET
              open=excluded.open, high=excluded.high, low=excluded.low,
              close=excluded.close, volume=excluded.volume,
              source=excluded.source, fetched_at=excluded.fetched_at
        """, values)
        conn.commit()
    return len(values)


def read_daily_bars(symbol: str, start_date: str | None = None,
                    path: str | None = None) -> pd.DataFrame:
    with closing(_connect(path)) as conn:
        if start_date:
            rows = conn.execute(
                "SELECT date,open,high,low,close,volume FROM daily_bars WHERE symbol=? AND date>=? ORDER BY date",
                (str(symbol), start_date),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT date,open,high,low,close,volume FROM daily_bars WHERE symbol=? ORDER BY date",
                (str(symbol),),
            ).fetchall()
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def price_cache_summary(path: str | None = None) -> dict:
    with closing(_connect(path)) as conn:
        count, symbols, latest = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol), MAX(date) FROM daily_bars"
        ).fetchone()
    return {"資料列數": int(count), "已快取標的": int(symbols), "最新交易日": latest}


def period_start(period: str, now=None) -> str:
    current = now or datetime.now(TZ)
    years = {"5y": 5, "3y": 3, "2y": 2, "1y": 1}.get(period, 5)
    return (current.date() - timedelta(days=365 * years + 2)).isoformat()


def get_incremental_history(symbol: str, period: str, token: str = "",
                            fallback_fetcher=None, path: str | None = None):
    """Refresh a 10-calendar-day overlap, then return cached history and status."""
    requested_start = period_start(period)
    cached_all = read_daily_bars(symbol, path=path)
    if cached_all.empty or cached_all["date"].min().strftime("%Y-%m-%d") > requested_start:
        # Backfill the requested horizon if this symbol was previously cached
        # with a shorter period.
        fetch_start = requested_start
    else:
        overlap = (cached_all["date"].max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        fetch_start = max(requested_start, overlap)

    source_status = "FinMind 還原日線"
    refresh_error = None
    try:
        fresh = fetch_finmind_daily(symbol, fetch_start, token=token)
        if not fresh.empty:
            upsert_daily_bars(symbol, fresh, "FinMind TaiwanStockPriceAdj", path)
    except Exception as exc:
        refresh_error = str(exc)

    cached = read_daily_bars(symbol, requested_start, path)
    if cached.empty and fallback_fetcher is not None:
        try:
            fallback = fallback_fetcher(symbol, period)
            if fallback is not None and not fallback.empty:
                upsert_daily_bars(symbol, fallback, "Yahoo Finance auto-adjusted", path)
                cached = read_daily_bars(symbol, requested_start, path)
                source_status = "Yahoo Finance 備援（還原價）"
        except Exception as exc:
            refresh_error = (refresh_error + "；" if refresh_error else "") + str(exc)

    if refresh_error:
        source_status += "；更新失敗，使用可用快取"
    return cached, {"來源狀態": source_status, "更新錯誤": refresh_error,
                    "最後交易日": cached["date"].max().strftime("%Y-%m-%d") if not cached.empty else None}
