"""Append-only daily snapshots of Taiwan's listed/OTC security universe.

Snapshots preserve membership observed from the capture date onward. They do
not reconstruct historical constituents before the first successful capture.
Use load_universe_as_of() rather than today's stock master for date-scoped
research, and reject requests earlier than the first available snapshot.
"""
from __future__ import annotations

from contextlib import closing
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo
import sqlite3

import pandas as pd

from historical_store import database_path

TAIPEI = ZoneInfo("Asia/Taipei")
REQUIRED = {"代號", "名稱", "市場", "產業", "市場資料日期"}


def _connect(path: str | None = None):
    target = path or database_path()
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=20)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS universe_snapshot_meta (
            snapshot_date TEXT PRIMARY KEY,
            captured_at TEXT NOT NULL,
            source_date TEXT,
            source TEXT NOT NULL,
            security_count INTEGER NOT NULL CHECK(security_count > 0)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS universe_snapshot_rows (
            snapshot_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            market TEXT NOT NULL,
            industry TEXT NOT NULL,
            source_date TEXT,
            PRIMARY KEY(snapshot_date, symbol),
            FOREIGN KEY(snapshot_date) REFERENCES universe_snapshot_meta(snapshot_date)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_universe_snapshot_symbol ON universe_snapshot_rows(symbol,snapshot_date)")
    return conn


def _capture_time(value=None) -> datetime:
    stamp = value or datetime.now(TAIPEI)
    if isinstance(stamp, date) and not isinstance(stamp, datetime):
        stamp = datetime.combine(stamp, time(23, 59, 59), tzinfo=TAIPEI)
    if not isinstance(stamp, datetime):
        stamp = pd.Timestamp(stamp).to_pydatetime()
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=TAIPEI)
    return stamp.astimezone(TAIPEI)


def save_daily_universe_snapshot(universe: pd.DataFrame, captured_at=None,
                                 source: str = "FinMind TaiwanStockInfo",
                                 path: str | None = None) -> dict:
    """Save the first valid universe observed each Taiwan calendar day."""
    if universe is None or universe.empty:
        raise ValueError("不可保存空白股票主檔")
    missing = REQUIRED.difference(universe.columns)
    if missing:
        raise ValueError("股票主檔缺少欄位：" + ", ".join(sorted(missing)))
    stamp = _capture_time(captured_at)
    snapshot_date = stamp.date().isoformat()
    frame = universe[list(REQUIRED)].copy()
    frame["代號"] = frame["代號"].astype(str).str.strip().str.upper()
    for col in ("名稱", "市場", "產業"):
        frame[col] = frame[col].fillna("").astype(str).str.strip()
    frame["市場資料日期"] = pd.to_datetime(frame["市場資料日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame[
        frame["代號"].str.fullmatch(r"[A-Z0-9]{4,8}", na=False)
        & frame["名稱"].ne("")
        & frame["市場"].isin(["上市", "上櫃"])
    ].drop_duplicates("代號", keep="last")
    if frame.empty:
        raise ValueError("股票主檔沒有可保存的上市／上櫃證券")
    source_dates = frame["市場資料日期"].dropna()
    source_date = source_dates.max() if not source_dates.empty else None
    rows = [
        (snapshot_date, row["代號"], row["名稱"], row["市場"],
         row["產業"], row["市場資料日期"])
        for _, row in frame.iterrows()
    ]
    with closing(_connect(path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT captured_at,source_date,source,security_count "
            "FROM universe_snapshot_meta WHERE snapshot_date=?",
            (snapshot_date,),
        ).fetchone()
        if existing:
            conn.execute("COMMIT")
            return {
                "saved": False, "snapshot_date": snapshot_date,
                "captured_at": existing[0], "source_date": existing[1],
                "source": existing[2], "security_count": int(existing[3]),
                "status": "當日快照已存在，保留首次觀測資料",
            }
        conn.execute(
            "INSERT INTO universe_snapshot_meta VALUES(?,?,?,?,?)",
            (snapshot_date, stamp.isoformat(), source_date, str(source), len(rows)),
        )
        conn.executemany(
            "INSERT INTO universe_snapshot_rows VALUES(?,?,?,?,?,?)", rows
        )
        conn.execute("COMMIT")
    return {
        "saved": True, "snapshot_date": snapshot_date,
        "captured_at": stamp.isoformat(), "source_date": source_date,
        "source": str(source), "security_count": len(rows),
        "status": "已保存當日首次觀測快照",
    }


def load_universe_as_of(as_of, path: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Return the latest snapshot captured no later than the requested instant."""
    cutoff = _capture_time(as_of)
    with closing(_connect(path)) as conn:
        meta = conn.execute(
            "SELECT snapshot_date,captured_at,source_date,source,security_count "
            "FROM universe_snapshot_meta WHERE captured_at<=? "
            "ORDER BY captured_at DESC LIMIT 1",
            (cutoff.isoformat(),),
        ).fetchone()
        if meta is None:
            return pd.DataFrame(columns=["代號", "名稱", "市場", "產業", "市場資料日期"]), {
                "available": False, "status": "指定時間以前沒有已保存的歷史股票主檔；不可使用目前名單代替",
                "requested_as_of": cutoff.isoformat(),
            }
        rows = conn.execute(
            "SELECT symbol,name,market,industry,source_date FROM universe_snapshot_rows "
            "WHERE snapshot_date=? ORDER BY market,symbol",
            (meta[0],),
        ).fetchall()
    frame = pd.DataFrame(rows, columns=["代號", "名稱", "市場", "產業", "市場資料日期"])
    return frame, {
        "available": True, "snapshot_date": meta[0], "captured_at": meta[1],
        "source_date": meta[2], "source": meta[3], "security_count": int(meta[4]),
        "requested_as_of": cutoff.isoformat(),
        "status": "使用當時已觀測並保存的證券名單",
    }


def universe_snapshot_history(path: str | None = None) -> pd.DataFrame:
    with closing(_connect(path)) as conn:
        rows = conn.execute(
            "SELECT snapshot_date,captured_at,source_date,source,security_count "
            "FROM universe_snapshot_meta ORDER BY snapshot_date"
        ).fetchall()
    return pd.DataFrame(rows, columns=[
        "快照日期", "擷取時間", "主檔資料日期", "資料來源", "證券數"
    ])
