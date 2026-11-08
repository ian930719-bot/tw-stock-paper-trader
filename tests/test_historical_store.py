"""Tests for incremental daily market data storage."""
import tempfile
import unittest
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from historical_store import (
    normalize_daily_bars, upsert_daily_bars, read_daily_bars,
    price_cache_summary, period_start,
)


class HistoricalStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "market.sqlite3")

    def tearDown(self):
        self.tmp.cleanup()

    def sample(self):
        return pd.DataFrame([{
            "date": "2026-11-06", "open": 100, "high": 105,
            "low": 98, "close": 103, "volume": 1000,
        }])

    def test_normalizes_finmind_columns(self):
        raw = pd.DataFrame([{
            "date": "2026-11-06", "open": 100, "max": 105,
            "min": 98, "close": 103, "Trading_Volume": 1000,
        }])
        result = normalize_daily_bars(raw, "2330")
        self.assertEqual(result.columns.tolist(), ["date", "open", "high", "low", "close", "volume"])
        self.assertEqual(float(result.iloc[0]["high"]), 105)

    def test_rejects_invalid_ohlc(self):
        raw = self.sample()
        raw.loc[0, "high"] = 90
        self.assertTrue(normalize_daily_bars(raw, "2330").empty)

    def test_upsert_is_idempotent_and_updates_corrections(self):
        upsert_daily_bars("2330", self.sample(), "test", self.db)
        corrected = self.sample()
        corrected.loc[0, "close"] = 104
        upsert_daily_bars("2330", corrected, "test2", self.db)
        result = read_daily_bars("2330", path=self.db)
        self.assertEqual(len(result), 1)
        self.assertEqual(float(result.iloc[0]["close"]), 104)

    def test_cache_summary(self):
        upsert_daily_bars("2330", self.sample(), "test", self.db)
        summary = price_cache_summary(self.db)
        self.assertEqual(summary, {"資料列數": 1, "已快取標的": 1, "最新交易日": "2026-11-06"})

    def test_legacy_adjusted_cache_is_excluded_after_schema_migration(self):
        conn = sqlite3.connect(self.db)
        conn.execute("""
            CREATE TABLE daily_bars (
                symbol TEXT NOT NULL, date TEXT NOT NULL, open REAL NOT NULL,
                high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
                volume REAL NOT NULL, source TEXT NOT NULL, fetched_at TEXT NOT NULL,
                PRIMARY KEY(symbol, date)
            )
        """)
        conn.execute("""
            INSERT INTO daily_bars VALUES(
                '2330','2026-11-06',100,105,98,103,1000,
                'FinMind TaiwanStockPriceAdj','2026-11-08T09:00:00+08:00'
            )
        """)
        conn.commit()
        conn.close()
        self.assertTrue(read_daily_bars("2330", path=self.db).empty)
        self.assertEqual(price_cache_summary(self.db)["資料列數"], 0)
        upsert_daily_bars("2330", self.sample(), "FinMind TaiwanStockPrice", self.db)
        self.assertEqual(len(read_daily_bars("2330", path=self.db)), 1)

    def test_adjusted_price_basis_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "只接受 raw"):
            upsert_daily_bars("2330", self.sample(), "adjusted", self.db, price_basis="adjusted")

    def test_period_start(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime(2026, 11, 8, tzinfo=ZoneInfo("Asia/Taipei"))
        self.assertEqual(period_start("1y", now), "2025-11-06")


if __name__ == "__main__":
    unittest.main()
