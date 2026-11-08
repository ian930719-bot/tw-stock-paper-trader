"""Tests for point-in-time listed/OTC universe snapshots."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from universe_history import (
    save_daily_universe_snapshot, load_universe_as_of, universe_snapshot_history,
)

TZ = ZoneInfo("Asia/Taipei")


class UniverseHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "universe.sqlite3")

    def tearDown(self):
        self.tmp.cleanup()

    def frame(self, symbols):
        return pd.DataFrame([
            {"代號": code, "名稱": name, "市場": "上市", "產業": "電子",
             "市場資料日期": "2026-11-06"}
            for code, name in symbols
        ])

    def test_as_of_uses_only_snapshots_already_captured(self):
        save_daily_universe_snapshot(
            self.frame([("2330", "台積電"), ("2603", "長榮")]),
            datetime(2026, 11, 6, 15, 0, tzinfo=TZ), path=self.db,
        )
        save_daily_universe_snapshot(
            self.frame([("2330", "台積電")]),
            datetime(2026, 11, 9, 15, 0, tzinfo=TZ), path=self.db,
        )
        before_first, missing = load_universe_as_of(
            datetime(2026, 11, 6, 14, 59, tzinfo=TZ), self.db
        )
        self.assertTrue(before_first.empty)
        self.assertFalse(missing["available"])
        friday, friday_meta = load_universe_as_of(
            datetime(2026, 11, 8, 12, 0, tzinfo=TZ), self.db
        )
        self.assertTrue(friday_meta["available"])
        self.assertEqual(set(friday["代號"]), {"2330", "2603"})
        monday, monday_meta = load_universe_as_of(
            datetime(2026, 11, 9, 15, 1, tzinfo=TZ), self.db
        )
        self.assertTrue(monday_meta["available"])
        self.assertEqual(monday["代號"].tolist(), ["2330"])

    def test_daily_snapshot_is_immutable_and_idempotent(self):
        first = save_daily_universe_snapshot(
            self.frame([("2330", "台積電"), ("2603", "長榮")]),
            datetime(2026, 11, 6, 15, 0, tzinfo=TZ), path=self.db,
        )
        second = save_daily_universe_snapshot(
            self.frame([("2330", "台積電")]),
            datetime(2026, 11, 6, 16, 0, tzinfo=TZ), path=self.db,
        )
        self.assertTrue(first["saved"])
        self.assertFalse(second["saved"])
        self.assertEqual(second["security_count"], 2)
        self.assertEqual(len(universe_snapshot_history(self.db)), 1)

    def test_rejects_empty_or_malformed_universe(self):
        with self.assertRaises(ValueError):
            save_daily_universe_snapshot(pd.DataFrame(), path=self.db)
        with self.assertRaises(ValueError):
            save_daily_universe_snapshot(pd.DataFrame([{"代號": "2330"}]), path=self.db)


if __name__ == "__main__":
    unittest.main()
