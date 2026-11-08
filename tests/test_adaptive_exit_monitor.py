import os
import tempfile
import unittest
from datetime import datetime

from adaptive_exit_monitor import AdaptiveExitMonitor
from execution_engine import TZ


class AdaptiveExitMonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.monitor = AdaptiveExitMonitor(os.path.join(self.tmp.name, "adaptive.sqlite"))
        self.now = datetime(2026, 11, 9, 10, 0, tzinfo=TZ)

    def tearDown(self):
        self.tmp.cleanup()

    def quote(self, minute=0):
        stamp = self.now.replace(minute=minute)
        return stamp.isoformat()

    def configure(self, profile="atr_dynamic", **kwargs):
        return self.monitor.configure_position(
            "2330", 100, 100.0, 2.0, position_version=1,
            profile=profile, now=self.now, **kwargs
        )

    def test_atr_stop_initializes_and_only_tightens(self):
        first = self.configure(atr_multiple=2.0)
        self.assertAlmostEqual(first["stop_price"], 96.0)
        self.assertIsNone(self.monitor.evaluate_quote(
            "2330", 110.0, self.quote(), "test", now=self.now
        ))
        second = self.monitor.configure_position(
            "2330", 100, 100.0, 1.0, position_version=1,
            profile="atr_dynamic", atr_multiple=2.0, now=self.now
        )
        self.assertGreaterEqual(second["stop_price"], first["stop_price"])

    def test_atr_trailing_stop_emits_exit(self):
        self.configure()
        self.monitor.evaluate_quote("2330", 110.0, self.quote(), "test", now=self.now)
        signal = self.monitor.evaluate_quote(
            "2330", 105.0, self.quote(1), "test",
            now=self.now.replace(minute=1)
        )
        self.assertIsNotNone(signal)
        self.assertEqual(signal["side"], "SELL")
        self.assertEqual(signal["quantity"], 100)

    def test_partial_position_update_preserves_original_risk_and_protective_stop(self):
        self.configure()
        self.monitor.evaluate_quote("2330", 110, self.quote(), "test", now=self.now)
        self.monitor.configure_position(
            "2330", 50, 102, 3, position_version=2,
            profile="atr_dynamic", atr_multiple=2.0, now=self.now
        )
        import sqlite3
        with sqlite3.connect(self.monitor.db_path) as conn:
            row = conn.execute(
                "SELECT entry_price,initial_risk,high_water,stop_price,quantity,position_version "
                "FROM adaptive_exit_state WHERE symbol='2330'"
            ).fetchone()
        self.assertEqual(row[0], 100.0)
        self.assertEqual(row[1], 4.0)
        self.assertEqual(row[2], 110.0)
        self.assertGreaterEqual(row[3], 106.0)
        self.assertEqual(row[4:], (50, 2))

    def test_flat_position_clears_state_before_reentry(self):
        self.configure()
        self.assertEqual(self.monitor.clear_absent_positions([]), 1)
        fresh = self.monitor.configure_position(
            "2330", 100, 102, 2, position_version=3,
            profile="atr_dynamic", atr_multiple=2.0, now=self.now
        )
        self.assertAlmostEqual(fresh["stop_price"], 98.0)
        self.assertEqual(fresh["high_water"], 102.0)

    def test_minervini_requires_structural_invalidation(self):
        with self.assertRaises(ValueError):
            self.configure(profile="minervini_sepa")

    def test_minervini_moves_to_breakeven_after_two_r(self):
        self.monitor.configure_position(
            "2330", 100, 100, 2, position_version=1,
            profile="minervini_sepa", technical_invalidation=95,
            now=self.now
        )
        self.monitor.evaluate_quote("2330", 110, self.quote(), "test", now=self.now)
        self.monitor.evaluate_quote(
            "2330", 105, self.quote(1), "test", now=self.now.replace(minute=1)
        )
        state = self.monitor.evaluate_quote(
            "2330", 99, self.quote(2), "test", now=self.now.replace(minute=2)
        )
        # 2R at a 5-point initial risk moves protection to entry (100).
        self.assertIsNotNone(state)
        self.assertEqual(state["trigger_price"], 100)

    def test_oneil_profit_review_is_not_a_hard_take_profit(self):
        self.monitor.configure_position(
            "2330", 100, 100, 2, position_version=1,
            profile="oneil_growth", now=self.now
        )
        self.assertIsNone(self.monitor.evaluate_quote(
            "2330", 121, self.quote(), "test", now=self.now
        ))

    def test_stale_quote_is_rejected(self):
        self.configure()
        old = self.now.replace(minute=0).replace(hour=9)
        with self.assertRaises(ValueError):
            self.monitor.evaluate_quote(
                "2330", 95, old.isoformat(), "test", now=self.now
            )


if __name__ == "__main__":
    unittest.main()
