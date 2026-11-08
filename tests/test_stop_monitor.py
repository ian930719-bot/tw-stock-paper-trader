import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from execution_engine import PaperExecutionEngine, OrderIntent, Side
from paper_account import PaperAccount
from stop_monitor import StopMonitor

TZ = ZoneInfo("Asia/Taipei")
NOW = datetime(2026, 11, 9, 10, 0, tzinfo=TZ)


class StopMonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "paper.sqlite3")
        self.engine = PaperExecutionEngine(self.db)
        self.account = PaperAccount(self.db, initial_cash=100000)
        self.monitor = StopMonitor(self.db, take_profit_pct=0.99)

    def tearDown(self):
        self.tmp.cleanup()

    def buy_position(self, quantity=10, price=100):
        order = self.engine.submit(
            OrderIntent("2330", Side.BUY, quantity, price, "seed", "seed-buy"),
            cash=100000, positions={}, equity=100000, now=NOW)
        self.account.process_quote("2330", price, NOW.isoformat(), "test", now=NOW)
        self.assertEqual(self.engine.get_order(order.order_id).status, "FILLED")

    def evaluate(self, price, at=NOW, now=NOW):
        return self.monitor.evaluate_quote(
            "2330", price, at.isoformat(), "test-feed", now=now)

    def test_stop_loss_signal(self):
        self.buy_position()
        signal = self.evaluate(94)
        self.assertEqual(signal["reason"], "STOP_LOSS")
        self.assertEqual(signal["quantity"], 10)

    def test_take_profit_signal(self):
        self.buy_position()
        take_profit_monitor = StopMonitor(self.db, take_profit_pct=0.10)
        signal = take_profit_monitor.evaluate_quote(
            "2330", 111, NOW.isoformat(), "test-feed", now=NOW)
        self.assertEqual(signal["reason"], "TAKE_PROFIT")

    def test_trailing_stop_after_new_high(self):
        self.buy_position()
        self.assertIsNone(self.evaluate(120))
        signal = self.evaluate(113)
        self.assertEqual(signal["reason"], "TRAILING_STOP")

    def test_new_position_resets_previous_trailing_high_water(self):
        self.buy_position()
        self.assertIsNone(self.evaluate(120))
        # Simulate a full close followed by a new position with the same
        # quantity and average cost. The lifecycle version must still differ.
        import sqlite3
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE paper_positions SET position_version=position_version+1 WHERE symbol='2330'")
        self.assertIsNone(self.evaluate(112))
        with sqlite3.connect(self.db) as conn:
            high_water, version = conn.execute(
                "SELECT high_water,position_version FROM paper_stop_state WHERE symbol='2330'"
            ).fetchone()
        self.assertEqual(high_water, 112)
        self.assertEqual(version, 2)

    def test_flat_position_clears_trailing_state(self):
        self.buy_position()
        self.assertIsNone(self.evaluate(120))
        import sqlite3
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE paper_positions SET quantity=0,average_cost=0,position_version=position_version+1 WHERE symbol='2330'")
        self.assertIsNone(self.evaluate(100))
        with sqlite3.connect(self.db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM paper_stop_state WHERE symbol='2330'").fetchone()[0]
        self.assertEqual(count, 0)

    def test_stale_quote_does_not_trigger(self):
        self.buy_position()
        old = NOW - timedelta(minutes=5)
        with self.assertRaisesRegex(ValueError, "過期"):
            self.evaluate(90, old, now=NOW)

    def test_no_position_no_signal(self):
        self.assertIsNone(self.evaluate(90))

    def test_duplicate_event_not_emitted(self):
        self.buy_position()
        self.assertIsNotNone(self.evaluate(94))
        self.assertIsNone(self.evaluate(94))
        self.assertEqual(len(self.monitor.events()), 1)

    def test_reject_invalid_threshold(self):
        with self.assertRaises(ValueError):
            StopMonitor(self.db, stop_loss_pct=1.0)


if __name__ == "__main__":
    unittest.main()
