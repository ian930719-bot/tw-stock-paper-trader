import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from execution_engine import OrderIntent, PaperExecutionEngine, RiskLimits, Side
from paper_account import PaperAccount

TZ = ZoneInfo("Asia/Taipei")
NOW = datetime(2026, 11, 9, 10, 0, tzinfo=TZ)


class PaperAccountTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "paper.sqlite3")
        self.engine = PaperExecutionEngine(self.db, RiskLimits(
            max_order_notional=50000, max_symbol_exposure=100000,
            max_portfolio_exposure=150000))
        self.account = PaperAccount(self.db, initial_cash=100000)

    def tearDown(self):
        self.tmp.cleanup()

    def order(self, side, quantity, key):
        return self.engine.submit(
            OrderIntent("2330", side, quantity, 10, "test", key),
            cash=100000, positions={"2330": 1000 if side == Side.SELL else 0},
            equity=100000, now=NOW)

    def quote(self, price=9, at=NOW, source="test-feed"):
        return self.account.process_quote(
            "2330", price, at.isoformat(), source, now=NOW)

    def test_buy_fill_updates_cash_and_average_cost(self):
        order = self.order(Side.BUY, 1000, "buy-1")
        fills = self.quote()
        self.assertEqual(len(fills), 1)
        snap = self.account.snapshot()
        self.assertEqual(snap["positions"][0]["quantity"], 1000)
        self.assertEqual(snap["fills"][0]["price"], 10)
        self.assertEqual(self.engine.get_order(order.order_id).average_fill_price, 10)
        self.assertLess(snap["cash"], 100000)
        self.assertEqual(self.engine.get_order(order.order_id).status, "FILLED")

    def test_sell_fill_updates_cash_and_realized_pnl(self):
        self.order(Side.BUY, 1000, "buy-first")
        self.quote(9)
        sell = self.engine.submit(
            OrderIntent("2330", Side.SELL, 1000, 11, "exit", "sell-1"),
            cash=90000, positions={"2330": 1000}, equity=100000, now=NOW)
        fills = self.quote(12)
        self.assertEqual(len(fills), 1)
        self.assertGreater(fills[0]["realized_pnl"], 0)
        self.assertEqual(self.account.snapshot()["positions"], [])
        self.assertEqual(self.engine.get_order(sell.order_id).status, "FILLED")

    def test_stale_quote_is_rejected(self):
        self.order(Side.BUY, 1000, "stale-1")
        stale = datetime(2026, 11, 9, 9, 55, tzinfo=TZ)
        with self.assertRaisesRegex(ValueError, "過期"):
            self.quote(at=stale)

    def test_missing_source_is_rejected(self):
        self.order(Side.BUY, 1000, "source-1")
        with self.assertRaisesRegex(ValueError, "來源"):
            self.quote(source="")

    def test_outside_session_is_rejected(self):
        self.order(Side.BUY, 1000, "closed-1")
        closed = datetime(2026, 11, 9, 14, 0, tzinfo=TZ)
        with self.assertRaisesRegex(ValueError, "非台股"):
            self.account.process_quote("2330", 9, closed.isoformat(), "test", now=closed)

    def test_no_fill_when_limit_not_crossed(self):
        order = self.order(Side.BUY, 1000, "not-crossed")
        self.assertEqual(self.quote(11), [])
        self.assertEqual(self.engine.get_order(order.order_id).status, "ACCEPTED")

    def test_idempotent_fill_processing(self):
        order = self.order(Side.BUY, 1000, "once-only")
        self.quote(9)
        self.assertEqual(self.quote(99), [])
        self.assertEqual(len(self.account.snapshot()["fills"]), 1)
        self.assertEqual(self.engine.get_order(order.order_id).status, "FILLED")


if __name__ == "__main__":
    unittest.main()
