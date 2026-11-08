import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from execution_engine import (
    OrderIntent, PaperExecutionEngine, RiskLimits, Side,
    validate_order, taiwan_regular_session,
)

TZ = ZoneInfo("Asia/Taipei")
OPEN = datetime(2026, 11, 9, 10, 0, tzinfo=TZ)


class ExecutionEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = PaperExecutionEngine(
            str(Path(self.tmp.name) / "execution.sqlite3"),
            RiskLimits(max_order_notional=50000, max_symbol_exposure=100000,
                       max_portfolio_exposure=150000, max_daily_loss=5000)
        )

    def tearDown(self):
        self.tmp.cleanup()

    def intent(self, **kw):
        base = dict(symbol="2330", side=Side.BUY, quantity=1000,
                    limit_price=10, reason="unit test", idempotency_key="test-001")
        base.update(kw)
        return OrderIntent(**base)

    def test_buy_order_accepted_within_limits(self):
        result = self.engine.submit(self.intent(), cash=100000, positions={},
                                    equity=100000, now=OPEN)
        self.assertEqual(result.status, "ACCEPTED")

    def test_reject_outside_market_hours(self):
        with self.assertRaisesRegex(ValueError, "交易時段"):
            validate_order(self.intent(), cash=100000, positions={}, equity=100000,
                           daily_realized_pnl=0, open_order_count=0,
                           daily_order_count=0, limits=self.engine.limits,
                           now=datetime(2026, 11, 9, 15, 0, tzinfo=TZ))

    def test_reject_insufficient_cash(self):
        with self.assertRaisesRegex(ValueError, "現金不足"):
            validate_order(self.intent(), cash=10, positions={}, equity=100000,
                           daily_realized_pnl=0, open_order_count=0,
                           daily_order_count=0, limits=self.engine.limits, now=OPEN)

    def test_reject_daily_loss_limit(self):
        with self.assertRaisesRegex(ValueError, "虧損"):
            validate_order(self.intent(), cash=100000, positions={}, equity=100000,
                           daily_realized_pnl=-5000, open_order_count=0,
                           daily_order_count=0, limits=self.engine.limits, now=OPEN)

    def test_reject_short_sell_without_position(self):
        with self.assertRaisesRegex(ValueError, "持股"):
            validate_order(self.intent(side=Side.SELL), cash=100000, positions={},
                           equity=100000, daily_realized_pnl=0, open_order_count=0,
                           daily_order_count=0, limits=self.engine.limits, now=OPEN)

    def test_idempotency_prevents_duplicate_order(self):
        first = self.engine.submit(self.intent(), cash=100000, positions={},
                                   equity=100000, now=OPEN)
        second = self.engine.submit(self.intent(), cash=100000, positions={},
                                    equity=100000, now=OPEN)
        self.assertEqual(first.order_id, second.order_id)
        self.assertEqual(len(self.engine.list_orders()), 1)

    def test_limit_order_only_fills_when_market_crosses(self):
        order = self.engine.submit(self.intent(), cash=100000, positions={},
                                   equity=100000, now=OPEN)
        with self.assertRaisesRegex(ValueError, "尚未觸及"):
            self.engine.match_limit(order.order_id, 11, now=OPEN)
        filled = self.engine.match_limit(order.order_id, 9, now=OPEN)
        self.assertEqual(filled.status, "FILLED")
        self.assertEqual(filled.filled_quantity, 1000)
        self.assertEqual(filled.average_fill_price, 10)

    def test_reject_order_notional_cap(self):
        with self.assertRaisesRegex(ValueError, "單筆"):
            validate_order(self.intent(quantity=6000, limit_price=10), cash=100000,
                           positions={}, equity=100000, daily_realized_pnl=0,
                           open_order_count=0, daily_order_count=0,
                           limits=self.engine.limits, now=OPEN)

    def test_regular_session_clock(self):
        self.assertTrue(taiwan_regular_session(OPEN))
        self.assertFalse(taiwan_regular_session(
            datetime(2026, 11, 9, 14, 0, tzinfo=TZ)))

    def test_cancel_pending_order(self):
        order = self.engine.submit(self.intent(), cash=100000, positions={},
                                   equity=100000, now=OPEN)
        cancelled = self.engine.cancel_order(order.order_id, now=OPEN)
        self.assertEqual(cancelled.status, "CANCELLED")
        with self.assertRaisesRegex(ValueError, "等待中"):
            self.engine.cancel_order(order.order_id, now=OPEN)

    def test_reject_invalid_tick_and_non_board_lot(self):
        with self.assertRaisesRegex(ValueError, "升降單位"):
            validate_order(self.intent(limit_price=10.02), cash=100000,
                           positions={}, equity=100000, daily_realized_pnl=0,
                           open_order_count=0, daily_order_count=0,
                           limits=self.engine.limits, now=OPEN)
        with self.assertRaisesRegex(ValueError, "1,000 股"):
            validate_order(self.intent(quantity=999), cash=100000,
                           positions={}, equity=100000, daily_realized_pnl=0,
                           open_order_count=0, daily_order_count=0,
                           limits=self.engine.limits, now=OPEN)

    def test_expire_previous_day_accepted_orders(self):
        order = self.engine.submit(self.intent(), cash=100000, positions={},
                                   equity=100000, now=OPEN)
        next_day = datetime(2026, 11, 10, 10, 0, tzinfo=TZ)
        self.assertEqual(self.engine.expire_day_orders(next_day), 1)
        self.assertEqual(self.engine.get_order(order.order_id).status, "EXPIRED")


if __name__ == "__main__":
    unittest.main()
