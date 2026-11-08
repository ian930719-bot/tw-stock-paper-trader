import unittest

from broker_gateway import BrokerOrderRequest, LiveTradingDisabled, UnconfiguredBroker
from trading_config import INITIAL_CAPITAL_NTD, TradingConfig
from trading_mode import Readiness, StagedTradingController, TradingMode, live_trading_blockers


class StagedTradingTests(unittest.TestCase):
    def test_initial_capital_is_one_hundred_thousand(self):
        self.assertEqual(INITIAL_CAPITAL_NTD, 100000)

    def test_conservative_limits_fit_initial_capital(self):
        config = TradingConfig()
        limits = config.risk_limits()
        self.assertEqual(config.initial_capital, 100000)
        self.assertEqual(limits.max_order_notional, 20000)
        self.assertEqual(limits.max_daily_loss, 2000)
        self.assertFalse(limits.allow_short)

    def test_invalid_daily_loss_is_rejected(self):
        with self.assertRaises(ValueError):
            TradingConfig(max_daily_loss=6000)

    def test_invalid_liquidity_participation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "參與率"):
            TradingConfig(max_participation_rate=0.5)

    def test_live_is_locked_by_default(self):
        controller = StagedTradingController()
        self.assertEqual(controller.mode, TradingMode.PAPER)
        with self.assertRaises(LiveTradingDisabled):
            controller.enable_live(Readiness())

    def test_even_all_readiness_checks_need_a_real_adapter(self):
        controller = StagedTradingController(UnconfiguredBroker())
        ready = Readiness(True, True, True, True, True, True, True)
        self.assertEqual(live_trading_blockers(ready), [])
        with self.assertRaisesRegex(LiveTradingDisabled, "Adapter"):
            controller.enable_live(ready)

    def test_unconfigured_broker_never_reports_fake_success(self):
        broker = UnconfiguredBroker()
        request = BrokerOrderRequest("2330", "BUY", 1000, 100)
        with self.assertRaises(LiveTradingDisabled):
            broker.submit_order(request)


if __name__ == "__main__":
    unittest.main()
