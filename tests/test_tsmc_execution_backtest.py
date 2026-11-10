import unittest
import numpy as np
import pandas as pd

from tsmc_2330_execution_backtest import (
    commission, sell_tax, is_limit_locked, buy_quantity, build_signals,
)


class TSMCExecutionBacktestTests(unittest.TestCase):
    def test_standard_costs(self):
        self.assertEqual(commission(100000), 143)
        self.assertEqual(commission(1000), 20)
        self.assertEqual(sell_tax(100000), 300)

    def test_quantity_respects_commission_and_cash(self):
        qty = buy_quantity(100000, 200.0)
        self.assertGreater(qty, 0)
        self.assertLessEqual(qty * 200.0 + commission(qty * 200.0), 100000)

    def test_locked_limit_buy_and_sell(self):
        self.assertTrue(is_limit_locked("BUY", 110.0, 110.0, 110.0, 110.0, 90.0))
        self.assertFalse(is_limit_locked("BUY", 110.0, 111.0, 109.5, 110.0, 90.0))
        self.assertTrue(is_limit_locked("SELL", 90.0, 90.0, 89.0, 110.0, 90.0))
        self.assertFalse(is_limit_locked("SELL", 90.0, 91.0, 89.0, 110.0, 90.0))

    def test_frozen_rule_signals_are_causal_threshold_comparisons(self):
        frame = pd.DataFrame({
            "return_3": [0.02, 0.00, -0.02],
            "distance_sma_20": [0.03, 0.00, -0.03],
            "bb_position20": [0.9, 0.5, 0.1],
            "close_location": [0.9, 0.5, 0.1],
            "corporate_action_window": [False, False, False],
        })
        rules = {"rules": {
            "HIGH": {"selected": True, "features": ["return_3", "distance_sma_20"],
                     "training_thresholds": {"return_3": 0.01, "distance_sma_20": 0.02}},
            "LOW": {"selected": True, "features": ["bb_position20", "close_location"],
                    "training_thresholds": {"bb_position20": 0.2, "close_location": 0.3}},
        }}
        low, high = build_signals(frame, rules)
        np.testing.assert_array_equal(high, [True, False, False])
        np.testing.assert_array_equal(low, [False, False, True])

    def test_corporate_action_window_suppresses_signals(self):
        frame = pd.DataFrame({
            "return_3": [0.02], "distance_sma_20": [0.03],
            "bb_position20": [0.1], "close_location": [0.1],
            "corporate_action_window": [True],
        })
        rules = {"rules": {
            "HIGH": {"selected": True, "features": ["return_3"],
                     "training_thresholds": {"return_3": 0.01}},
            "LOW": {"selected": True, "features": ["bb_position20"],
                    "training_thresholds": {"bb_position20": 0.2}},
        }}
        low, high = build_signals(frame, rules)
        self.assertFalse(low[0])
        self.assertFalse(high[0])


if __name__ == "__main__":
    unittest.main()
