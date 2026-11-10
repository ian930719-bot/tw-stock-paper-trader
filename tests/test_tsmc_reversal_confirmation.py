import unittest
import numpy as np
import pandas as pd

from tsmc_2330_reversal_confirmation_backtest import build_confirmed_signals


class TSMCReversalConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.rules = {"rules": {
            "LOW": {"selected": True, "features": ["bb_position20", "close_location"],
                    "training_thresholds": {"bb_position20": 0.25, "close_location": 0.35}},
            "HIGH": {"selected": True, "features": ["return_3", "distance_sma_20", "close_location"],
                     "training_thresholds": {"return_3": 0.01, "distance_sma_20": 0.02, "close_location": 0.65}},
        }}

    def frame(self):
        return pd.DataFrame({
            "bb_position20": [0.1, 0.5, 0.5, 0.5],
            "close_location": [0.2, 0.8, 0.5, 0.2],
            "return_3": [0.0, 0.0, 0.02, 0.0],
            "distance_sma_20": [0.0, 0.0, 0.03, 0.0],
            "return_1": [-0.02, 0.01, 0.02, -0.01],
            "rsi14": [35.0, 40.0, 60.0, 55.0],
            "distance_sma_5": [-0.03, 0.01, 0.02, -0.01],
            "close": [90.0, 92.0, 110.0, 108.0],
            "high": [95.0, 93.0, 111.0, 109.0],
            "low": [89.0, 90.0, 109.0, 107.0],
            "corporate_action_window": [False, False, False, False],
        })

    def test_positive_negative_close_waits_for_confirmation(self):
        low, high = build_confirmed_signals(self.frame(), self.rules, "positive_negative_close")
        self.assertFalse(low[0])
        self.assertTrue(low[1])
        self.assertFalse(high[2])
        self.assertTrue(high[3])

    def test_unconfirmed_baseline_is_unchanged(self):
        low, high = build_confirmed_signals(self.frame(), self.rules, "unconfirmed_baseline")
        np.testing.assert_array_equal(low, [True, False, False, False])
        np.testing.assert_array_equal(high, [False, False, True, False])

    def test_confirmation_expires_after_max_wait(self):
        frame = self.frame()
        frame.loc[1:3, "close_location"] = 0.2
        frame.loc[3, "return_1"] = 0.01
        frame.loc[3, "close_location"] = 0.8
        low, _ = build_confirmed_signals(frame, self.rules, "positive_negative_close", max_wait=2)
        self.assertFalse(low[3])

    def test_corporate_action_window_suppresses_confirmation(self):
        frame = self.frame()
        frame.loc[1, "corporate_action_window"] = True
        low, _ = build_confirmed_signals(frame, self.rules, "positive_negative_close")
        self.assertFalse(low[1])


if __name__ == "__main__":
    unittest.main()
