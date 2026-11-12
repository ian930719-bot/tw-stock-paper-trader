import unittest

import numpy as np
import pandas as pd

from tsmc_2330_atr_reentry_rule_search import make_signals
from tsmc_2330_execution_backtest import trailing_atr_stop_hit


class TestTsmcAtrReentryRuleSearch(unittest.TestCase):
    def test_atr_stop_threshold(self):
        self.assertTrue(trailing_atr_stop_hit(90.0, 100.0, 0.03, 3.5))
        self.assertFalse(trailing_atr_stop_hit(95.0, 100.0, 0.03, 3.5))
        self.assertFalse(trailing_atr_stop_hit(90.0, 100.0, 0.0, 3.5))

    def test_stop_signal_is_not_enabled_during_corporate_action_window(self):
        frame = pd.DataFrame({
            "corporate_action_window": [False, True],
            "atr_pct14": [0.02, 0.02],
            "relative_volume20": [1.3, 1.3],
            "return_1": [-0.01, -0.01],
            "stoch_k14": [20, 20],
            "close_location": [0.1, 0.1],
            "obv_flow_ratio_5": [-0.1, -0.1],
            "intraday_range_pct": [0.02, 0.02],
            "distance_sma_200": [0.1, 0.1],
            "macd_hist_change_3": [-0.1, -0.1],
            "sma50_slope_10": [0.1, 0.1],
        })
        low, high, atr_mask = make_signals(frame, 1.2, 0.7, 0.08)
        self.assertTrue(low[0])
        self.assertFalse(low[1])
        self.assertFalse(high.any())
        self.assertTrue(atr_mask[0])
        self.assertFalse(atr_mask[1])

    def test_atr_stop_invalid_values_fail_safe(self):
        self.assertFalse(trailing_atr_stop_hit(float("nan"), 100.0, 0.02, 3.0))
        self.assertFalse(trailing_atr_stop_hit(90.0, 100.0, -0.02, 3.0))


if __name__ == "__main__":
    unittest.main()
