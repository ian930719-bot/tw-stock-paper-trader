import unittest
import pandas as pd
import numpy as np

from tsmc_2330_trend_reentry_diagnostic import _signals, _future_diagnostic


class TrendReentryDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.bars = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "raw_open": [10, 10, 11, 12, 13],
            "raw_high": [11, 11, 12, 13, 14],
            "raw_low": [9, 9, 10, 11, 12],
            "raw_close": [10, 10, 12, 13, 14],
            "distance_ema_20": [-0.02, -0.01, 0.02, 0.03, 0.04],
            "rsi14": [45, 49, 51, 53, 54],
            "macd_hist_pct": [-1, -0.5, 0.1, 0.2, 0.3],
            "relative_volume20": [1, 1, 1, 1, 1],
            "atr_pct14": [.02] * 5,
            "sma20_slope_5": [0] * 5,
            "sma50_slope_10": [0] * 5,
            "obv_flow_ratio_5": [0] * 5,
            "close_location": [.5] * 5,
            "intraday_range_pct": [.01] * 5,
        })

    def test_signal_uses_current_and_prior_close_data(self):
        s = _signals(self.bars)
        self.assertTrue(bool(s["ema_20_rsi_macd"].iloc[2]))
        self.assertTrue(bool(s["break_prev_high"].iloc[2]))
        self.assertFalse(bool(s["ema_20_rsi_macd"].iloc[0]))

    def test_outcome_diagnostic_is_separate(self):
        x = _future_diagnostic(self.bars, 1, 90)
        self.assertEqual(x["observations"], 3)
        self.assertEqual(x["future_max_return"], 14 / 10 - 1)
        self.assertEqual(x["future_min_return"], 10 / 10 - 1)


if __name__ == "__main__":
    unittest.main()
