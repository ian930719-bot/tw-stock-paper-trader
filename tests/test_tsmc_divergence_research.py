import unittest
import numpy as np
import pandas as pd

from tsmc_2330_divergence_research import candidate_signals, calc_metrics, target_for


class TSMCDivergenceResearchTests(unittest.TestCase):
    def test_predeclared_divergence_combinations(self):
        frame = pd.DataFrame({
            "bearish_rsi_divergence20": [1, 0, 1],
            "bullish_rsi_divergence20": [0, 1, 0],
            "bearish_macd_divergence20": [1, 1, 0],
            "bullish_macd_divergence20": [0, 1, 1],
            "bearish_volume_divergence20": [0, 1, 0],
            "bullish_volume_divergence20": [1, 0, 1],
            "upper_wick_pct": [0.01, 0.0, 0.0],
            "lower_wick_pct": [0.0, 0.01, 0.0],
            "close_location": [0.2, 0.8, 0.4],
        })
        high = np.array([True, False, True])
        low = np.array([False, True, False])
        candidates = candidate_signals(frame, low, high)
        np.testing.assert_array_equal(candidates["HIGH"]["bearish_rsi"], [True, False, True])
        np.testing.assert_array_equal(candidates["LOW"]["bullish_rsi"], [False, True, False])
        np.testing.assert_array_equal(candidates["HIGH"]["base_high_and_bearish_rsi"], [True, False, True])
        np.testing.assert_array_equal(candidates["LOW"]["base_low_and_bullish_macd"], [False, True, False])

    def test_metrics_precision_recall(self):
        result = calc_metrics(np.array([True, True, False]), np.array([True, False, True]))
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 0.5)

    def test_pivot_target_marks_only_pre_pivot_sessions(self):
        dates = pd.Series(pd.bdate_range("2020-01-01", periods=10))
        events = pd.DataFrame({"type": ["HIGH"], "pivot_date": [dates.iloc[5]]})
        target = target_for(dates, events, 3, "HIGH")
        self.assertEqual(np.flatnonzero(target).tolist(), [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
