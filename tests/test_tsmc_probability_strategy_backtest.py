import unittest

import numpy as np

from tsmc_2330_probability_strategy_backtest import (
    probability_signal, rolling_probability_signal, rolling_probability_threshold, score,
)
from tsmc_2330_selected_threshold_signal_diagnostics import build_target
import pandas as pd


class ProbabilityStrategyTests(unittest.TestCase):
    def test_probability_signal_uses_frozen_cutoff(self):
        p = np.array([0.1, 0.5, 0.7, 0.9, np.nan])
        np.testing.assert_array_equal(
            probability_signal(p, 0.7),
            np.array([False, False, True, True, False]),
        )

    def test_rolling_threshold_is_lagged_and_adapts(self):
        p = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.6])
        threshold = rolling_probability_threshold(p, 0.5, window=2, min_periods=2)
        self.assertTrue(np.isnan(threshold[0]))
        self.assertTrue(np.isnan(threshold[1]))
        self.assertAlmostEqual(threshold[2], 0.15)
        self.assertAlmostEqual(threshold[3], 0.55)
        signal, _ = rolling_probability_signal(p, 0.5, window=2, min_periods=2)
        np.testing.assert_array_equal(signal, np.array([False, False, True, True, False, False]))

    def test_rolling_threshold_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            rolling_probability_threshold(np.array([0.1, 0.2]), 1.0)
        with self.assertRaises(ValueError):
            rolling_probability_threshold(np.array([0.1, 0.2]), 0.5, window=2, min_periods=3)

    def test_score_penalizes_drawdown(self):
        strong = {"cagr": 0.30, "max_drawdown": -0.20}
        weak = {"cagr": 0.30, "max_drawdown": -0.50}
        self.assertGreater(score(strong), score(weak))

    def test_next_session_target_marks_only_previous_session(self):
        dates = pd.Series(pd.to_datetime([
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"
        ]))
        events = pd.DataFrame({
            "type": ["LOW"],
            "pivot_date": [pd.Timestamp("2024-01-04")],
        })
        target = build_target(dates, events, 1, "LOW")
        np.testing.assert_array_equal(target, np.array([0, 1, 0, 0], dtype=np.int8))

    def test_probability_signal_is_boolean_and_causal_shape(self):
        p = np.linspace(0, 1, 21)
        signal = probability_signal(p, 0.8)
        self.assertEqual(signal.shape, p.shape)
        self.assertEqual(signal.dtype, np.bool_)
        self.assertEqual(int(signal.sum()), 5)


if __name__ == "__main__":
    unittest.main()
