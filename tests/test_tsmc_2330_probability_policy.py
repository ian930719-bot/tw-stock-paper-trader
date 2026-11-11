import unittest
import numpy as np

from tsmc_2330_probability_strategy_backtest import (
    rolling_probability_threshold, rolling_probability_signal,
)


class CausalProbabilityPolicyTests(unittest.TestCase):
    def test_threshold_uses_only_prior_probabilities(self):
        values = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
        thresholds = rolling_probability_threshold(
            values, quantile=0.5, window=3, min_periods=2
        )
        self.assertTrue(np.isnan(thresholds[0]))
        self.assertTrue(np.isnan(thresholds[1]))
        self.assertAlmostEqual(thresholds[2], 0.15)
        self.assertAlmostEqual(thresholds[3], 0.20)
        self.assertAlmostEqual(thresholds[4], 0.30)

    def test_current_probability_does_not_change_its_own_cutoff(self):
        baseline = np.array([0.10, 0.20, 0.30, 0.40])
        changed = baseline.copy()
        changed[3] = 0.99
        a = rolling_probability_threshold(
            baseline, quantile=0.5, window=3, min_periods=2
        )
        b = rolling_probability_threshold(
            changed, quantile=0.5, window=3, min_periods=2
        )
        self.assertEqual(a[3], b[3])
        self.assertNotEqual(a[3], b[2])

    def test_signal_requires_prior_history(self):
        signal, threshold = rolling_probability_signal(
            np.array([0.1, 0.2, 0.3, 0.4]), quantile=0.5
        )
        self.assertFalse(signal.any())
        self.assertTrue(np.isnan(threshold).all())


if __name__ == "__main__":
    unittest.main()
