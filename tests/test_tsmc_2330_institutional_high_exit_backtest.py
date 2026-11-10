import unittest
import numpy as np

from tsmc_2330_institutional_high_exit_backtest import frozen_probability_threshold, score


class InstitutionalHighExitBacktestTests(unittest.TestCase):
    def test_validation_probability_threshold_is_quantile(self):
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self.assertAlmostEqual(frozen_probability_threshold(values, 0.9), 0.91)

    def test_threshold_rejects_empty_or_invalid_quantile(self):
        with self.assertRaises(ValueError):
            frozen_probability_threshold(np.array([]), 0.9)
        with self.assertRaises(ValueError):
            frozen_probability_threshold(np.array([0.1, 0.2]), 1.0)

    def test_validation_score_penalizes_drawdown(self):
        strong = {"cagr": 0.25, "max_drawdown": -0.20}
        weak = {"cagr": 0.30, "max_drawdown": -0.60}
        self.assertGreater(score(strong), score(weak))


if __name__ == "__main__":
    unittest.main()
