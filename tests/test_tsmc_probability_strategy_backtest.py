import unittest

import numpy as np

from tsmc_2330_probability_strategy_backtest import probability_signal, score


class ProbabilityStrategyTests(unittest.TestCase):
    def test_probability_signal_uses_frozen_cutoff(self):
        p = np.array([0.1, 0.5, 0.7, 0.9, np.nan])
        np.testing.assert_array_equal(
            probability_signal(p, 0.7),
            np.array([False, False, True, True, False]),
        )

    def test_score_penalizes_drawdown(self):
        strong = {"cagr": 0.30, "max_drawdown": -0.20}
        weak = {"cagr": 0.30, "max_drawdown": -0.50}
        self.assertGreater(score(strong), score(weak))

    def test_probability_signal_is_boolean_and_causal_shape(self):
        p = np.linspace(0, 1, 21)
        signal = probability_signal(p, 0.8)
        self.assertEqual(signal.shape, p.shape)
        self.assertEqual(signal.dtype, np.bool_)
        self.assertEqual(int(signal.sum()), 5)


if __name__ == "__main__":
    unittest.main()
