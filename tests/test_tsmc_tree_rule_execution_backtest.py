import unittest
import numpy as np
import pandas as pd

from tsmc_2330_tree_rule_execution_backtest import tree_low_setup, confirm_low_setup


class TreeRuleExecutionTests(unittest.TestCase):
    def test_low_setup_uses_frozen_return_and_close_location_thresholds(self):
        frame = pd.DataFrame({
            "return_1": [-0.003, -0.001, -0.004],
            "close_location": [0.20, 0.10, 0.30],
            "corporate_action_window": [False, False, False],
        })
        signal = tree_low_setup(frame)
        self.assertEqual(signal.tolist(), [True, False, False])

    def test_confirmation_occurs_only_after_setup_within_wait_window(self):
        frame = pd.DataFrame({
            "high": [10.0, 9.0, 9.5, 10.0, 10.0],
            "close": [9.5, 8.8, 9.2, 10.1, 10.2],
            "close_location": [0.4, 0.2, 0.4, 0.8, 0.8],
            "corporate_action_window": [False] * 5,
        })
        setup = np.array([False, True, False, False, False])
        signal = confirm_low_setup(frame, setup, max_wait=2)
        self.assertEqual(signal.tolist(), [False, False, False, True, False])

    def test_corporate_action_window_suppresses_setup(self):
        frame = pd.DataFrame({
            "return_1": [-0.01],
            "close_location": [0.1],
            "corporate_action_window": [True],
        })
        self.assertEqual(tree_low_setup(frame).tolist(), [False])


if __name__ == "__main__":
    unittest.main()
