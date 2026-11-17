import unittest

import numpy as np
import pandas as pd

from tsmc_2330_sma200_regime_research import confirm_consecutive


class SMA200RegimeResearchTests(unittest.TestCase):
    def test_requires_full_consecutive_confirmation(self):
        signal = pd.Series([False, True, True, False, True, True, True])
        got = confirm_consecutive(signal, 2)
        self.assertEqual(got.tolist(), [False, False, True, False, False, True, True])

    def test_one_session_confirmation_preserves_signal(self):
        signal = np.array([False, True, False, True], dtype=bool)
        got = confirm_consecutive(signal, 1)
        self.assertEqual(got.tolist(), signal.tolist())

    def test_rejects_zero_confirmation(self):
        with self.assertRaises(ValueError):
            confirm_consecutive([True, False], 0)


if __name__ == "__main__":
    unittest.main()
