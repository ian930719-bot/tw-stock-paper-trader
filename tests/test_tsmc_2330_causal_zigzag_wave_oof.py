import unittest
import numpy as np
import pandas as pd

from tsmc_2330_causal_zigzag_wave_oof import causal_atr_zigzag


class CausalZigZagTests(unittest.TestCase):
    def test_low_pivot_only_emitted_after_reversal_confirmation(self):
        close = [100, 98, 96, 94, 92, 93, 95, 97, 99, 101, 103]
        d = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=len(close), freq="B"),
            "raw_close": close,
            "raw_high": np.array(close) + 1,
            "raw_low": np.array(close) - 1,
            "atr_pct14": np.full(len(close), 0.02),
        })
        low_sig, high_sig, events = causal_atr_zigzag(d, 1.5, 2)
        lows = [e for e in events if e["pivot_type"] == "LOW"]
        self.assertEqual(len(lows), 1)
        self.assertGreater(lows[0]["confirmation_index"], lows[0]["pivot_index"])
        self.assertTrue(low_sig[lows[0]["confirmation_index"]])
        self.assertFalse(low_sig[lows[0]["pivot_index"]])

    def test_high_pivot_confirmation_is_not_backdated(self):
        close = [100, 102, 104, 106, 108, 107, 105, 103, 101]
        d = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=len(close), freq="B"),
            "raw_close": close,
            "raw_high": np.array(close) + 1,
            "raw_low": np.array(close) - 1,
            "atr_pct14": np.full(len(close), 0.02),
        })
        low_sig, high_sig, events = causal_atr_zigzag(d, 1.5, 2)
        highs = [e for e in events if e["pivot_type"] == "HIGH"]
        self.assertEqual(len(highs), 1)
        self.assertGreater(highs[0]["confirmation_index"], highs[0]["pivot_index"])
        self.assertTrue(high_sig[highs[0]["confirmation_index"]])
        self.assertFalse(high_sig[highs[0]["pivot_index"]])

    def test_minimum_separation_prevents_same_day_pivot(self):
        d = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "raw_close": [100, 90, 100, 90, 100],
            "raw_high": [101, 91, 101, 91, 101],
            "raw_low": [99, 89, 99, 89, 99],
            "atr_pct14": np.full(5, 0.01),
        })
        _, _, events = causal_atr_zigzag(d, 1.0, 3)
        for e in events:
            self.assertGreaterEqual(e["confirmation_lag_sessions"], 3)


if __name__ == "__main__":
    unittest.main()
