import unittest
import numpy as np
import pandas as pd

from multifactor_engine import factor_frame, multifactor_signals


def sample(n=420):
    t = np.arange(n, dtype=float)
    close = 80 * np.exp(0.0008 * t + 0.025 * np.sin(t / 8))
    return pd.DataFrame({
        "date": pd.bdate_range("2023-01-02", periods=n),
        "open": close * 0.999,
        "high": close * 1.012,
        "low": close * 0.988,
        "close": close,
        "volume": 1_000_000 + (t % 17) * 30_000,
    })


class MultiFactorTests(unittest.TestCase):
    def test_calculates_bollinger_ma_momentum_volume_and_regime(self):
        result = factor_frame(sample())
        for column in ["bb_percent_b", "bb_bandwidth", "rsi14", "macd_hist",
                       "volume_ratio20", "momentum20", "momentum60",
                       "atr14_pct", "market_regime"]:
            self.assertIn(column, result.columns)
        self.assertTrue(result["market_regime"].notna().any())

    def test_signals_are_boolean_and_auditable(self):
        entry, exit_signal, audit = multifactor_signals(sample())
        self.assertEqual(len(entry), 420)
        self.assertEqual(len(exit_signal), 420)
        self.assertTrue(pd.api.types.is_bool_dtype(entry))
        self.assertIn("因子得分", audit.columns)

    def test_no_future_data_leakage(self):
        original = sample()
        changed = original.copy()
        changed.loc[300:, ["open", "high", "low", "close", "volume"]] *= 3
        _, _, a = multifactor_signals(original)
        _, _, b = multifactor_signals(changed)
        pd.testing.assert_frame_equal(a.iloc[:300].reset_index(drop=True),
                                      b.iloc[:300].reset_index(drop=True))

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            multifactor_signals(sample(), min_entry_score=0)


if __name__ == "__main__":
    unittest.main()
