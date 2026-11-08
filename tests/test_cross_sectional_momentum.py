import unittest

import numpy as np
import pandas as pd

from cross_sectional_momentum import build_cross_sectional_overrides


def make_universe(symbols=15, bars=420):
    dates = pd.bdate_range("2020-01-02", periods=bars)
    result = {}
    for i in range(symbols):
        # Deterministic, distinct trends and cyclical noise; no future fields.
        t = np.arange(bars, dtype=float)
        close = (40 + i * 2) * np.exp((0.00015 + i * 0.000025) * t)
        close = close * (1 + 0.025 * np.sin(t / (11 + i % 5)))
        result[str(1000 + i)] = pd.DataFrame({
            "date": dates,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(bars, 1_000_000 + i * 1000),
        })
    return result


class CrossSectionalMomentumTests(unittest.TestCase):
    def test_returns_causal_boolean_signals_for_full_universe(self):
        frames = make_universe()
        signals = build_cross_sectional_overrides(frames, minimum_universe=15)
        self.assertEqual(set(signals), set(frames))
        for entry, exit_signal in signals.values():
            self.assertEqual(len(entry), len(frames["1000"]))
            self.assertEqual(len(exit_signal), len(frames["1000"]))
            self.assertTrue(entry.isin([True, False]).all())
            self.assertTrue(exit_signal.isin([True, False]).all())

    def test_future_bars_do_not_change_past_signals(self):
        full = make_universe(bars=420)
        cutoff = 350
        truncated = {code: frame.iloc[:cutoff].copy() for code, frame in full.items()}
        full_signals = build_cross_sectional_overrides(full, minimum_universe=15)
        past_signals = build_cross_sectional_overrides(truncated, minimum_universe=15)
        for code in full:
            np.testing.assert_array_equal(
                full_signals[code][0].iloc[:cutoff].to_numpy(),
                past_signals[code][0].to_numpy(),
            )
            np.testing.assert_array_equal(
                full_signals[code][1].iloc[:cutoff].to_numpy(),
                past_signals[code][1].to_numpy(),
            )

    def test_rejects_invalid_rank_thresholds(self):
        with self.assertRaises(ValueError):
            build_cross_sectional_overrides(make_universe(), entry_quantile=0.4)


if __name__ == "__main__":
    unittest.main()
