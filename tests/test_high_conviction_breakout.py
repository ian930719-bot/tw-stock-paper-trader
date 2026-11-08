import unittest
import numpy as np
import pandas as pd

from high_conviction_breakout import build_high_conviction_breakout_overrides


def universe(symbols=15, bars=500):
    dates = pd.bdate_range("2020-01-02", periods=bars)
    frames = {}
    for i in range(symbols):
        t = np.arange(bars, dtype=float)
        # Trend with deterministic breakouts and volume spikes.
        close = (35 + i) * np.exp((0.0003 + i * 0.00002) * t)
        close *= 1 + 0.018 * np.sin(t / (9 + i % 4))
        volume = np.full(bars, 1_000_000 + i * 10_000, dtype=float)
        volume[::37] *= 2.2
        frames[str(1000 + i)] = pd.DataFrame({
            "date": dates, "open": close * .998, "high": close * 1.012,
            "low": close * .988, "close": close, "volume": volume,
        })
    return frames


class HighConvictionBreakoutTests(unittest.TestCase):
    def test_outputs_boolean_signals_for_every_symbol(self):
        frames = universe()
        signals = build_high_conviction_breakout_overrides(frames, minimum_universe=15)
        self.assertEqual(set(signals), set(frames))
        for entry, exit_signal in signals.values():
            self.assertEqual(len(entry), len(frames["1000"]))
            self.assertTrue(entry.isin([True, False]).all())
            self.assertTrue(exit_signal.isin([True, False]).all())

    def test_future_bars_cannot_change_past_signals(self):
        full = universe()
        cutoff = 410
        truncated = {k: v.iloc[:cutoff].copy() for k, v in full.items()}
        a = build_high_conviction_breakout_overrides(full, minimum_universe=15)
        b = build_high_conviction_breakout_overrides(truncated, minimum_universe=15)
        for code in full:
            np.testing.assert_array_equal(a[code][0].iloc[:cutoff], b[code][0])
            np.testing.assert_array_equal(a[code][1].iloc[:cutoff], b[code][1])

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            build_high_conviction_breakout_overrides(
                universe(), relative_strength_quantile=0.4
            )


if __name__ == "__main__":
    unittest.main()
