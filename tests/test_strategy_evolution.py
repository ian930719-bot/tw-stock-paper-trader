"""Tests for the bounded AI strategy evolution research module."""
import unittest
import numpy as np
import pandas as pd

from strategy_evolution import build_signals, candidate_grid, evolve_strategy


def frame(n=620):
    dates = pd.bdate_range("2021-01-04", periods=n)
    t = np.arange(n, dtype=float)
    close = 50 * np.exp(0.0005 * t + 0.025 * np.sin(t / 7))
    open_ = close * (1 - 0.001 * np.sin(t / 5))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = 1_000_000 + (t % 19) * 20_000
    return pd.DataFrame({
        "date": dates, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


class StrategyEvolutionTests(unittest.TestCase):
    def test_candidate_space_is_bounded_and_reproducible(self):
        candidates = list(candidate_grid())
        self.assertEqual(len(candidates), 18)
        self.assertEqual(len({tuple(sorted(x.items())) for x in candidates}), 18)

    def test_signals_do_not_use_future_bars(self):
        original = frame()
        altered = original.copy()
        altered.loc[400:, ["open", "high", "low", "close", "volume"]] *= 4
        params = {"breakout": 20, "volume_multiple": 1.2, "trend_period": 50}
        entry_a, exit_a = build_signals(original, params)
        entry_b, exit_b = build_signals(altered, params)
        pd.testing.assert_series_equal(entry_a.iloc[:400], entry_b.iloc[:400])
        pd.testing.assert_series_equal(exit_a.iloc[:400], exit_b.iloc[:400])

    def test_multifactor_candidate_signals_use_only_past_bars(self):
        original = frame()
        altered = original.copy()
        altered.loc[400:, ["open", "high", "low", "close", "volume"]] *= 3
        params = {"signal_family": "multifactor", "min_entry_score": 4, "max_atr_pct": 0.12}
        entry_a, exit_a = build_signals(original, params)
        entry_b, exit_b = build_signals(altered, params)
        pd.testing.assert_series_equal(entry_a.iloc[:400], entry_b.iloc[:400])
        pd.testing.assert_series_equal(exit_a.iloc[:400], exit_b.iloc[:400])

    def test_insufficient_history_is_rejected_cleanly(self):
        result = evolve_strategy({"2330": frame(300)}, warmup=260)
        self.assertEqual(result["status"], "insufficient_history")
        self.assertTrue(result["candidates"].empty)


if __name__ == "__main__":
    unittest.main()
