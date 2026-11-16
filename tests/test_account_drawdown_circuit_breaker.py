import unittest
import numpy as np
import pandas as pd

from tsmc_2330_execution_backtest import _simulate_period


class AccountDrawdownCircuitBreakerTests(unittest.TestCase):
    def test_stop_executes_next_open_and_reentry_waits_for_cooldown_signal(self):
        dates = pd.date_range("2024-01-02", periods=7, freq="B")
        bars = pd.DataFrame({
            "date": dates,
            "raw_open": [100, 100, 100, 90, 92, 95, 97],
            "raw_high": [101, 101, 101, 92, 94, 97, 99],
            "raw_low": [99, 99, 89, 89, 91, 94, 96],
            "raw_close": [100, 100, 90, 92, 94, 96, 98],
            "corporate_action_window": [False] * 7,
        })
        limits = {d.strftime("%Y-%m-%d"): (120.0, 80.0) for d in dates}
        low = np.array([True, False, False, False, False, False, False])
        high = np.zeros(7, dtype=bool)
        reentry = np.array([False, False, False, False, True, False, False])
        result = _simulate_period(
            bars, low, high, limits, {},
            "2024-01-02", "2024-01-10", 0, 100000.0, True,
            account_drawdown_stop_pct=0.05,
            account_reentry_signal=reentry,
            account_reentry_cooldown=1,
        )
        self.assertEqual(result["account_drawdown_stop_count"], 1)
        self.assertEqual(result["account_drawdown_reentry_count"], 1)
        self.assertEqual(len(result["trades"]), 2)
        self.assertEqual(result["trades"][0]["entry_date"], "2024-01-03")
        self.assertEqual(result["trades"][0]["exit_date"], "2024-01-05")
        self.assertEqual(result["trades"][1]["entry_date"], "2024-01-09")

    def test_account_overlay_is_opt_in_and_default_has_no_stops(self):
        dates = pd.date_range("2024-02-01", periods=4, freq="B")
        bars = pd.DataFrame({
            "date": dates,
            "raw_open": [100, 100, 90, 90],
            "raw_high": [101, 101, 91, 91],
            "raw_low": [99, 89, 89, 89],
            "raw_close": [100, 90, 90, 90],
            "corporate_action_window": [False] * 4,
        })
        limits = {d.strftime("%Y-%m-%d"): (120.0, 80.0) for d in dates}
        low = np.array([True, False, False, False])
        high = np.zeros(4, dtype=bool)
        baseline = _simulate_period(
            bars, low, high, limits, {}, "2024-02-01", "2024-02-06",
            0, 100000.0, True,
        )
        self.assertEqual(baseline["account_drawdown_stop_count"], 0)
        self.assertEqual(baseline["account_drawdown_reentry_count"], 0)


if __name__ == "__main__":
    unittest.main()
