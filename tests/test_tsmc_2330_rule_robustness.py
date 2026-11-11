import unittest
import pandas as pd

from tsmc_2330_rule_robustness import build_candidate_signals, simulate


class RuleRobustnessTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "raw_open": [100.0, 101.0, 99.0, 110.0],
            "raw_close": [100.0, 100.0, 105.0, 112.0],
            "relative_volume20": [1.3, 1.0, 1.0, 1.0],
            "return_1": [-0.02, 0.0, 0.02, 0.0],
            "stoch_k14": [20.0, 50.0, 50.0, 50.0],
            "close_location": [0.1, 0.5, 0.9, 0.5],
            "obv_flow_ratio_5": [-0.2, 0.0, 0.0, 0.0],
            "intraday_range_pct": [0.02, 0.01, 0.02, 0.01],
            "distance_sma_200": [0.12, 0.05, 0.12, 0.05],
            "macd_hist_change_3": [0.0, 0.0, -0.001, 0.0],
            "sma50_slope_10": [0.01, 0.01, 0.01, 0.01],
            "corporate_action_window": [False, False, False, False],
        })

    def test_frozen_rule_identifies_capitulation_and_climax(self):
        d = self.frame()
        low, high = build_candidate_signals(d)
        self.assertEqual(low.tolist(), [True, False, False, False])
        self.assertEqual(high.tolist(), [False, False, True, False])

    def test_corporate_action_window_suppresses_signals(self):
        d = self.frame()
        d.loc[0, "corporate_action_window"] = True
        low, high = build_candidate_signals(d)
        self.assertFalse(bool(low.iloc[0]))

    def test_next_open_execution_and_all_out_exit(self):
        d = self.frame()
        low = pd.Series([True, False, False, False])
        high = pd.Series([False, False, True, False])
        metrics, trades, curve = simulate(d, low, high)
        self.assertEqual(metrics["closed_trades"], 1)
        self.assertEqual(trades.iloc[0]["entry_date"], "2024-01-03")
        self.assertEqual(trades.iloc[0]["exit_date"], "2024-01-05")
        self.assertEqual(int(curve.iloc[-1]["shares"]), 0)

    def test_delay_does_not_use_future_signal(self):
        d = self.frame()
        low = pd.Series([True, False, False, False])
        high = pd.Series([False, False, True, False])
        metrics, trades, _ = simulate(d, low, high, delay_sessions=1)
        self.assertEqual(metrics["closed_trades"], 0)


if __name__ == "__main__":
    unittest.main()
