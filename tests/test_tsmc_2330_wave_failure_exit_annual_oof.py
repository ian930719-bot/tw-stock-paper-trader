import unittest
import numpy as np
import pandas as pd

from tsmc_2330_wave_failure_exit_annual_oof import (
    WIN_FLOOR, build_wave_failure_signals, choose_policy, is_eligible,
)


class WaveFailureExitTests(unittest.TestCase):
    def frame(self, n=80):
        return pd.DataFrame({
            "raw_close": np.linspace(100, 90, n),
            "raw_low": np.linspace(99, 89, n),
            "corporate_action_window": False,
            "distance_prior_low20": 0.01,
            "distance_sma_50": -0.01,
            "distance_ema_50": -0.01,
            "ema50_slope_10": -0.01,
            "ema20_slope_5": -0.01,
            "distance_ema_20": -0.01,
            "rsi14": 40.0,
            "atr_pct14": 0.02,
            "close_location": 0.1,
        })

    def test_each_exit_family_is_causal_and_action_masked(self):
        d = self.frame()
        d.loc[30, "distance_prior_low20"] = -0.02
        d.loc[31, "corporate_action_window"] = True
        d.loc[40, "distance_prior_low20"] = 0.01
        d.loc[40, "distance_ema_50"] = -0.02
        d.loc[50, "distance_ema_20"] = -0.02
        d.loc[49, "distance_ema_20"] = -0.02
        d.loc[60, "raw_close"] = 80
        signals = build_wave_failure_signals(d)
        self.assertTrue(signals["structure_break"][30])
        self.assertFalse(signals["structure_break"][31])
        self.assertTrue(signals["ema50_failure"][40])
        self.assertTrue(signals["two_close_ema20_failure"][50])
        self.assertTrue(signals["atr_shock_weak_trend"][60])

    def test_win_floor_and_trade_minimum(self):
        self.assertFalse(is_eligible({
            "completed_trades": 4, "win_rate_completed_only": 1.0,
            "cagr_mark_to_market": 0.5,
        }))
        self.assertFalse(is_eligible({
            "completed_trades": 10, "win_rate_completed_only": WIN_FLOOR - 0.001,
            "cagr_mark_to_market": 0.5,
        }))
        self.assertTrue(is_eligible({
            "completed_trades": 10, "win_rate_completed_only": WIN_FLOOR,
            "cagr_mark_to_market": 0.5,
        }))

    def test_no_eligible_candidate_falls_back_to_baseline(self):
        metrics = {
            "baseline": {"completed_trades": 2, "win_rate_completed_only": 1.0,
                         "cagr_mark_to_market": 0.2},
            "risk_exit": {"completed_trades": 10, "win_rate_completed_only": 0.8,
                          "cagr_mark_to_market": 0.9},
        }
        self.assertEqual(choose_policy(metrics), "baseline")


if __name__ == "__main__":
    unittest.main()
