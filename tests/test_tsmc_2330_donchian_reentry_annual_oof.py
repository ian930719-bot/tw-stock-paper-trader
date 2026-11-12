import unittest
import numpy as np
import pandas as pd

from tsmc_2330_donchian_reentry_annual_oof import (
    WIN_FLOOR, build_breakout_signals, choose_policy, eligible,
)


class DonchianReentryTests(unittest.TestCase):
    def test_breakout_signals_are_causal_and_require_trend_gate(self):
        n = 70
        close = np.linspace(100, 120, n)
        high = close + 1
        frame = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=n),
            "raw_high": high,
            "raw_close": close,
            "corporate_action_window": False,
            "distance_prior_high20": 0.01,
            "trend_stack": True,
            "adx14": 25.0,
            "plus_di14": 30.0,
            "minus_di14": 10.0,
            "distance_ema_20": 0.02,
            "relative_volume20": 1.2,
        })
        frame.loc[10, "corporate_action_window"] = True
        frame.loc[20, "trend_stack"] = False
        signals = build_breakout_signals(frame)
        self.assertEqual(len(signals["donchian20_reentry"]), n)
        self.assertFalse(signals["donchian20_reentry"][10])
        self.assertFalse(signals["donchian20_reentry"][20])
        self.assertFalse(signals["donchian55_reentry"][0])

    def test_win_floor_and_minimum_trades_are_mandatory(self):
        self.assertFalse(eligible({
            "completed_trades": 4, "win_rate_completed_only": 1.0,
            "cagr_mark_to_market": 0.5,
        }))
        self.assertFalse(eligible({
            "completed_trades": 10, "win_rate_completed_only": WIN_FLOOR - 0.01,
            "cagr_mark_to_market": 0.5,
        }))
        self.assertTrue(eligible({
            "completed_trades": 10, "win_rate_completed_only": WIN_FLOOR,
            "cagr_mark_to_market": 0.5,
        }))

    def test_policy_selection_falls_back_to_baseline(self):
        metrics = {
            "baseline": {"completed_trades": 2, "win_rate_completed_only": 1.0,
                         "cagr_mark_to_market": 0.2},
            "candidate": {"completed_trades": 5, "win_rate_completed_only": 0.8,
                          "cagr_mark_to_market": 0.9},
        }
        self.assertEqual(choose_policy(metrics), "baseline")


if __name__ == "__main__":
    unittest.main()
