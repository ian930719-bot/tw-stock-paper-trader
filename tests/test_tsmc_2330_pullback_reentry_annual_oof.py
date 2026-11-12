import unittest
import numpy as np
import pandas as pd

from tsmc_2330_pullback_reentry_annual_oof import (
    WIN_FLOOR, build_pullback_reentry_signals, choose_policy, eligible,
)


class PullbackReentryTests(unittest.TestCase):
    def frame(self, n=80):
        return pd.DataFrame({
            "raw_open": np.full(n, 100.0),
            "raw_high": np.full(n, 101.0),
            "raw_low": np.full(n, 99.0),
            "raw_close": np.full(n, 100.0),
            "corporate_action_window": False,
            "distance_ema_20": np.full(n, 0.01),
            "distance_sma_50": np.full(n, 0.02),
            "ema20_slope_5": np.full(n, 0.01),
            "sma50_slope_10": np.full(n, 0.01),
            "rsi14": np.full(n, 55.0),
            "adx14": np.full(n, 25.0),
            "plus_di14": np.full(n, 30.0),
            "minus_di14": np.full(n, 10.0),
            "relative_volume20": np.full(n, 1.2),
        })

    def test_pullback_recovery_requires_prior_pullback_and_break(self):
        d = self.frame()
        d.loc[74, "distance_ema_20"] = -0.01
        d.loc[79, "raw_close"] = 102.0
        d.loc[79, "raw_high"] = 103.0
        d.loc[78, "raw_high"] = 101.0
        signals = build_pullback_reentry_signals(d)
        self.assertTrue(signals["pullback_recovery"][79])
        self.assertFalse(signals["pullback_recovery"][70])

    def test_corporate_action_window_blocks_both_signals(self):
        d = self.frame()
        d.loc[79, "corporate_action_window"] = True
        signals = build_pullback_reentry_signals(d)
        self.assertFalse(signals["pullback_recovery"][79])
        self.assertFalse(signals["higher_low_break"][79])

    def test_selection_preserves_win_floor(self):
        self.assertFalse(eligible({
            "completed_trades": 4, "win_rate_completed_only": 1.0,
            "cagr_mark_to_market": 0.5,
        }))
        self.assertFalse(eligible({
            "completed_trades": 10, "win_rate_completed_only": WIN_FLOOR - 0.01,
            "cagr_mark_to_market": 0.5,
        }))
        self.assertEqual(choose_policy({
            "baseline": {"completed_trades": 2, "win_rate_completed_only": 1.0, "cagr_mark_to_market": 0.2}
        }), "baseline")


if __name__ == "__main__":
    unittest.main()
