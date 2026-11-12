import unittest
import numpy as np
import pandas as pd

from tsmc_2330_zigzag_post_high_reentry_oof import choose_policy, eligible, WIN_FLOOR


class ZigZagPostHighReentryTests(unittest.TestCase):
    def test_win_floor_is_not_relaxed(self):
        self.assertFalse(eligible({
            "completed_trades": 8, "win_rate_completed_only": WIN_FLOOR - 0.001,
            "cagr_mark_to_market": 0.8,
        }))
        self.assertTrue(eligible({
            "completed_trades": 8, "win_rate_completed_only": WIN_FLOOR,
            "cagr_mark_to_market": 0.8,
        }))

    def test_minimum_sample_size_required(self):
        self.assertFalse(eligible({
            "completed_trades": 4, "win_rate_completed_only": 1.0,
            "cagr_mark_to_market": 0.8,
        }))

    def test_fallback_to_baseline_if_no_policy_qualifies(self):
        metrics = {
            "baseline": {"completed_trades": 2, "win_rate_completed_only": 1.0,
                         "cagr_mark_to_market": 0.2},
            "zigzag": {"completed_trades": 20, "win_rate_completed_only": 0.5,
                       "cagr_mark_to_market": 1.2},
        }
        self.assertEqual(choose_policy(metrics), "baseline")


if __name__ == "__main__":
    unittest.main()
