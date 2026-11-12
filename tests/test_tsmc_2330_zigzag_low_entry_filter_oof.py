import unittest
import numpy as np

from tsmc_2330_zigzag_low_entry_filter_oof import (
    causal_recent_base_low, choose_policy, eligible, WIN_FLOOR,
)


class ZigZagLowEntryFilterTests(unittest.TestCase):
    def test_confluence_uses_only_current_and_prior_base_low(self):
        base = np.array([False, True, False, False, False, False, False])
        pivot = np.array([False, False, False, False, True, False, True])
        out = causal_recent_base_low(base, pivot, 3)
        self.assertTrue(out[4])
        self.assertFalse(out[6])

    def test_win_floor_and_sample_floor(self):
        self.assertFalse(eligible({
            "completed_trades": 10, "win_rate_completed_only": WIN_FLOOR - 1e-5,
            "cagr_mark_to_market": 1.0,
        }))
        self.assertFalse(eligible({
            "completed_trades": 4, "win_rate_completed_only": 1.0,
            "cagr_mark_to_market": 1.0,
        }))
        self.assertTrue(eligible({
            "completed_trades": 5, "win_rate_completed_only": WIN_FLOOR,
            "cagr_mark_to_market": 0.1,
        }))

    def test_fallback_baseline(self):
        metrics = {
            "baseline": {"completed_trades": 3, "win_rate_completed_only": 1.0,
                         "cagr_mark_to_market": 0.2},
            "candidate": {"completed_trades": 10, "win_rate_completed_only": 0.6,
                          "cagr_mark_to_market": 1.2},
        }
        self.assertEqual(choose_policy(metrics), "baseline")


if __name__ == "__main__":
    unittest.main()
