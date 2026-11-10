import unittest
import pandas as pd

from tsmc_2330_oracle_pivot_ceiling import pair_pivots, net_factor


class TSMCOraclePivotTests(unittest.TestCase):
    def test_pairs_low_then_next_high_only(self):
        features = pd.DataFrame({
            "corporate_action_window": [False] * 6,
        })
        events = pd.DataFrame([
            {"type": "HIGH", "pivot_index": 0, "confirmation_index": 1,
             "pivot_date": "2020-01-01", "pivot_price": 110.0, "confirmation_lag": 1},
            {"type": "LOW", "pivot_index": 2, "confirmation_index": 3,
             "pivot_date": "2020-01-03", "pivot_price": 90.0, "confirmation_lag": 1},
            {"type": "HIGH", "pivot_index": 4, "confirmation_index": 5,
             "pivot_date": "2020-01-05", "pivot_price": 120.0, "confirmation_lag": 1},
        ])
        pairs = pair_pivots(events, features)
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs.iloc[0]["gross_return"], 120 / 90 - 1)

    def test_pivot_inside_corporate_action_window_is_excluded(self):
        features = pd.DataFrame({
            "corporate_action_window": [False, True, False, False],
        })
        events = pd.DataFrame([
            {"type": "LOW", "pivot_index": 1, "confirmation_index": 2,
             "pivot_date": "2020-01-02", "pivot_price": 90.0, "confirmation_lag": 1},
            {"type": "HIGH", "pivot_index": 3, "confirmation_index": 3,
             "pivot_date": "2020-01-04", "pivot_price": 110.0, "confirmation_lag": 0},
        ])
        self.assertTrue(pair_pivots(events, features).empty)

    def test_cost_adjusted_factor_below_gross(self):
        gross = net_factor(0.10, 10)
        self.assertLess(gross, 1.10)
        self.assertGreater(gross, 1.08)


if __name__ == "__main__":
    unittest.main()
