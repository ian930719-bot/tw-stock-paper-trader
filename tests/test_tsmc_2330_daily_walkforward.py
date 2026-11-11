import unittest
import numpy as np
from tsmc_2330_daily_walkforward import make_labels, FEATURES, LEADS, INSTITUTIONAL_START

class DailyWalkForwardTests(unittest.TestCase):
    def test_forward_labels_exclude_pivot_day(self):
        y, known = make_labels(20, [10], 3)
        self.assertEqual(np.flatnonzero(y).tolist(), [7, 8, 9])
        self.assertEqual(int(y[10]), 0)
        self.assertTrue(known[16])
        self.assertFalse(known[17])

    def test_only_foreign_and_trust_flow_features(self):
        for direction in FEATURES.values():
            for name, cols in direction.items():
                self.assertIn(name, ("technical", "technical_plus_institutional"))
                self.assertFalse(any("dealer" in c for c in cols))
            cols = direction["technical_plus_institutional"]
            self.assertTrue(any(c.startswith("foreign_") for c in cols))
            self.assertTrue(any(c.startswith("trust_") for c in cols))

    def test_horizons_and_institutional_coverage(self):
        self.assertEqual(LEADS, (1, 3, 5, 10, 20))
        self.assertEqual(str(INSTITUTIONAL_START.date()), "2012-05-02")

if __name__ == "__main__":
    unittest.main()
