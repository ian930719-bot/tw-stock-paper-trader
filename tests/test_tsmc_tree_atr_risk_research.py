import unittest

from tsmc_2330_tree_atr_risk_research import score, ATR_STOPS


class TreeATRResearchTests(unittest.TestCase):
    def test_risk_adjusted_score_penalizes_drawdown(self):
        self.assertGreater(
            score({"cagr": 0.20, "max_drawdown": -0.10}),
            score({"cagr": 0.25, "max_drawdown": -0.30}),
        )

    def test_predeclared_atr_grid_contains_no_stop_and_multiple_widths(self):
        self.assertEqual(ATR_STOPS[0], None)
        self.assertIn(2.5, ATR_STOPS)
        self.assertIn(5.0, ATR_STOPS)
        self.assertEqual(len(ATR_STOPS), len(set(ATR_STOPS)))


if __name__ == "__main__":
    unittest.main()
