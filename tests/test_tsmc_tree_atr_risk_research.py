import unittest

import numpy as np
import pandas as pd

from tsmc_2330_tree_atr_risk_research import score, ATR_STOPS, build_atr_regime_masks


class TreeATRResearchTests(unittest.TestCase):
    def test_risk_adjusted_score_penalizes_drawdown(self):
        self.assertGreater(
            score({"cagr": 0.20, "max_drawdown": -0.10}),
            score({"cagr": 0.25, "max_drawdown": -0.30}),
        )

    def test_regime_masks_use_only_current_close_indicators(self):
        frame = pd.DataFrame({
            "distance_sma_200": [-0.1, 0.0, 0.1, np.nan],
            "sma50_slope_10": [-0.2, -0.1, 0.1, -0.1],
        })
        masks = build_atr_regime_masks(frame)
        np.testing.assert_array_equal(masks["below_sma200"], [True, False, False, False])
        np.testing.assert_array_equal(masks["above_sma200"], [False, True, True, False])
        np.testing.assert_array_equal(masks["negative_sma50_slope"], [True, True, False, True])
        np.testing.assert_array_equal(
            masks["below_sma200_and_negative_sma50_slope"], [True, False, False, False]
        )

    def test_predeclared_atr_grid_contains_no_stop_and_multiple_widths(self):
        self.assertEqual(ATR_STOPS[0], None)
        self.assertIn(2.5, ATR_STOPS)
        self.assertIn(5.0, ATR_STOPS)
        self.assertEqual(len(ATR_STOPS), len(set(ATR_STOPS)))


if __name__ == "__main__":
    unittest.main()
