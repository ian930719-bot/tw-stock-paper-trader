import unittest
import numpy as np
import pandas as pd

from tsmc_2330_trend_atr_risk_research import apply_entry_filter


class TSMCTrendATRResearchTests(unittest.TestCase):
    def test_entry_filters_only_remove_low_entries(self):
        frame = pd.DataFrame({
            "distance_sma_200": [0.1, -0.1, 0.0],
            "sma50_slope_10": [0.01, 0.02, -0.01],
            "trend_stack": [1.0, 0.0, 1.0],
        })
        base = np.array([True, True, True])
        np.testing.assert_array_equal(
            apply_entry_filter(frame, base, "above_sma200"),
            [True, False, False],
        )
        np.testing.assert_array_equal(
            apply_entry_filter(frame, base, "positive_sma50_slope"),
            [True, True, False],
        )
        np.testing.assert_array_equal(
            apply_entry_filter(frame, base, "trend_stack"),
            [True, False, True],
        )
        np.testing.assert_array_equal(
            apply_entry_filter(frame, base, "none"),
            [True, True, True],
        )


if __name__ == "__main__":
    unittest.main()
