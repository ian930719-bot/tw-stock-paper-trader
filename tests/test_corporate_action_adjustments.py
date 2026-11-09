import unittest

import pandas as pd

from corporate_action_adjustments import (
    apply_back_adjustments,
    normalize_corporate_action_events,
)


class CorporateActionAdjustmentTests(unittest.TestCase):
    def test_cash_dividend_adjusts_prior_prices_without_mutating_raw_execution_bars(self):
        bars = pd.DataFrame({
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "open": [100.0, 100.0, 95.0],
            "high": [101.0, 101.0, 96.0],
            "low": [99.0, 99.0, 94.0],
            "close": [100.0, 100.0, 95.0],
            "volume": [1000, 1100, 1200],
        })
        events = normalize_corporate_action_events(pd.DataFrame([{
            "stock_id": "2330", "date": "2024-01-04",
            "before_price": 100.0, "reference_price": 95.0,
        }]))
        result = apply_back_adjustments(bars, events)
        self.assertAlmostEqual(result.loc[0, "adj_close"], 95.0)
        self.assertAlmostEqual(result.loc[1, "adj_close"], 95.0)
        self.assertAlmostEqual(result.loc[2, "adj_close"], 95.0)
        self.assertAlmostEqual(result.loc[0, "close"], 100.0)
        self.assertEqual(result.loc[2, "volume"], 1200)

    def test_split_adjustment_masks_volume_features_window(self):
        bars = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=25),
            "open": [100.0] * 10 + [50.0] * 15,
            "high": [101.0] * 10 + [51.0] * 15,
            "low": [99.0] * 10 + [49.0] * 15,
            "close": [100.0] * 10 + [50.0] * 15,
            "volume": [1000] * 25,
        })
        events = normalize_corporate_action_events(split_prices=pd.DataFrame([{
            "stock_id": "2330", "date": bars.loc[10, "date"],
            "before_price": 100.0, "after_price": 50.0,
        }]))
        result = apply_back_adjustments(bars, events)
        self.assertAlmostEqual(result.loc[9, "adj_close"], 50.0)
        self.assertAlmostEqual(result.loc[10, "adj_close"], 50.0)
        self.assertTrue(result.loc[10:24, "corporate_action_window"].all())

    def test_same_day_duplicate_sources_are_deduplicated_when_ratio_matches(self):
        dividend = pd.DataFrame([{
            "stock_id": "2330", "date": "2024-01-04",
            "before_price": 100.0, "reference_price": 95.0,
        }])
        split = pd.DataFrame([{
            "stock_id": "2330", "date": "2024-01-04",
            "before_price": 100.0, "after_price": 95.0,
        }])
        events = normalize_corporate_action_events(dividend, split)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events.iloc[0]["adjustment_ratio"], 0.95)
        self.assertEqual(events.iloc[0]["source_count"], 1)


if __name__ == "__main__":
    unittest.main()
