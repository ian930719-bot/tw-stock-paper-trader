"""Tests for resumable historical data collector normalization."""
import unittest
from historical_data_collector import _normalize_bars


class HistoricalCollectorTests(unittest.TestCase):
    def test_normalizes_finmind_daily_rows_to_raw_panel_schema(self):
        rows = [{
            "date": "2024-01-02", "stock_id": "2330", "open": 590,
            "max": 600, "min": 585, "close": 598,
            "Trading_Volume": 123456, "Trading_money": 73500000,
        }]
        result = _normalize_bars(rows, "2330")
        self.assertEqual(result.columns.tolist(), [
            "symbol", "date", "open", "high", "low", "close", "volume", "turnover"
        ])
        self.assertEqual(result.iloc[0]["symbol"], "2330")
        self.assertEqual(float(result.iloc[0]["high"]), 600)
        self.assertEqual(float(result.iloc[0]["volume"]), 123456)

    def test_drops_invalid_ohlc_rows(self):
        rows = [{
            "date": "2024-01-02", "open": 590, "max": 580, "min": 585,
            "close": 598, "Trading_Volume": 123456,
        }]
        self.assertTrue(_normalize_bars(rows, "2330").empty)

    def test_empty_response_has_stable_schema(self):
        self.assertEqual(len(_normalize_bars([], "2330").columns), 8)


if __name__ == "__main__":
    unittest.main()
