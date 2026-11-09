import unittest
from unittest.mock import patch

import pandas as pd

from collect_top10_adjusted_prices import (
    fetch_symbol, normalize, normalize_yahoo,
)


class AdjustedPriceNormalizationTests(unittest.TestCase):
    def test_finmind_adjusted_schema_is_normalized(self):
        rows = [{
            "date": "2024-01-02", "stock_id": "2330", "open": 100,
            "max": 102, "min": 99, "close": 101,
            "Trading_Volume": 123456, "Trading_money": 12000000,
        }]
        frame = normalize(rows, "2330")
        self.assertEqual(frame.loc[0, "high"], 102)
        self.assertEqual(frame.loc[0, "volume"], 123456)
        self.assertEqual(frame.loc[0, "name"], "台積電")
        self.assertEqual(frame.loc[0, "turnover"], 12000000)
        self.assertEqual(frame.loc[0, "provider"], "FinMind TaiwanStockPriceAdj")

    def test_yahoo_auto_adjust_schema_is_normalized_and_keeps_actions(self):
        history = pd.DataFrame({
            "Open": [95.0], "High": [97.0], "Low": [94.0], "Close": [96.0],
            "Volume": [1000], "Dividends": [1.0], "Stock Splits": [0.0],
        }, index=pd.DatetimeIndex(["2024-01-02"], name="Date"))
        frame = normalize_yahoo(history, "2330")
        self.assertEqual(frame.loc[0, "close"], 96.0)
        self.assertEqual(frame.loc[0, "dividends"], 1.0)
        self.assertEqual(frame.loc[0, "stock_splits"], 0.0)
        self.assertTrue(pd.isna(frame.loc[0, "turnover"]))
        self.assertEqual(frame.loc[0, "provider"], "Yahoo Finance auto_adjust=True")

    def test_auto_mode_falls_back_to_yahoo_when_finmind_fails(self):
        history = pd.DataFrame({
            "Open": [95.0], "High": [97.0], "Low": [94.0], "Close": [96.0],
            "Volume": [1000], "Dividends": [0.0], "Stock Splits": [0.0],
        }, index=pd.DatetimeIndex(["2024-01-02"], name="Date"))
        with patch("collect_top10_adjusted_prices.fetch_rows", side_effect=RuntimeError("FinMind quota")):
            with patch("collect_top10_adjusted_prices.fetch_yahoo_history", return_value=history):
                frame, source = fetch_symbol("2330", "2010-01-04", None, "configured-token", "auto")
        self.assertEqual(source, "Yahoo Finance auto_adjust=True")
        self.assertEqual(len(frame), 1)

    def test_finmind_only_mode_fails_instead_of_silently_switching_provider(self):
        with patch("collect_top10_adjusted_prices.fetch_rows", side_effect=RuntimeError("FinMind unavailable")):
            with self.assertRaisesRegex(RuntimeError, "FinMind unavailable"):
                fetch_symbol("2330", "2010-01-04", None, "configured-token", "finmind")

    def test_finmind_mode_requires_token(self):
        with self.assertRaisesRegex(RuntimeError, "FINMIND_TOKEN"):
            fetch_symbol("2330", "2010-01-04", None, "", "finmind")

    def test_invalid_ohlc_is_rejected(self):
        rows = [{
            "date": "2024-01-02", "stock_id": "2330", "open": 100,
            "max": 98, "min": 99, "close": 101, "Trading_Volume": 123456,
        }]
        with self.assertRaises(RuntimeError):
            normalize(rows, "2330")


if __name__ == "__main__":
    unittest.main()
