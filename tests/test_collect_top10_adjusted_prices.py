import unittest
from collect_top10_adjusted_prices import normalize


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

    def test_invalid_ohlc_is_rejected(self):
        rows = [{
            "date": "2024-01-02", "stock_id": "2330", "open": 100,
            "max": 98, "min": 99, "close": 101,
            "Trading_Volume": 123456,
        }]
        with self.assertRaises(RuntimeError):
            normalize(rows, "2330")


if __name__ == "__main__":
    unittest.main()
