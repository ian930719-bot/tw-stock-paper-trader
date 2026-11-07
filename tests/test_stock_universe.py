"""Tests for the Taiwan listed and OTC security master."""
import unittest
import pandas as pd

from stock_universe import StockUniverseError, normalize_stock_universe, universe_summary


class StockUniverseTests(unittest.TestCase):
    def test_latest_market_row_wins_after_transfer(self):
        records = [
            {"stock_id": "1234", "stock_name": "甲公司", "industry_category": "電子", "type": "emerging", "date": "2020-01-01"},
            {"stock_id": "1234", "stock_name": "甲公司", "industry_category": "電子", "type": "tpex", "date": "2022-05-01"},
            {"stock_id": "2330", "stock_name": "台積電", "industry_category": "半導體", "type": "twse", "date": "2026-11-06"},
        ]
        result = normalize_stock_universe(records)
        market = result.set_index("代號")["市場"].to_dict()
        self.assertEqual(market, {"1234": "上櫃", "2330": "上市"})

    def test_emerging_only_is_not_listed_or_otc(self):
        records = [
            {"stock_id": "5678", "stock_name": "興櫃公司", "industry_category": "生技", "type": "emerging", "date": "2026-11-06"},
        ]
        with self.assertRaises(StockUniverseError):
            normalize_stock_universe(records)

    def test_excludes_indices_and_invalid_codes(self):
        records = [
            {"stock_id": "TAIEX", "stock_name": "加權指數", "industry_category": "Index", "type": "twse", "date": "2026-11-06"},
            {"stock_id": "123", "stock_name": "短代號", "industry_category": "電子", "type": "twse", "date": "2026-11-06"},
            {"stock_id": "006208", "stock_name": "富邦台50", "industry_category": "ETF", "type": "twse", "date": "2026-11-06"},
            {"stock_id": "00679B", "stock_name": "元大美債20年", "industry_category": "ETF", "type": "twse", "date": "2026-11-06"},
        ]
        result = normalize_stock_universe(records)
        self.assertEqual(result["代號"].tolist(), ["006208", "00679B"])

    def test_market_summary_counts(self):
        df = pd.DataFrame([
            {"代號": "2330", "名稱": "台積電", "市場": "上市", "產業": "半導體", "市場資料日期": "2026-11-06"},
            {"代號": "6488", "名稱": "環球晶", "市場": "上櫃", "產業": "半導體", "市場資料日期": "2026-11-05"},
        ])
        summary = universe_summary(df)
        self.assertEqual(summary["總數"], 2)
        self.assertEqual(summary["上市"], 1)
        self.assertEqual(summary["上櫃"], 1)
        self.assertEqual(summary["資料日期"], "2026-11-06")

    def test_missing_schema_is_rejected(self):
        with self.assertRaises(StockUniverseError):
            normalize_stock_universe([{"stock_id": "2330"}])


if __name__ == "__main__":
    unittest.main()
