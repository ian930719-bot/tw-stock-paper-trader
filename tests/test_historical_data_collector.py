"""Tests for resumable historical data collector normalization."""
import unittest
from historical_data_collector import _normalize_bars, build_candidate_universe


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


    def test_candidate_universe_prioritizes_common_equities_and_excludes_etfs(self):
        import pandas as pd
        current = pd.DataFrame([
            {"stock_id": "0050", "stock_name": "元大台灣50", "type": "twse", "industry_category": "指數股票型基金(ETF)"},
            {"stock_id": "2330", "stock_name": "台積電", "type": "twse", "industry_category": "半導體業"},
            {"stock_id": "1101", "stock_name": "台泥", "type": "twse", "industry_category": "水泥工業"},
            {"stock_id": "006208", "stock_name": "富邦台50", "type": "twse", "industry_category": "指數股票型基金(ETF)"},
            {"stock_id": "6208", "stock_name": "日揚", "type": "tpex", "industry_category": "其他電子業"},
        ])
        delisted = pd.DataFrame([
            {"stock_id": "2384", "stock_name": "勝華", "date": "2015-07-07"},
            {"stock_id": "00732", "stock_name": "ETF", "date": "2023-04-21"},
        ])
        result = build_candidate_universe(current, delisted)
        self.assertEqual(result["stock_id"].tolist(), ["1101", "2330", "6208", "2384"])

    def test_empty_response_has_stable_schema(self):
        self.assertEqual(len(_normalize_bars([], "2330").columns), 8)


if __name__ == "__main__":
    unittest.main()
