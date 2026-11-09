"""Tests for resumable historical data collector normalization."""
import unittest
import tempfile
from unittest.mock import patch
import pandas as pd
from historical_data_collector import _normalize_bars, build_candidate_universe, select_candidate_metadata, collect_history, collect_corporate_actions, save_universe_snapshot


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
        self.assertEqual(set(result["stock_id"]), {"1101", "2330", "6208", "2384"})


    def test_smoke_sample_is_industry_stratified_and_includes_delisted(self):
        import pandas as pd
        rows = []
        for i in range(1, 13):
            rows.append({
                "stock_id": str(1000 + i), "stock_name": f"Current {i}",
                "industry_category": f"Industry {i % 4}", "type": "twse",
                "candidate_source": "current_common_equity", "delisting_date": "",
            })
        for i in range(1, 6):
            rows.append({
                "stock_id": str(2000 + i), "stock_name": f"Delisted {i}",
                "industry_category": "", "type": "twse",
                "candidate_source": "delisted_common_equity_candidate", "delisting_date": "2018-01-01",
            })
        candidates = pd.DataFrame(rows)
        picked = select_candidate_metadata(candidates, 10, seed=7)
        self.assertEqual(len(picked), 10)
        self.assertGreaterEqual((picked["candidate_source"] == "delisted_common_equity_candidate").sum(), 1)
        self.assertGreaterEqual(picked.loc[picked["candidate_source"] == "current_common_equity", "industry_category"].nunique(), 3)

    def test_unlimited_candidate_selection_keeps_every_candidate(self):
        import pandas as pd
        candidates = pd.DataFrame({"stock_id": ["1101", "2330"], "candidate_source": ["current_common_equity"] * 2})
        self.assertEqual(len(select_candidate_metadata(candidates, None)), 2)


    def test_full_candidate_universe_is_frozen_across_resume_runs(self):
        current = pd.DataFrame([
            {"stock_id": "1101", "stock_name": "台泥", "industry_category": "水泥工業", "type": "twse"},
            {"stock_id": "2330", "stock_name": "台積電", "industry_category": "半導體業", "type": "twse"},
        ])
        delisted = pd.DataFrame(columns=["stock_id", "stock_name", "date"])
        bars = [{
            "date": "2024-01-02", "open": 50, "max": 51, "min": 49,
            "close": 50.5, "Trading_Volume": 1000, "Trading_money": 50000,
        }]
        with tempfile.TemporaryDirectory() as tmp:
            with patch("historical_data_collector.discover_symbols", return_value=(current, delisted)) as discover, \
                 patch("historical_data_collector._api_get", return_value=bars):
                first = collect_history(tmp, "2010-01-04", token="test-token", pause_seconds=0)
                self.assertTrue(first["candidate_universe_frozen"])
                self.assertEqual(first["requested_symbols"], 2)
                self.assertTrue(first["collection_complete"])
                second = collect_history(tmp, "2010-01-04", token="test-token", pause_seconds=0)
                self.assertEqual(second["requested_symbols"], 2)
                self.assertEqual(discover.call_count, 1)
                self.assertTrue(second["candidate_universe_frozen"])

    def test_empty_cached_symbol_is_not_downloaded_again(self):
        current = pd.DataFrame([
            {"stock_id": "2330", "stock_name": "台積電", "industry_category": "半導體業", "type": "twse"},
        ])
        delisted = pd.DataFrame(columns=["stock_id", "stock_name", "date"])
        with tempfile.TemporaryDirectory() as tmp:
            with patch("historical_data_collector.discover_symbols", return_value=(current, delisted)), \
                 patch("historical_data_collector._api_get", return_value=[]) as api_get:
                first = collect_history(tmp, "2010-01-04", token="test-token", pause_seconds=0)
                self.assertEqual(first["empty_symbols"], 1)
                self.assertTrue(first["collection_complete"])
                second = collect_history(tmp, "2010-01-04", token="test-token", pause_seconds=0)
                self.assertEqual(second["empty_symbols"], 1)
                self.assertTrue(second["collection_complete"])
                self.assertEqual(api_get.call_count, 1)

    def test_failed_symbol_is_not_marked_complete(self):
        current = pd.DataFrame([
            {"stock_id": "2330", "stock_name": "台積電", "industry_category": "半導體業", "type": "twse"},
        ])
        delisted = pd.DataFrame(columns=["stock_id", "stock_name", "date"])
        with tempfile.TemporaryDirectory() as tmp:
            with patch("historical_data_collector.discover_symbols", return_value=(current, delisted)), \
                 patch("historical_data_collector._api_get", side_effect=RuntimeError("temporary API failure")):
                result = collect_history(tmp, "2010-01-04", token="test-token", pause_seconds=0)
                self.assertEqual(result["failed_symbols"], 1)
                self.assertEqual(result["remaining_symbols"], 1)
                self.assertFalse(result["collection_complete"])

    def test_empty_response_has_stable_schema(self):
        self.assertEqual(len(_normalize_bars([], "2330").columns), 8)


    def test_corporate_action_collection_is_resumable_and_writes_adjustment_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame({"symbol": ["2330"], "name": ["台積電"]}).to_csv(
                f"{tmp}/candidate_symbols.csv", index=False
            )
            def fake_api(params, **kwargs):
                if params["dataset"] == "TaiwanStockDividendResult":
                    return [{
                        "date": "2024-01-04", "stock_id": "2330",
                        "before_price": 100, "reference_price": 95,
                    }]
                return []
            with patch("historical_data_collector._api_get", side_effect=fake_api) as api_get:
                sleep_patch = patch("historical_data_collector.time.sleep")
                sleep_patch.start()
                first = collect_corporate_actions(tmp, "2010-01-04", token="test-token", pause_seconds=0)
                self.assertTrue(first["corporate_actions_complete"])
                self.assertEqual(first["event_rows"], 1)
                self.assertEqual(api_get.call_count, 4)
                second = collect_corporate_actions(tmp, "2010-01-04", token="test-token", pause_seconds=0)
                self.assertTrue(second["corporate_actions_complete"])
                self.assertEqual(api_get.call_count, 4)
                sleep_patch.stop()
            events = pd.read_csv(f"{tmp}/corporate_actions/events.csv")
            self.assertAlmostEqual(events.iloc[0]["adjustment_ratio"], 0.95)


    def test_daily_universe_snapshot_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = pd.DataFrame([{"stock_id": "2330", "type": "twse"}])
            second = pd.DataFrame([{"stock_id": "1101", "type": "twse"}])
            a = save_universe_snapshot(tmp, first)
            b = save_universe_snapshot(tmp, second)
            self.assertEqual(a["snapshot_sha256"], b["snapshot_sha256"])
            saved = pd.read_csv(f"{tmp}/{a['snapshot_path']}", dtype={"stock_id": str})
            self.assertEqual(saved["stock_id"].tolist(), ["2330"])


if __name__ == "__main__":
    unittest.main()
