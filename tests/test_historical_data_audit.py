"""Tests for offline historical-data audit and FinMind HTTP retry semantics."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from historical_data_collector import _api_get
from historical_data_audit import audit_dataset


class FinMindHttpErrorTests(unittest.TestCase):
    @patch("historical_data_collector.requests.get")
    def test_permanent_http_400_is_not_retried_and_body_is_reported(self, get):
        response = Mock()
        response.status_code = 400
        response.text = '{"msg":"invalid dataset parameters"}'
        error = requests.HTTPError("400 Client Error")
        error.response = response
        response.raise_for_status.side_effect = error
        get.return_value = response
        with self.assertRaisesRegex(RuntimeError, "HTTP 400") as caught:
            _api_get({"dataset": "BadDataset"}, token="secret-token", retries=3)
        self.assertIn("invalid dataset parameters", str(caught.exception))
        self.assertEqual(get.call_count, 1)
        self.assertNotIn("secret-token", str(caught.exception))


class HistoricalDataAuditTests(unittest.TestCase):
    def test_audit_writes_coverage_quality_integrity_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bars = root / "bars"
            bars.mkdir()
            pd.DataFrame([
                {"symbol":"2330","date":"2024-01-02","open":100,"high":105,"low":99,"close":104,"volume":1000},
                {"symbol":"2330","date":"2024-01-02","open":100,"high":105,"low":99,"close":104,"volume":1000},
                {"symbol":"2330","date":"2024-01-03","open":100,"high":98,"low":99,"close":97,"volume":1000},
            ]).to_csv(bars / "2330.csv", index=False)
            pd.DataFrame([{"symbol":"2330","candidate_source":"current_common_equity","delisting_date":""},
                          {"symbol":"3099","candidate_source":"delisted_common_equity_candidate","delisting_date":"2010-04-12"}]).to_csv(root / "candidate_symbols.csv", index=False)
            digest = hashlib.sha256((bars / "2330.csv").read_bytes()).hexdigest()
            (root / "download_manifest.json").write_text(json.dumps({
                "data_files_sha256":{"2330.csv":digest},
                "status_rows":[{"symbol":"2330","status":"cached"}],
                "point_in_time_universe_complete":False
            }), encoding="utf-8")
            result = audit_dataset(root)
            self.assertEqual(result["candidate_count"], 2)
            self.assertEqual(result["successful_symbols"], 1)
            self.assertEqual(result["failed_symbols"], 1)
            self.assertEqual(result["valid_ohlcv_rows"], 1)
            self.assertEqual(result["duplicate_valid_date_rows"], 1)
            self.assertEqual(result["hash_mismatches"], 0)
            self.assertEqual(result["missing_price_files"], 1)
            self.assertFalse(result["research_grade_eligible"])
            for name in ("candidate_coverage.csv","data_quality.csv","manifest_integrity.csv","data_audit_summary.json"):
                self.assertTrue((root / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
