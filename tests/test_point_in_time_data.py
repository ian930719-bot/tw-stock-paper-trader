import unittest
import pandas as pd

from point_in_time_data import normalize_point_in_time_records, point_in_time_snapshot, assert_no_future_records


class PointInTimeDataTests(unittest.TestCase):
    def test_uses_publication_date_not_fiscal_period(self):
        raw = pd.DataFrame([
            {"stock_id": "2330", "date": "2024-12-31", "announcement_date": "2025-03-15", "EPS": 10},
            {"stock_id": "2330", "date": "2025-03-31", "announcement_date": "2025-05-10", "EPS": 12},
        ])
        normalized = normalize_point_in_time_records(raw, source="fixture")
        self.assertEqual(point_in_time_snapshot(normalized, "2025-04-01").iloc[0]["EPS"], 10)
        self.assertEqual(point_in_time_snapshot(normalized, "2025-06-01").iloc[0]["EPS"], 12)

    def test_rejects_period_only_data(self):
        with self.assertRaises(ValueError):
            normalize_point_in_time_records(pd.DataFrame([
                {"stock_id": "2330", "date": "2024-12-31", "EPS": 10}
            ]))

    def test_no_future_data_assertion(self):
        data = normalize_point_in_time_records(pd.DataFrame([
            {"stock_id": "2330", "announcement_date": "2025-05-10", "EPS": 12}
        ]))
        with self.assertRaises(ValueError):
            assert_no_future_records(data, "2025-05-01")

    def test_missing_announcement_rows_are_dropped(self):
        data = normalize_point_in_time_records(pd.DataFrame([
            {"stock_id": "2330", "announcement_date": None, "EPS": 12},
            {"stock_id": "2330", "announcement_date": "2025-05-10", "EPS": 13},
        ]))
        self.assertEqual(len(data), 1)


if __name__ == "__main__":
    unittest.main()
