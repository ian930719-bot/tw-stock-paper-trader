"""Tests for conservative Taiwan quote freshness labels."""
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from market_data import quote_freshness

TZ = ZoneInfo("Asia/Taipei")


class QuoteFreshnessTests(unittest.TestCase):
    def test_recent_quote_during_session(self):
        now = datetime(2026, 11, 4, 10, 5, tzinfo=TZ)
        label, age = quote_freshness("2026-11-04T10:00:00", now=now)
        self.assertEqual(label, "盤中更新")
        self.assertEqual(age, 5.0)

    def test_old_quote_during_session_is_flagged(self):
        now = datetime(2026, 11, 4, 10, 5, tzinfo=TZ)
        label, _ = quote_freshness("2026-11-04T09:30:00", now=now)
        self.assertEqual(label, "盤中延遲／可能休市")

    def test_same_day_quote_after_close_is_not_called_live(self):
        now = datetime(2026, 11, 4, 22, 40, tzinfo=TZ)
        label, _ = quote_freshness("2026-11-04T13:24:00", now=now)
        self.assertEqual(label, "當日最後報價（非盤中）")

    def test_previous_date_quote_is_flagged(self):
        now = datetime(2026, 11, 4, 10, 5, tzinfo=TZ)
        label, _ = quote_freshness("2026-11-03T13:24:00", now=now)
        self.assertEqual(label, "非當日報價")

    def test_missing_or_invalid_quote(self):
        self.assertEqual(quote_freshness("無報價")[0], "無報價")
        self.assertEqual(quote_freshness("not-a-date")[0], "時間異常")

    def test_future_timestamp_is_flagged(self):
        now = datetime(2026, 11, 4, 10, 5, tzinfo=TZ)
        label, _ = quote_freshness("2026-11-04T10:20:00", now=now)
        self.assertEqual(label, "時間異常")


if __name__ == "__main__":
    unittest.main()
