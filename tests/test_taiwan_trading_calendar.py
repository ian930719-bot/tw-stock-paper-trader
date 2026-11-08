import unittest
from datetime import date

from taiwan_trading_calendar import (
    TradingCalendarUnavailable,
    parse_calendar_payload,
    trading_day_status,
)


class TradingCalendarTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "stat": "ok",
            "date": "20260101",
            "data": [
                ["2026-01-01", "中華民國開國紀念日", "依規定放假1日。"],
                ["2026-01-02", "國曆新年開始交易日", "國曆新年開始交易。"],
                ["2026-02-12", "市場無交易，僅辦理結算交割作業", ""],
                ["2026-02-20", "農曆除夕及春節", "於2月20日（星期五）補假。"],
                ["2026-02-23", "農曆春節後開始交易日", "農曆春節後開始交易。"],
            ],
        }

    def test_parse_only_weekday_closure_rows(self):
        closures, source_date = parse_calendar_payload(self.payload, 2026)
        self.assertEqual(source_date, "20260101")
        self.assertEqual(
            closures,
            frozenset({"2026-01-01", "2026-02-12", "2026-02-20"}),
        )

    def test_open_weekday_uses_official_calendar(self):
        def fetcher(year):
            self.assertEqual(year, 2026)
            return frozenset({"2026-02-12"}), "20260101"

        is_open, note = trading_day_status(date(2026, 2, 23), fetcher=fetcher)
        self.assertTrue(is_open)
        self.assertIn("TWSE", note)

    def test_official_weekday_closure(self):
        fetcher = lambda year: (frozenset({"2026-02-12"}), "20260101")
        is_open, note = trading_day_status(date(2026, 2, 12), fetcher=fetcher)
        self.assertFalse(is_open)
        self.assertIn("休市", note)

    def test_weekend_does_not_require_network(self):
        def fail(year):
            raise AssertionError("weekend should not fetch calendar")
        is_open, note = trading_day_status(date(2026, 2, 21), fetcher=fail)
        self.assertFalse(is_open)
        self.assertEqual(note, "週末休市")

    def test_unavailable_calendar_fails_closed(self):
        def fail(year):
            raise TradingCalendarUnavailable("offline")
        is_open, note = trading_day_status(date(2026, 2, 23), fetcher=fail)
        self.assertFalse(is_open)
        self.assertIn("停止委託", note)

    def test_invalid_payload_rejected(self):
        with self.assertRaises(TradingCalendarUnavailable):
            parse_calendar_payload({"stat": "error", "data": []}, 2026)


if __name__ == "__main__":
    unittest.main()
