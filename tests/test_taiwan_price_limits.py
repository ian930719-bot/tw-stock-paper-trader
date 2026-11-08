import unittest
from datetime import date

from taiwan_price_limits import (
    PriceLimitUnavailable, _parse_rows, validate_limit_price,
)


class TaiwanPriceLimitTests(unittest.TestCase):
    def test_parse_twse_official_fields(self):
        rows = [{
            "證券代號": "2330",
            "本日漲停價": "1650.00",
            "本日開盤競價基準": "1500.00",
            "本日跌停價": "1350.00",
            "最近成交日": "2026/11/09",
        }]
        band = _parse_rows(rows, "2330", date(2026, 11, 9), "上市")
        self.assertEqual(band["reference_price"], 1500)
        self.assertEqual(band["upper_limit"], 1650)
        self.assertEqual(band["lower_limit"], 1350)

    def test_parse_tpex_next_day_official_fields(self):
        rows = [{
            "SecuritiesCompanyCode": "6488",
            "Date": "1151106",
            "NextDayReferencePrice": "100",
            "NextDayLimitUpPrice": "110",
            "NextDayLimitDownPrice": "90",
        }]
        band = _parse_rows(rows, "6488", date(2026, 11, 9), "上櫃")
        self.assertEqual(band["upper_limit"], 110)
        self.assertEqual(band["lower_limit"], 90)

    def test_tpex_no_limit_sentinel(self):
        rows = [{
            "SecuritiesCompanyCode": "00679B",
            "Date": "1151106",
            "NextDayReferencePrice": "26.9",
            "NextDayLimitUpPrice": "9999.95",
            "NextDayLimitDownPrice": "0.01",
        }]
        band = _parse_rows(rows, "00679B", date(2026, 11, 9), "上櫃")
        self.assertTrue(band["no_price_limit"])
        self.assertIsNone(band["upper_limit"])
        validate_limit_price(999.0, band)

    def test_tpex_rejects_same_day_next_session_limits(self):
        rows = [{
            "SecuritiesCompanyCode": "6488",
            "Date": "1151109",
            "NextDayReferencePrice": "100",
            "NextDayLimitUpPrice": "110",
            "NextDayLimitDownPrice": "90",
        }]
        with self.assertRaises(PriceLimitUnavailable):
            _parse_rows(rows, "6488", date(2026, 11, 9), "上櫃")

    def test_reject_price_outside_official_band(self):
        band = {"no_price_limit": False, "lower_limit": 90, "upper_limit": 110}
        validate_limit_price(100, band)
        with self.assertRaisesRegex(ValueError, "漲跌停範圍"):
            validate_limit_price(111, band)
        with self.assertRaisesRegex(ValueError, "漲跌停範圍"):
            validate_limit_price(89, band)

    def test_missing_or_stale_official_row_fails_closed(self):
        rows = [{
            "證券代號": "2330",
            "本日漲停價": "1650",
            "本日開盤競價基準": "1500",
            "本日跌停價": "1350",
            "資料日期": "2026/11/06",
        }]
        with self.assertRaises(PriceLimitUnavailable):
            _parse_rows(rows, "2330", date(2026, 11, 9), "上市")

    def test_unrecognized_special_security_fails_closed(self):
        rows = [{"證券代號": "2330", "本日漲停價": "1650"}]
        with self.assertRaises(PriceLimitUnavailable):
            _parse_rows(rows, "2330", date(2026, 11, 9), "上市")


if __name__ == "__main__":
    unittest.main()
