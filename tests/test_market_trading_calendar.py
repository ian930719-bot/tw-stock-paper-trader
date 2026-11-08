import unittest

from market_trading_calendar import parse_tpex_holiday_html


class TpexCalendarTests(unittest.TestCase):
    def test_parse_roc_date_and_weekday_holiday(self):
        html = """
        <table>
          <tr><th>日期</th><th>名稱</th><th>說明</th></tr>
          <tr><td>115/01/01</td><td>中華民國開國紀念日</td><td>放假1日</td></tr>
          <tr><td>115/01/02</td><td>開始交易日</td><td>正常交易</td></tr>
          <tr><td>115/02/28</td><td>和平紀念日</td><td>放假1日</td></tr>
        </table>
        """
        closures, source = parse_tpex_holiday_html(html, 2026)
        self.assertEqual(closures, frozenset({"2026-01-01"}))
        self.assertIn("TPEx", source)

    def test_missing_year_fails_closed(self):
        html = "<table><tr><th>日期</th></tr><tr><td>114/01/01</td></tr></table>"
        with self.assertRaisesRegex(ValueError, "未找到"):
            parse_tpex_holiday_html(html, 2026)


if __name__ == "__main__":
    unittest.main()
