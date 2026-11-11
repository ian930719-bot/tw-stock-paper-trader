import unittest
import pandas as pd

from annual_top20_shared_capital_research import (
    _valid_bars, build_annual_universe, simulate,
)


class AnnualTop20ResearchTests(unittest.TestCase):
    def test_universe_uses_prior_calendar_year_turnover(self):
        dates_2015 = pd.bdate_range("2015-01-01", periods=110)
        dates_2016 = pd.bdate_range("2016-01-01", periods=110)
        frames = {}
        for symbol, old_turnover, current_turnover in (
            ("A", 300.0, 10.0), ("B", 200.0, 500.0), ("C", 100.0, 100.0)
        ):
            dates = dates_2015.append(dates_2016)
            turnover = [old_turnover] * 110 + [current_turnover] * 110
            frames[symbol] = pd.DataFrame({
                "date": dates, "turnover": turnover,
            })
        universe = build_annual_universe(frames)
        selected_2016 = set(universe.loc[universe.year == 2016, "symbol"])
        selected_2017 = set(universe.loc[universe.year == 2017, "symbol"])
        self.assertEqual(selected_2016, {"A", "B", "C"})
        self.assertEqual(selected_2017, {"A", "B", "C"})
        rank16 = universe.loc[universe.year == 2016].set_index("symbol")["rank"]
        rank17 = universe.loc[universe.year == 2017].set_index("symbol")["rank"]
        self.assertLess(rank16["A"], rank16["B"])
        self.assertLess(rank17["B"], rank17["A"])

    def test_invalid_ohlc_rows_are_rejected(self):
        frame = pd.DataFrame({
            "date": ["2016-01-04", "2016-01-05"],
            "open": [10.0, 10.0], "high": [11.0, 9.0],
            "low": [9.0, 8.0], "close": [10.5, 9.5],
            "volume": [1000, 1000], "turnover": [10000, 10000],
        })
        valid = _valid_bars(frame, "TEST")
        self.assertEqual(len(valid), 1)
        self.assertEqual(float(valid.iloc[0]["high"]), 11.0)

    def test_entry_is_filled_no_earlier_than_next_session_open(self):
        dates = pd.to_datetime(["2016-01-04", "2016-01-05", "2016-01-06", "2016-01-07"])
        frame = pd.DataFrame({
            "date": dates, "open": [10.0, 10.0, 11.0, 12.0],
            "high": [10.5, 11.0, 12.0, 12.5], "low": [9.8, 9.9, 10.8, 11.8],
            "close": [10.2, 10.8, 11.8, 12.2],
            "signal_atr_raw": [0.5, 0.5, 0.5, 0.5],
            "adv20": [1_000_000.0] * 4, "mom126": [0.2] * 4,
            "entry_trend_breakout": [True, False, False, False],
            "exit_trend_breakout": [False, False, False, False],
        })
        universe = pd.DataFrame({
            "year": [2016], "symbol": ["2330"], "rank": [1],
        })
        _, trades, _ = simulate({"2330": frame}, universe, "trend_breakout")
        self.assertEqual(len(trades), 1)
        self.assertEqual(pd.Timestamp(trades.iloc[0]["signal_date"]), dates[0])
        self.assertEqual(pd.Timestamp(trades.iloc[0]["entry_date"]), dates[1])
        self.assertGreater(float(trades.iloc[0]["entry_price"]), 10.0)


    def test_higher_cost_stress_reduces_ending_equity(self):
        dates = pd.to_datetime(["2016-01-04", "2016-01-05", "2016-01-06", "2016-01-07"])
        frame = pd.DataFrame({
            "date": dates, "open": [10.0, 10.0, 11.0, 12.0],
            "high": [10.5, 11.0, 12.0, 12.5], "low": [9.8, 9.9, 10.8, 11.8],
            "close": [10.2, 10.8, 11.8, 12.2],
            "signal_atr_raw": [0.5] * 4, "adv20": [1_000_000.0] * 4,
            "mom126": [0.2] * 4, "entry_trend_breakout": [True, False, False, False],
            "exit_trend_breakout": [False] * 4,
        })
        universe = pd.DataFrame({"year": [2016], "symbol": ["2330"], "rank": [1]})
        base, _, _ = simulate({"2330": frame}, universe, "trend_breakout", cost_multiplier=1.0)
        stressed, _, _ = simulate({"2330": frame}, universe, "trend_breakout", cost_multiplier=2.0)
        self.assertLess(stressed["ending_equity"], base["ending_equity"])

if __name__ == "__main__":
    unittest.main()
