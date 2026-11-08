"""Tests for Taiwan corporate-action normalization and event integrity."""
import unittest
import pandas as pd

from corporate_actions import (
    normalize_corporate_actions, validate_action_frame, point_in_time_recalculate,
)


class CorporateActionTests(unittest.TestCase):
    def test_dividend_ex_dates_and_payment_date_are_distinct(self):
        events = normalize_corporate_actions("2330", dividends=[{
            "StockExDividendTradingDate":"2026-07-01",
            "CashExDividendTradingDate":"2026-07-01",
            "CashDividendPaymentDate":"2026-08-01",
            "StockEarningsDistribution":0.5,
            "StockStatutorySurplus":0.0,
            "CashEarningsDistribution":4.0,
            "CashStatutorySurplus":0.0,
        }])
        stock = events.loc[events["action_type"]=="stock_dividend"].iloc[0]
        cash = events.loc[events["action_type"]=="cash_dividend"].iloc[0]
        self.assertAlmostEqual(float(stock["share_multiplier"]), 1.05)
        self.assertAlmostEqual(float(cash["cash_per_share"]), 4.0)
        self.assertEqual(str(cash["payment_date"]), "2026-08-01")

    def test_unknown_cash_payment_date_remains_unsettled(self):
        events = normalize_corporate_actions("2330", dividends=[{
            "CashExDividendTradingDate":"2026-07-01",
            "CashEarningsDistribution":2.0,
        }])
        cash = events.loc[events["action_type"]=="cash_dividend"].iloc[0]
        self.assertTrue(pd.isna(cash["payment_date"]))
        clean = validate_action_frame(events, "2330")
        self.assertTrue(pd.isna(clean.iloc[0]["payment_date"]))

    def test_split_capital_reduction_and_par_value_ratios(self):
        events = normalize_corporate_actions(
            "0050",
            splits=[{"date":"2026-01-01","before_price":200,"after_price":50}],
            capital_reductions=[{"date":"2026-02-01","ClosingPriceonTheLastTradingDay":100,
                                 "PostReductionReferencePrice":50}],
            par_value_changes=[{"date":"2026-03-01","before_close":80,"after_ref_close":40}],
        )
        self.assertEqual(len(events), 3)
        self.assertEqual(sorted(events["share_multiplier"].astype(float).tolist()), [2.0, 2.0, 4.0])

    def test_official_dividend_result_price_factor_is_attached_once(self):
        events = normalize_corporate_actions(
            "2330",
            dividends=[{
                "CashExDividendTradingDate":"2026-07-01",
                "CashEarningsDistribution":5.0,
                "StockExDividendTradingDate":"2026-07-01",
                "StockEarningsDistribution":1.0,
            }],
            dividend_results=[{
                "date":"2026-07-01","stock_id":"2330",
                "before_price":100.0,"after_price":86.0,
                "reference_price":86.0,
            }],
        )
        self.assertEqual(len(events), 2)
        self.assertTrue((events["price_adjustment_factor"] == 0.86).all())

    def test_point_in_time_recalculation_does_not_rewrite_pre_ex_date_features(self):
        frame = pd.DataFrame({
            "date":pd.date_range("2026-01-01", periods=6),
            "open":[80,90,100,50,55,60],
            "high":[81,91,101,51,56,61],
            "low":[79,89,99,49,54,59],
            "close":[80,90,100,50,55,60],
            "volume":[100,100,100,200,200,200],
        })
        actions = pd.DataFrame([{
            "symbol":"2330","effective_date":"2026-01-04",
            "action_type":"stock_split","share_multiplier":2.0,
            "cash_per_share":0.0,"payment_date":None,
            "reference_price":50.0,"price_adjustment_factor":0.5,
            "event_id":"split-pit",
        }])
        def builder(z):
            close = pd.to_numeric(z["close"])
            return pd.DataFrame({
                "sma3":close.rolling(3,min_periods=1).mean(),
                "return1":close.pct_change(),
                "volume3":pd.to_numeric(z["volume"]).rolling(3,min_periods=1).mean(),
            })
        result = point_in_time_recalculate(frame, actions, builder)
        self.assertAlmostEqual(float(result.loc[2,"sma3"]), 90.0)
        self.assertAlmostEqual(float(result.loc[3,"sma3"]), 45.0)
        self.assertAlmostEqual(float(result.loc[3,"return1"]), 0.0)
        self.assertAlmostEqual(float(result.loc[3,"volume3"]), 200.0)

    def test_cash_dividend_price_gap_is_adjusted_only_from_ex_date(self):
        frame = pd.DataFrame({
            "date":pd.date_range("2026-01-01", periods=4),
            "open":[100,100,90,91],"high":[101,101,91,92],
            "low":[99,99,89,90],"close":[100,100,90,91],
            "volume":[100,100,100,100],
        })
        actions = pd.DataFrame([{
            "symbol":"2330","effective_date":"2026-01-03",
            "action_type":"cash_dividend","share_multiplier":1.0,
            "cash_per_share":10.0,"payment_date":"2026-01-20",
            "reference_price":90.0,"price_adjustment_factor":0.9,
            "event_id":"cash-pit",
        }])
        def builder(z):
            close = pd.to_numeric(z["close"])
            return pd.DataFrame({"return1":close.pct_change()})
        result = point_in_time_recalculate(frame, actions, builder)
        self.assertAlmostEqual(float(result.loc[1,"return1"]), 0.0)
        self.assertAlmostEqual(float(result.loc[2,"return1"]), 0.0)

    def test_invalid_action_ratios_are_filtered(self):
        events = pd.DataFrame([{
            "symbol":"2330","effective_date":"2026-01-01",
            "action_type":"stock_split","share_multiplier":-1,
            "cash_per_share":0,
        }])
        clean = validate_action_frame(events, "2330")
        self.assertTrue(clean.empty)

    def test_duplicate_actions_are_idempotent(self):
        row = {"date":"2026-01-01","before_price":200,"after_price":100}
        events = normalize_corporate_actions("0050", splits=[row,row])
        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
