"""Tests for Taiwan corporate-action normalization and event integrity."""
import unittest
import pandas as pd

from corporate_actions import normalize_corporate_actions, validate_action_frame


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

    def test_split_capital_reduction_and_par_value_ratios(self):
        events = normalize_corporate_actions(
            "0050",
            splits=[{"date":"2026-01-01","before_price":200,"after_price":50}],
            capital_reductions=[{"date":"2026-02-01","ClosingPriceonTheLastTradingDay":100,
                                 "PostReductionReferencePrice":50}],
            par_value_changes=[{"date":"2026-03-01","before_close":80,"after_ref_close":40}],
        )
        self.assertEqual(len(events), 3)
        self.assertTrue((events["share_multiplier"] == 2.0).all())

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
