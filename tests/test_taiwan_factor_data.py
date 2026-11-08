import unittest
import pandas as pd

from taiwan_factor_data import (
    add_conservative_availability, summarize_institutional,
    enrich_industry_comparison, point_in_time_revenue,
)


class TaiwanFactorDataTests(unittest.TestCase):
    def test_chip_data_available_after_trade_date(self):
        raw = pd.DataFrame([{"date": "2025-01-02", "name": "Foreign_Investor", "buy": 100, "sell": 40}])
        z = add_conservative_availability(raw, "TaiwanStockInstitutionalInvestorsBuySell")
        self.assertEqual(z.loc[0, "available_date"], pd.Timestamp("2025-01-03"))

    def test_financial_statement_period_is_not_publication_date(self):
        raw = pd.DataFrame([{"date": "2024-12-31", "type": "EPS", "value": 5}])
        z = add_conservative_availability(raw, "TaiwanStockFinancialStatements")
        self.assertTrue(pd.isna(z.loc[0, "available_date"]))
        self.assertIn("不可用於歷史回測", z.loc[0, "availability_quality"])

    def test_revenue_uses_only_recorded_ingestion_date(self):
        raw = pd.DataFrame([
            {"date": "2025-01-01", "create_time": "", "revenue": 100},
            {"date": "2025-02-01", "create_time": "2026-04-25", "revenue": 120},
        ])
        z = add_conservative_availability(raw, "TaiwanStockMonthRevenue")
        self.assertTrue(pd.isna(z.loc[0, "available_date"]))
        self.assertEqual(len(point_in_time_revenue(z, "2026-04-24")), 0)
        self.assertEqual(len(point_in_time_revenue(z, "2026-04-25")), 1)

    def test_institutional_summary_and_industry_peer_rank(self):
        raw = pd.DataFrame([
            {"date": "2025-01-02", "name": "Foreign", "buy": 100, "sell": 50},
            {"date": "2025-01-02", "name": "Trust", "buy": 80, "sell": 20},
        ])
        summary = summarize_institutional(raw)
        self.assertEqual(summary.iloc[0]["Foreign"], 50)
        fund = pd.DataFrame([{"代號": "2330", "基本面分數": 80}, {"代號": "2454", "基本面分數": 60}])
        master = pd.DataFrame([{"代號": "2330", "產業": "半導體"}, {"代號": "2454", "產業": "半導體"}])
        ranked = enrich_industry_comparison(fund, master)
        self.assertEqual(list(ranked["產業內排名"]), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
