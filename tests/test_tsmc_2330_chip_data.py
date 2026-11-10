import unittest
import pandas as pd

from tsmc_2330_chip_data import (
    normalize_institutional, normalize_broker_rows, broker_daily_features,
)


class TSMCChipDataTests(unittest.TestCase):
    def test_institutional_categories_and_net_are_normalized(self):
        frame = normalize_institutional([{
            "date": "2024-01-02",
            "Foreign_Investor_buy": 10000, "Foreign_Investor_sell": 7000,
            "Foreign_Dealer_Self_buy": 500, "Foreign_Dealer_Self_sell": 1000,
            "Investment_Trust_buy": 2000, "Investment_Trust_sell": 3000,
            "Dealer_self_buy": 400, "Dealer_self_sell": 100,
            "Dealer_Hedging_buy": 600, "Dealer_Hedging_sell": 900,
        }])
        row = frame.iloc[0]
        self.assertEqual(row["foreign_net_shares"], 2500)
        self.assertEqual(row["trust_net_shares"], -1000)
        self.assertEqual(row["dealer_net_shares"], 0)
        self.assertEqual(row["institutional_net_shares"], 1500)

    def test_broker_rows_keep_branch_ids_and_convert_shares_to_lots(self):
        frame = normalize_broker_rows([{
            "securities_trader_id": "9A9A", "securities_trader": "測試分點",
            "buy": 12500, "sell": 8500, "price": 100,
        }], "2024-01-02")
        row = frame.iloc[0]
        self.assertEqual(row["net_shares"], 4000)
        self.assertEqual(row["buy_lots"], 12.5)
        self.assertEqual(row["net_lots"], 4.0)
        self.assertEqual(row["securities_trader_id"], "9A9A")

    def test_broker_daily_features_rank_concentration_without_fabricating_zero(self):
        branches = pd.DataFrame([
            {"date": "2024-01-02", "securities_trader_id": "A", "buy_lots": 20, "sell_lots": 2, "net_lots": 18},
            {"date": "2024-01-02", "securities_trader_id": "B", "buy_lots": 1, "sell_lots": 8, "net_lots": -7},
        ])
        out = broker_daily_features(branches).iloc[0]
        self.assertEqual(out["broker_branch_count"], 2)
        self.assertEqual(out["broker_net_lots"], 11)
        self.assertEqual(out["broker_top5_net_lots"], 11)
        empty = broker_daily_features(pd.DataFrame())
        self.assertTrue(empty.empty)


if __name__ == "__main__":
    unittest.main()
