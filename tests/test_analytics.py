"""Unit tests for supplemental portfolio analytics."""
import unittest
import numpy as np
import pandas as pd

from analytics import equity_risk_metrics, trade_statistics


class AnalyticsTests(unittest.TestCase):
    def test_equity_risk_metrics_include_initial_capital_drawdown(self):
        equity = pd.DataFrame({
            "資產": [990.0, 1010.0, 900.0],
            "持倉市值": [500.0, 700.0, 0.0],
        })
        result = equity_risk_metrics(equity, 1000.0)
        self.assertGreater(result["年化波動率"], 0)
        self.assertLess(result["Calmar"], 0)
        self.assertGreaterEqual(result["平均曝險"], 0)
        self.assertLessEqual(result["平均曝險"], 1)

    def test_trade_statistics(self):
        trades = pd.DataFrame({
            "損益": [100.0, -50.0, 150.0],
            "持有交易日": [3, 5, 2],
        })
        result = trade_statistics(trades)
        self.assertEqual(result["交易筆數"], 3)
        self.assertAlmostEqual(result["勝率"], 2 / 3)
        self.assertAlmostEqual(result["Profit Factor"], 5.0)
        self.assertAlmostEqual(result["期望值"], 200 / 3)
        self.assertAlmostEqual(result["平均持有日數"], 10 / 3)

    def test_empty_trade_statistics_are_safe(self):
        result = trade_statistics(pd.DataFrame())
        self.assertEqual(result["交易筆數"], 0)
        self.assertEqual(result["勝率"], 0.0)
        self.assertEqual(result["Profit Factor"], 0.0)

    def test_metrics_are_finite_except_profit_factor_without_losses(self):
        result = trade_statistics(pd.DataFrame({"損益": [10.0, 20.0]}))
        self.assertTrue(np.isinf(result["Profit Factor"]))


if __name__ == "__main__":
    unittest.main()
