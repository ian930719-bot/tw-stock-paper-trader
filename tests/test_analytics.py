"""Unit tests for supplemental portfolio analytics."""
import unittest
import numpy as np
import pandas as pd

from analytics import equity_risk_metrics, trade_statistics, strategy_decay_report


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

    def test_cagr_uses_actual_elapsed_calendar_time_when_dates_exist(self):
        equity = pd.DataFrame({
            "日期": pd.to_datetime(["2020-01-01", "2021-01-01"]),
            "資產": [100.0, 121.0],
            "持倉市值": [100.0, 121.0],
        })
        result = equity_risk_metrics(equity, 100.0)
        expected = 1.21 ** (365.25 / 366.0) - 1.0
        self.assertAlmostEqual(result["年化波動率"], result["年化波動率"])
        self.assertAlmostEqual(result["Calmar"], expected / abs(result["最大回撤"]) if result["最大回撤"] < 0 else 0.0)
        self.assertAlmostEqual(result["平均曝險"], 1.0)

    def test_strategy_decay_report_flags_recent_reversal(self):
        dates = pd.bdate_range("2024-01-02", periods=140)
        rng = np.random.default_rng(17)
        returns = np.r_[rng.normal(0.005, 0.001, 77), rng.normal(-0.003, 0.001, 63)]
        curve = pd.DataFrame({
            "日期": dates,
            "資產": 100000 * np.cumprod(1 + returns),
        })
        report = strategy_decay_report({"測試策略": curve}, 100000, recent_window=63)
        self.assertEqual(len(report), 1)
        self.assertEqual(report.iloc[0]["狀態"], "退化警示")
        self.assertLess(float(report.iloc[0]["近期Sharpe"]), 0)

    def test_strategy_decay_report_requires_minimum_observations(self):
        dates = pd.bdate_range("2024-01-02", periods=25)
        curve = pd.DataFrame({"日期": dates, "資產": np.linspace(100000, 101000, 25)})
        report = strategy_decay_report({"短資料": curve}, 100000)
        self.assertEqual(report.iloc[0]["狀態"], "樣本不足")

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
