import unittest

import pandas as pd

from performance_validation import concentration_diagnostics


class ConcentrationDiagnosticsTests(unittest.TestCase):
    def test_reports_top_trade_and_window_concentration(self):
        trades = pd.DataFrame({"損益": [1000.0, 500.0, 250.0, -200.0]})
        windows = pd.DataFrame({
            "期初資產": [100000.0, 101000.0],
            "期末資產": [101000.0, 102550.0],
        })
        result = concentration_diagnostics(trades, windows, 100000.0, 102550.0)
        self.assertEqual(result["交易筆數"], 4)
        self.assertAlmostEqual(result["最大單筆佔淨利比例"], 1000.0 / 2550.0)
        self.assertAlmostEqual(result["前三筆佔淨利比例"], 1750.0 / 2550.0)
        self.assertAlmostEqual(result["最大區段佔淨利比例"], 1550.0 / 2550.0)
        self.assertEqual(result["移除最大單筆後近似期末資產_NT"], 101550.0)

    def test_nonpositive_net_profit_does_not_emit_misleading_share(self):
        trades = pd.DataFrame({"損益": [-100.0, 50.0]})
        result = concentration_diagnostics(
            trades, pd.DataFrame(), 100000.0, 99950.0
        )
        self.assertTrue(pd.isna(result["最大單筆佔淨利比例"]))
        self.assertTrue(pd.isna(result["前三筆佔淨利比例"]))


if __name__ == "__main__":
    unittest.main()
