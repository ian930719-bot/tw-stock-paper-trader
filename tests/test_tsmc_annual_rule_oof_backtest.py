import unittest
import pandas as pd

from tsmc_2330_annual_rule_oof_backtest import annual_metrics, _params


class AnnualRuleOOFTests(unittest.TestCase):
    def test_parameter_grid_mapping_is_stable(self):
        params = _params((-0.008, 1.2, 0.75, 0.10))
        self.assertEqual(params, {
            "low_return": -0.008,
            "low_relative_volume": 1.2,
            "high_close_location": 0.75,
            "high_distance_sma200": 0.10,
        })

    def test_annual_metrics_carry_equity_forward(self):
        curve = pd.DataFrame([
            {"date": "2015-01-05", "equity": 100000.0},
            {"date": "2015-12-31", "equity": 110000.0},
            {"date": "2016-01-04", "equity": 110000.0},
            {"date": "2016-12-30", "equity": 99000.0},
            {"date": "2017-01-03", "equity": 100000.0},
        ])
        rows = annual_metrics(curve, [], 100000.0)
        self.assertEqual([r["year"] for r in rows], [2015, 2016, 2017])
        self.assertAlmostEqual(rows[0]["annual_return"], 0.10)
        self.assertAlmostEqual(rows[1]["annual_return"], -0.10)
        self.assertAlmostEqual(rows[2]["annual_return"], 1000 / 99000)
        self.assertEqual(rows[0]["trade_count"], 0)


if __name__ == "__main__":
    unittest.main()
