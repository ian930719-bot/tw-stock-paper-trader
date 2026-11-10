import unittest
import pandas as pd

from tsmc_2330_chip_factor_ablation import feature_groups, evaluate


class TSMCChipFactorAblationTests(unittest.TestCase):
    def test_feature_groups_separate_technical_institutional_and_broker(self):
        features = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3),
            "close": [1, 2, 3],
            "return_1": [0.1, 0.2, 0.3],
            "foreign_net_lots": [1, -1, 2],
            "institutional_net_lots": [1, -1, 2],
            "broker_net_lots": [0, 2, -1],
        })
        groups = feature_groups(features)
        self.assertEqual(groups["technical_only"], ["return_1"])
        self.assertIn("foreign_net_lots", groups["technical_plus_institutional"])
        self.assertNotIn("broker_net_lots", groups["technical_plus_institutional"])
        self.assertIn("broker_net_lots", groups["technical_plus_all_chip"])

    def test_evaluate_reports_lift_against_prevalence(self):
        from sklearn.dummy import DummyClassifier
        X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        model = DummyClassifier(strategy="prior")
        model.fit(X, y)
        metrics = evaluate(model, X, [True] * 10, pd.Series(y).to_numpy())
        self.assertAlmostEqual(metrics["prevalence"], 0.2)
        self.assertIn("top_decile_lift", metrics)


if __name__ == "__main__":
    unittest.main()
