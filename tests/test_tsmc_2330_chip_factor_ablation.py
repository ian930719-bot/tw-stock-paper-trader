import unittest
import pandas as pd
import numpy as np

from tsmc_2330_chip_factor_ablation import feature_groups, evaluate, annual_fold_years, _events_for_fold


class TSMCChipFactorAblationTests(unittest.TestCase):
    def test_feature_groups_separate_technical_institutional_and_broker(self):
        features = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3),
            "close": [1, 2, 3],
            "return_1": [0.1, 0.2, 0.3],
            "foreign_net_lots": [1, -1, 2],
            "foreign_dealer_net_lots": [0, 1, -1],
            "trust_net_lots": [1, 0, -1],
            "dealer_net_lots": [0, -1, 1],
            "institutional_net_lots": [1, -1, 2],
            "broker_net_lots": [0, 2, -1],
        })
        groups = feature_groups(features)
        self.assertEqual(groups["technical_only"], ["return_1"])
        self.assertIn("foreign_net_lots", groups["technical_plus_institutional"])
        self.assertNotIn("broker_net_lots", groups["technical_plus_institutional"])
        self.assertIn("foreign_net_lots", groups["technical_plus_foreign_only"])
        self.assertNotIn("trust_net_lots", groups["technical_plus_foreign_only"])
        self.assertIn("trust_net_lots", groups["technical_plus_trust_only"])
        self.assertIn("dealer_net_lots", groups["technical_plus_dealer_only"])
        self.assertIn("institutional_net_lots", groups["technical_plus_institutional_total_only"])
        self.assertIn("broker_net_lots", groups["technical_plus_all_chip"])

    def test_annual_walk_forward_uses_expanding_training_years(self):
        dates = pd.Series(pd.to_datetime([
            "2010-01-04", "2010-12-31", "2011-01-03",
            "2011-12-30", "2012-01-02", "2012-12-31",
        ]))
        self.assertEqual(annual_fold_years(dates), [2011, 2012])

    def test_training_fold_excludes_labels_not_confirmed_by_cutoff(self):
        dates = pd.Series(pd.date_range("2020-01-01", periods=8, freq="D"))
        events = pd.DataFrame({
            "pivot_date": [dates.iloc[3], dates.iloc[4]],
            "confirmation_date": [dates.iloc[4], dates.iloc[6]],
            "type": ["HIGH", "LOW"],
        })
        selected = _events_for_fold(
            events, dates, 0, 4, 1, training=True, available_end_idx=7
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(pd.Timestamp(selected.iloc[0]["pivot_date"]), dates.iloc[3])

    def test_evaluate_reports_lift_against_prevalence(self):
        from sklearn.dummy import DummyClassifier
        X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        model = DummyClassifier(strategy="prior")
        model.fit(X, y)
        metrics = evaluate(model, X, np.ones(10, dtype=bool), pd.Series(y).to_numpy())
        self.assertAlmostEqual(metrics["prevalence"], 0.2)
        self.assertIn("top_decile_lift", metrics)


if __name__ == "__main__":
    unittest.main()
