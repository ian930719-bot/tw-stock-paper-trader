import unittest
import numpy as np
import pandas as pd

from tsmc_2330_selected_threshold_model_comparison import evaluate, make_model


class FakeModel:
    def predict_proba(self, frame):
        p = np.asarray(frame["score"], dtype=float)
        return np.column_stack([1 - p, p])


class SelectedThresholdModelTests(unittest.TestCase):
    def test_predeclared_models_construct(self):
        for name in ("logistic", "random_forest", "hist_gradient_boosting"):
            self.assertIsNotNone(make_model(name))

    def test_evaluation_reports_precision_at_multiple_cutoffs(self):
        frame = pd.DataFrame({"score": [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]})
        mask = np.ones(10, dtype=bool)
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        result = evaluate(FakeModel(), frame, mask, y)
        self.assertAlmostEqual(result["top_10_precision"], 1.0)
        self.assertAlmostEqual(result["top_10_recall"], 1.0)
        self.assertAlmostEqual(result["prevalence"], 0.2)
        self.assertIn("top_1_precision", result)
        self.assertIn("top_5_precision", result)


if __name__ == "__main__":
    unittest.main()
