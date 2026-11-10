import unittest
import numpy as np
import pandas as pd

from tsmc_2330_interpretable_pivot_rules import make_tree, evaluate, _leaf_rows


class InterpretablePivotRuleTests(unittest.TestCase):
    def test_shallow_tree_produces_human_readable_rules(self):
        x = pd.DataFrame({"rsi14": np.linspace(10, 90, 120), "return_3": np.sin(np.arange(120))})
        y = (x["rsi14"].to_numpy() > 55).astype(int)
        model = make_tree(2)
        model.fit(x, y)
        rows = _leaf_rows(model, x, np.ones(len(x), dtype=bool), y,
                          list(x.columns), "test", "HIGH", 1, 2)
        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(any("rsi14" in row["rule"] for row in rows))
        self.assertTrue(all(row["rows"] > 0 for row in rows))

    def test_evaluation_rejects_one_class_segments(self):
        x = pd.DataFrame({"x": np.arange(60, dtype=float)})
        y = (x["x"].to_numpy() > 30).astype(int)
        model = make_tree(2)
        model.fit(x, y)
        result = evaluate(model, x, np.ones(60, dtype=bool), np.zeros(60, dtype=int))
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
