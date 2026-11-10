import unittest
import pandas as pd

from tsmc_2330_chip_pivot_rules import feature_groups, make_tree, rules_for


class TSMCChipPivotRuleTests(unittest.TestCase):
    def test_feature_groups_separate_technical_and_institutional(self):
        frame = pd.DataFrame({
            "return_1": [0.1, -0.1],
            "rsi14": [40, 60],
            "foreign_net_lots": [2, -3],
            "trust_net_5d_lots": [4, -1],
            "broker_net_lots": [None, None],
            "close": [100, 101],
        })
        groups = feature_groups(frame)
        self.assertEqual(groups["technical_only"], ["return_1", "rsi14"])
        self.assertEqual(groups["technical_plus_institutional"], [
            "return_1", "rsi14", "foreign_net_lots", "trust_net_5d_lots"
        ])

    def test_rule_feature_indices_match_filtered_training_columns(self):
        frame = pd.DataFrame({
            "x": [0, 0, 1, 1, 2, 2, 3, 3],
            "empty": [None] * 8,
        })
        cols = [c for c in frame if frame[c].notna().any()]
        model = make_tree(2)
        model.fit(frame[cols], [0, 0, 0, 0, 1, 1, 1, 1])
        rules = rules_for(model, cols)
        self.assertTrue(any(rule.startswith("x <=") for rule in rules.values()))
        self.assertFalse(any("empty" in rule for rule in rules.values()))


if __name__ == "__main__":
    unittest.main()
