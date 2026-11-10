import unittest

from tsmc_2330_tree_exit_rule_research import selection_score


class TreeExitRuleResearchTests(unittest.TestCase):
    def test_selection_score_penalizes_drawdown(self):
        lower_drawdown = {"cagr": 0.20, "max_drawdown": -0.10}
        higher_drawdown = {"cagr": 0.25, "max_drawdown": -0.30}
        self.assertGreater(selection_score(lower_drawdown), selection_score(higher_drawdown))

    def test_score_uses_cagr_and_absolute_drawdown(self):
        result = {"cagr": 0.30, "max_drawdown": -0.20}
        self.assertAlmostEqual(selection_score(result), 0.20)


if __name__ == "__main__":
    unittest.main()
