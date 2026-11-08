import unittest
from strategy_exits import PROFILES, initial_stop, risk_position_size, trailing_stop


class StrategyExitTests(unittest.TestCase):
    def test_profiles_are_explicitly_research_references(self):
        self.assertIn("oneil_growth", PROFILES)
        self.assertIn("minervini_sepa", PROFILES)
        self.assertIn("atr_dynamic", PROFILES)
        self.assertIn("候選", PROFILES["atr_dynamic"]["description"])

    def test_oneil_stop_reference_and_technical_level(self):
        result = initial_stop(100, "oneil_growth", oneil_loss_cap=0.08)
        self.assertAlmostEqual(result["stop_price"], 92)
        result = initial_stop(100, "oneil_growth", technical_invalidation=95)
        self.assertEqual(result["stop_price"], 95)

    def test_minervini_requires_valid_technical_invalidation(self):
        with self.assertRaisesRegex(ValueError, "必須提供"):
            initial_stop(100, "minervini_sepa")
        with self.assertRaises(ValueError):
            initial_stop(100, "minervini_sepa", technical_invalidation=101)
        self.assertEqual(initial_stop(100, "minervini_sepa",
                                     technical_invalidation=94)["stop_price"], 94)

    def test_atr_stop(self):
        self.assertEqual(initial_stop(100, "atr_dynamic", atr=2,
                                      atr_multiple=2)["stop_price"], 96)

    def test_risk_sizing_is_account_risk_not_stock_stop_percent(self):
        result = risk_position_size(1_000_000, 0.01, 100, 95)
        self.assertEqual(result["risk_budget"], 10_000)
        self.assertEqual(result["shares"], 2_000)
        self.assertEqual(result["estimated_price_risk"], 10_000)

    def test_risk_sizing_respects_cap_and_rejects_bad_stop(self):
        result = risk_position_size(1_000_000, 0.01, 100, 95, max_shares=100)
        self.assertEqual(result["shares"], 100)
        with self.assertRaises(ValueError):
            risk_position_size(100_000, 0.01, 100, 100)

    def test_trailing_stop_never_moves_down(self):
        self.assertEqual(trailing_stop(120, 2, 2, previous_stop=110), 116)
        self.assertEqual(trailing_stop(110, 2, 2, previous_stop=115), 115)


if __name__ == "__main__":
    unittest.main()
