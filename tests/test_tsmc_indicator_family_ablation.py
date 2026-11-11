import unittest
from tsmc_2330_indicator_family_ablation import family_for

class IndicatorFamilyAblationTests(unittest.TestCase):
    def test_indicator_families_are_disjoint_and_named(self):
        self.assertEqual(family_for("ema_20"), "trend_ema_momentum")
        self.assertEqual(family_for("stoch_k14"), "oscillator_trend_strength")
        self.assertEqual(family_for("mfi14"), "volume_money_flow")
        self.assertEqual(family_for("donchian_position20"), "price_channel")
        self.assertEqual(family_for("foreign_net_shares"), "institutional")
        self.assertEqual(family_for("rsi14"), "core_technical")
        self.assertEqual(family_for("obv"), "core_technical")

if __name__ == "__main__":
    unittest.main()
