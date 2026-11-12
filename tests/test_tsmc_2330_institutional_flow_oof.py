import unittest
from tsmc_2330_institutional_flow_oof import score, TECH, CHIP

class InstitutionalFlowOOFTests(unittest.TestCase):
    def test_feature_families_are_explicit_and_no_broker_fields(self):
        self.assertIn("foreign_net_to_volume", CHIP)
        self.assertIn("trust_net_to_volume", CHIP)
        self.assertIn("institutional_net_to_volume", CHIP)
        self.assertFalse(any(x.startswith("broker_") for x in CHIP))
        self.assertGreater(len(TECH), 10)

    def test_rank_metrics_use_top_decile(self):
        result=score([1,0,0,0,1,0,0,0,0,0],[.9,.8,.7,.6,.5,.4,.3,.2,.1,0])
        self.assertEqual(result["n"],10)
        self.assertEqual(result["top_decile_n"],1)
        self.assertEqual(result["top_decile_precision"],1.0)
        self.assertAlmostEqual(result["positive_rate"],0.2)

    def test_single_class_fold_is_reported_not_scored(self):
        result=score([0,0,0],[.1,.2,.3])
        self.assertIn("error",result)

if __name__=="__main__":
    unittest.main()
