import unittest
from tsmc_2330_institutional_flow_oof import score, TECH, FOREIGN, TRUST, GROUPS

class InstitutionalFlowOOFTests(unittest.TestCase):
 def test_only_foreign_and_trust_are_used(self):
  self.assertTrue(all(x.startswith("foreign_") for x in FOREIGN))
  self.assertTrue(all(x.startswith("trust_") for x in TRUST))
  self.assertFalse(any("dealer" in x or "institutional_" in x or "broker_" in x for x in FOREIGN+TRUST))
  self.assertEqual(set(GROUPS),{"technical_only","technical_plus_foreign","technical_plus_trust","technical_plus_foreign_trust"})
  self.assertGreater(len(TECH),10)
 def test_rank_metrics_use_top_decile(self):
  r=score([1,0,0,0,1,0,0,0,0,0],[.9,.8,.7,.6,.5,.4,.3,.2,.1,0])
  self.assertEqual(r["n"],10);self.assertEqual(r["top_decile_n"],1)
  self.assertEqual(r["top_decile_precision"],1.0);self.assertAlmostEqual(r["positive_rate"],.2)
 def test_single_class_fold_is_not_scored(self):
  self.assertIn("error",score([0,0,0],[.1,.2,.3]))
if __name__=="__main__":unittest.main()
