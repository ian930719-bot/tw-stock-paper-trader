import unittest
import numpy as np
import pandas as pd
from tsmc_2330_all_feature_event_audit import paired_feature_stats, bh_adjust

class AllFeatureAuditTests(unittest.TestCase):
    def test_clustered_feature_difference(self):
        n=60
        frame=pd.DataFrame({
          "feature":np.ones(n)*2,
          "control_feature":np.ones(n),
          "control_date":[f"2020-01-{(i%20)+1:02d}" for i in range(n)]
        })
        result=paired_feature_stats(frame,"feature",np.random.default_rng(7),200)
        self.assertIsNotNone(result)
        self.assertEqual(result["n_pairs"],60)
        self.assertEqual(result["n_control_clusters"],20)
        self.assertAlmostEqual(result["mean_difference"],1.0)
        self.assertGreater(result["cluster_sign_p"],0)

    def test_bh_adjustment_is_monotonic_in_rank(self):
        q=bh_adjust([0.001,0.02,0.5,0.03])
        self.assertEqual(len(q),4)
        self.assertTrue(np.all((q>=0)&(q<=1)))
        self.assertLess(q[0],q[2])

if __name__=="__main__":
    unittest.main()
