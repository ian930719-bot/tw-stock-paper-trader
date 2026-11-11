import unittest
import numpy as np
import pandas as pd
from tsmc_2330_combinatorial_purged_cv import make_groups, purged_train_mask, labels_for, summarize_coefficients, build_feature_sets, GROUPS, TEST_GROUPS
import itertools

class CombinatorialPurgedCVTests(unittest.TestCase):
    def test_six_contiguous_groups_cover_all_rows_once(self):
        groups=make_groups(600)
        self.assertEqual(len(groups),GROUPS)
        self.assertEqual(np.concatenate(groups).tolist(),list(range(600)))
        self.assertTrue(all(len(g)>0 for g in groups))

    def test_all_two_group_combinations(self):
        self.assertEqual(len(list(itertools.combinations(range(GROUPS),TEST_GROUPS))),15)

    def test_purge_removes_test_blocks_and_neighboring_label_window(self):
        groups=make_groups(600)
        mask=purged_train_mask(600,groups,(2,4),5)
        for gid in (2,4):
            lo,hi=int(groups[gid][0]),int(groups[gid][-1])
            self.assertFalse(mask[max(0,lo-5):min(600,hi+6)].any())

    def test_coefficient_stability_summary(self):
        coef=pd.DataFrame({
            "feature_set":["technical_only"]*4,
            "threshold_id":["p0.03_atr2.0"]*4,
            "horizon_sessions":[1]*4,
            "pivot_type":["LOW"]*4,
            "fold":[1,2,3,4],
            "feature":["rsi14"]*4,
            "coefficient":[0.4,0.2,-0.1,0.3],
        })
        out=summarize_coefficients(coef).iloc[0]
        self.assertEqual(out.folds,4)
        self.assertAlmostEqual(out.positive_fold_share,0.75)
        self.assertAlmostEqual(out.median_coefficient,0.25)


    def test_institutional_ablation_uses_same_rows_and_excludes_broker_fields(self):
        frame=pd.DataFrame({
            "institutional_data_available":[1]*400,
            "foreign_net_lots":[0.1]*400,
            "trust_net_lots":[0.2]*400,
            "dealer_net_lots":[-0.1]*400,
            "broker_net_lots":[9.0]*400,
        })
        sets,available,count,minimum=build_feature_sets(frame)
        self.assertIn("technical_only",sets)
        self.assertIn("technical_plus_institutional",sets)
        self.assertIn("foreign_net_lots",sets["technical_plus_institutional"])
        self.assertNotIn("broker_net_lots",sets["technical_plus_institutional"])
        self.assertEqual(int(available.sum()),400)
        self.assertEqual(count,3)

    def test_institutional_ablation_is_skipped_without_sufficient_coverage(self):
        frame=pd.DataFrame({
            "institutional_data_available":[1]*20+[0]*380,
            "foreign_net_lots":[0.1]*20+[np.nan]*380,
        })
        sets,available,count,minimum=build_feature_sets(frame)
        self.assertNotIn("technical_plus_institutional",sets)
        self.assertEqual(int(available.sum()),20)

    def test_exact_lead_labels(self):
        dates=pd.bdate_range("2020-01-01",periods=30)
        events=pd.DataFrame({"type":["HIGH"],"pivot_date":[dates[15]]})
        self.assertEqual(np.flatnonzero(labels_for(dates,events,5,"HIGH")).tolist(),[10])

if __name__=="__main__": unittest.main()
