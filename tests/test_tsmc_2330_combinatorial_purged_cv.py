import unittest
import numpy as np
import pandas as pd
from tsmc_2330_combinatorial_purged_cv import make_groups, purged_train_mask, labels_for, GROUPS, TEST_GROUPS
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

    def test_exact_lead_labels(self):
        dates=pd.bdate_range("2020-01-01",periods=30)
        events=pd.DataFrame({"type":["HIGH"],"pivot_date":[dates[15]]})
        self.assertEqual(np.flatnonzero(labels_for(dates,events,5,"HIGH")).tolist(),[10])

if __name__=="__main__": unittest.main()
