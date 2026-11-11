import tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from tsmc_2330_500_pivot_study import FEATURES, THRESHOLD_ID, run

class FiveHundredPivotStudyTests(unittest.TestCase):
    def test_balanced_sample_and_pre_pivot_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); n=900; dates=pd.bdate_range("2020-01-01",periods=n)
            x=pd.DataFrame({"date":dates,"trend_stack":np.tile([0,1],n//2)})
            for j,c in enumerate(FEATURES): x[c]=np.arange(n,dtype=float)*(j+1)/n
            rows=[]
            for kind,offset in [("HIGH",0),("LOW",1)]:
                for k in range(20):
                    pi=30+k*40+offset
                    rows.append({"threshold_id":THRESHOLD_ID,"type":kind,"pivot_index":pi,
                      "confirmation_index":pi+2,"pivot_date":dates[pi],"pivot_price":100+pi,
                      "reversal_pct":.03,"confirmation_lag":2})
            x.to_csv(root/"features.csv",index=False); pd.DataFrame(rows).to_csv(root/"events.csv",index=False)
            result=run(root/"features.csv",root/"events.csv",root/"out",sample_per_side=10)
            selected=pd.read_csv(root/"out/tsmc_2330_500_selected_pivots.csv")
            snapshots=pd.read_csv(root/"out/tsmc_2330_500_pivot_snapshots.csv")
            self.assertEqual(result["selected_pivots"],20)
            self.assertEqual(result["selected_high"],10); self.assertEqual(result["selected_low"],10)
            self.assertEqual(len(selected),20); self.assertEqual(len(snapshots),100)
            self.assertTrue((pd.to_datetime(snapshots.feature_date)<pd.to_datetime(snapshots.pivot_date)).all())
            self.assertTrue((pd.to_datetime(snapshots.control_date)!=pd.to_datetime(snapshots.pivot_date)).all())

if __name__=="__main__": unittest.main()
