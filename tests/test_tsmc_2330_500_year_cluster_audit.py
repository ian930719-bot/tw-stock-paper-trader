import tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from tsmc_2330_500_year_cluster_audit import analyze

class YearClusterAuditTests(unittest.TestCase):
    def test_outputs_and_year_clusters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rows=[]
            for kind in ["HIGH","LOW"]:
                for year in range(2010,2022):
                    for event in range(6):
                        for lead in [1,3,5,10]:
                            v=1.0+(year-2010)*.01+event*.001
                            rows.append({"event_no":f"{kind}-{year}-{event}","pivot_type":kind,
                              "pivot_date":f"{year}-06-01","lead_sessions":lead,
                              "signal":v,"control_signal":v-.5,
                              "control_match":"year_and_trend"})
            src=root/"snapshots.csv"; pd.DataFrame(rows).to_csv(src,index=False)
            result=analyze(src,root/"out",bootstrap=100)
            self.assertEqual(result["input_rows"],len(rows))
            detail=pd.read_csv(root/"out/tsmc_2330_500_year_cluster_feature_audit.csv")
            ranking=pd.read_csv(root/"out/tsmc_2330_500_stable_feature_candidates.csv")
            self.assertEqual(len(detail),8)
            self.assertTrue((detail.cluster_bootstrap_ci_low>0).all())
            self.assertTrue(ranking.stable_candidate.all())

if __name__=="__main__": unittest.main()
