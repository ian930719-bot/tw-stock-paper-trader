import os
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from tsmc_2330_500_pivot_study_rebuilt import run

class RebuiltPivotControlTests(unittest.TestCase):
    def test_rebuilt_controls_exclude_action_and_pivot_windows(self):
        features=Path(os.environ.get("TSMC_FEATURES","/mnt/data/run42/artifacts/tsmc_2330_swing/tsmc_2330_causal_features.csv"))
        events=Path(os.environ.get("TSMC_EVENTS","/mnt/data/run42/artifacts/tsmc_2330_swing/tsmc_2330_pivot_events.csv"))
        if not features.exists() or not events.exists():
            self.skipTest("frozen TSMC features/events unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            result=run(str(features),str(events),tmp,250)
            snap=pd.read_csv(Path(tmp)/"tsmc_2330_500_pivot_snapshots.csv",parse_dates=["feature_date","control_date"])
            self.assertEqual(result["selected_pivots"],500)
            self.assertGreater(len(snap),2000)
            self.assertTrue(result["corporate_action_rows_excluded"])
            self.assertTrue(result["event_feature_dates_excluded_from_controls"])
            self.assertGreater(result["unique_control_dates"],400)

if __name__=="__main__":
    unittest.main()
