import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from tsmc_2330_paired_annual_blind_validation import event_target, ambiguous_rows, calibrate, run, CORE_FEATURES
from tsmc_swing_research import THRESHOLDS

class PairedAnnualBlindValidationTests(unittest.TestCase):
    def test_sample_expansion_grid_is_predeclared_and_unique(self):
        self.assertEqual(len(THRESHOLDS), len(set(THRESHOLDS)))
        for threshold in ((0.01, 1.0), (0.02, 1.0), (0.03, 1.0), (0.04, 1.0), (0.02, 1.5)):
            self.assertIn(threshold, THRESHOLDS)

    def test_exact_lead_label_is_one_session_only(self):
        dates=pd.Series(pd.bdate_range("2020-01-01",periods=20))
        events=pd.DataFrame({"type":["HIGH"],"pivot_date":[dates.iloc[10]]})
        self.assertEqual(np.flatnonzero(event_target(dates,events,3,"HIGH")).tolist(),[7])

    def test_unconfirmed_pivot_lead_row_is_excluded(self):
        dates=pd.Series(pd.bdate_range("2020-01-01",periods=20))
        events=pd.DataFrame({"type":["HIGH"],"pivot_date":[dates.iloc[10]],
                             "confirmation_date":[dates.iloc[12]]})
        self.assertEqual(np.flatnonzero(ambiguous_rows(dates,events,3,dates.iloc[9])).tolist(),[7])

    def test_calibration_requires_minimum_events(self):
        threshold,info=calibrate(np.array([0,1,0,0]),np.array([.1,.8,.2,.3]))
        self.assertIsNone(threshold)
        self.assertEqual(info["status"],"skipped_insufficient_calibration_events")

    def test_pairing_is_2010_to_2011_and_does_not_use_future_validation_for_fit(self):
        dates=pd.bdate_range("2010-01-01","2011-12-31"); n=len(dates)
        rng=np.random.default_rng(7); close=100+np.cumsum(rng.normal(0,1,n))
        x=pd.DataFrame({"date":dates,"open":close,"high":close*1.01,"low":close*.99,
                        "close":close,"feature_a":rng.normal(size=n),"feature_b":rng.normal(size=n)})
        # Mirror the production contract: include the complete predeclared core feature set.
        for feature in CORE_FEATURES:
            x[feature]=rng.normal(size=n)
        events=[]
        for year,months in ((2010,[2,3,4,5,10,11,12]),(2011,[2,4,6,8,10])):
            for kind in ("HIGH","LOW"):
                for month in months:
                    ix=np.flatnonzero((dates.year==year)&(dates.month==month))
                    if len(ix)<10: continue
                    pivot=int(ix[-5])
                    events.append({"threshold_id":"p0.050_atr2.0","type":kind,
                                   "pivot_date":dates[pivot],"confirmation_date":dates[pivot+2]})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); x.to_csv(root/"features.csv",index=False); pd.DataFrame(events).to_csv(root/"events.csv",index=False)
            summary=run(str(root/"features.csv"),str(root/"events.csv"),str(root/"out"),["p0.050_atr2.0"])
            result=pd.read_csv(root/"out"/"tsmc_2330_paired_annual_blind_validation.csv")
            evaluated=result[result.status=="evaluated"]
            self.assertTrue(summary["validation_is_never_used_for_fit_or_threshold"])
            self.assertTrue((evaluated.analysis_year==2010).all())
            self.assertTrue((evaluated.validation_year==2011).all())
            self.assertTrue((evaluated.training_cutoff<evaluated.calibration_cutoff).all())

if __name__=="__main__": unittest.main()
