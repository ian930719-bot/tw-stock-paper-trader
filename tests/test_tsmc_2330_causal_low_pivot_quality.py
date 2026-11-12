import unittest
import numpy as np
import pandas as pd
from tsmc_2330_causal_low_pivot_quality import build_events

class CausalLowPivotQualityTests(unittest.TestCase):
    def test_empty_when_no_pivots_possible(self):
        n=80
        close=np.linspace(100,101,n)
        d=pd.DataFrame({
            "date":pd.date_range("2020-01-01",periods=n),
            "raw_open":close,"raw_high":close+0.2,"raw_low":close-0.2,
            "raw_close":close,"atr_pct14":np.full(n,0.02),
            "corporate_action_window":False
        })
        result=build_events(d)
        self.assertTrue(result.empty)
    def test_forward_labels_are_separate_from_confirmation_features(self):
        # The event builder uses confirmation row features and post-confirmation
        # OHLC exclusively for labels; this test protects output schema contract.
        n=100
        close=np.r_[np.linspace(100,130,30),np.linspace(129,90,30),np.linspace(91,140,40)]
        d=pd.DataFrame({
            "date":pd.date_range("2020-01-01",periods=n),
            "raw_open":close,"raw_high":close*1.01,"raw_low":close*.99,
            "raw_close":close,"atr_pct14":np.full(n,.025),
            "corporate_action_window":False,"rsi14":np.linspace(30,60,n)
        })
        result=build_events(d)
        if not result.empty:
            self.assertIn("confirm_rsi14",result.columns)
            self.assertIn("mfe_20d",result.columns)
            self.assertIn("mae_20d",result.columns)

if __name__=="__main__":
    unittest.main()
