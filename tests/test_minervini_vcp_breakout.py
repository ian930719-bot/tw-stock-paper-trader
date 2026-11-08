import unittest
import numpy as np
import pandas as pd

from minervini_vcp_breakout import _features


class MinerviniVCPFeatureTests(unittest.TestCase):
    def make_frame(self):
        n=420
        dates=pd.bdate_range("2023-01-02",periods=n)
        close=50*np.exp(np.arange(n)*0.0015 + 0.015*np.sin(np.arange(n)/8))
        volume=np.full(n,100000.0)
        return pd.DataFrame({
            "date":dates,"open":close*0.999,"high":close*1.01,
            "low":close*0.99,"close":close,"volume":volume,
        })

    def test_required_features_are_created(self):
        out=_features(self.make_frame())
        for col in ["sma50","sma150","sma200","high252","low252",
                    "high20","low10","volavg50","mom126","mom63",
                    "atr_ratio","ema21"]:
            self.assertIn(col,out.columns)
        self.assertEqual(len(out),420)

    def test_future_bars_do_not_change_past_features(self):
        frame=self.make_frame()
        baseline=_features(frame)
        changed=frame.copy()
        changed.loc[350:,"close"]*=1.8
        changed.loc[350:,"high"]*=1.8
        changed.loc[350:,"low"]*=1.8
        changed.loc[350:,"volume"]*=4
        altered=_features(changed)
        cols=["sma50","sma150","sma200","high252","low252",
              "high20","low10","volavg50","mom126","mom63","atr_ratio","ema21"]
        pd.testing.assert_frame_equal(
            baseline.loc[:349,cols],altered.loc[:349,cols],
            check_exact=False,rtol=1e-12,atol=1e-12
        )


if __name__=="__main__":
    unittest.main()
