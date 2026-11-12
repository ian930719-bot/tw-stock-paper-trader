import unittest
import pandas as pd
import numpy as np
from tsmc_2330_4019_replication import build_candidate_signals


class CandidateReplicationTests(unittest.TestCase):
    def _frame(self, n=100):
        return pd.DataFrame({
            "date":pd.date_range("2020-01-01",periods=n),
            "raw_open":[100.0]*n,"raw_high":[101.0]*n,
            "raw_low":[99.0]*n,"raw_close":[100.0]*n,
            "corporate_action_window":[False]*n,
            "distance_sma_20":[0.01]*n,
            "distance_sma_50":[0.05]*n,
            "distance_sma_200":[-0.01]*n,
            "rsi14":[35.0]*n,
            "relative_volume20":[1.3]*n,
            "return_1":[-0.01]*n,
            "stoch_k14":[20.0]*n,
            "close_location":[0.1]*n,
            "obv_flow_ratio_5":[-0.1]*n,
            "intraday_range_pct":[0.02]*n,
            "macd_hist_change_3":[-0.1]*n,
            "sma50_slope_10":[0.1]*n,
        })

    def test_low_signal_filtered_above_sma50_four_percent(self):
        d=self._frame()
        low, high, risk=build_candidate_signals(d)
        self.assertFalse(bool(low[-1]))

    def test_risk_requires_sixty_of_eighty_and_rsi_below_40(self):
        d=self._frame(100)
        d["distance_sma_200"]=0.01
        d.loc[20:79,"distance_sma_200"]=-0.01
        low, high, risk=build_candidate_signals(d)
        self.assertTrue(bool(risk[79]))
        d.loc[79,"rsi14"]=40.0
        _,_,risk=build_candidate_signals(d)
        self.assertFalse(bool(risk[79]))

    def test_no_future_feature_in_risk_window(self):
        d=self._frame(100)
        d.loc[80:,"distance_sma_200"]=0.01
        _,_,risk=build_candidate_signals(d)
        self.assertFalse(bool(risk[79]))
        self.assertFalse(bool(risk[99]))


if __name__=="__main__":
    unittest.main()
