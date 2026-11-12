import unittest
import pandas as pd
from tsmc_2330_high_false_signal_exploration import hold_candidates
class HighHoldTests(unittest.TestCase):
 def test_trend_continuation_gate_requires_all_conditions(self):
  d=pd.DataFrame({"distance_ema_20":[.02,.02],"ema20_slope_5":[.01,.01],"rsi14":[60,57],"macd_hist_pct":[.01,.01],"adx14":[25,25],"plus_di14":[30,30],"minus_di14":[20,20],"distance_prior_high20":[-.01,-.01],"corporate_action_window":[False,False]})
  h=hold_candidates(d)["hold_ema20_rsi58_macd_positive"]
  self.assertTrue(bool(h[0]));self.assertFalse(bool(h[1]))
 def test_corporate_action_excluded(self):
  d=pd.DataFrame({"distance_ema_20":[.02],"ema20_slope_5":[.01],"rsi14":[60],"macd_hist_pct":[.01],"adx14":[25],"plus_di14":[30],"minus_di14":[20],"distance_prior_high20":[-.01],"corporate_action_window":[True]})
  self.assertFalse(bool(hold_candidates(d)["hold_ema20_rsi58_macd_positive"][0]))
if __name__=="__main__":unittest.main()
