import unittest
import pandas as pd
from tsmc_2330_ema20_reclaim_reentry import make_reentry_signal
class ReentrySignalTests(unittest.TestCase):
 def test_ema20_cross_with_trend_filters(self):
  d=pd.DataFrame({"distance_ema_20":[-.01,.01,.02],"rsi14":[48,50,51],"distance_sma_50":[.02,.03,.04],"sma50_slope_10":[.01,.01,.01],"adx14":[20,21,22],"plus_di14":[30,31,32],"minus_di14":[20,20,20],"corporate_action_window":[False,False,False]})
  s=make_reentry_signal(d)
  self.assertFalse(bool(s[0]));self.assertTrue(bool(s[1]));self.assertFalse(bool(s[2]))
 def test_corporate_action_excluded(self):
  d=pd.DataFrame({"distance_ema_20":[-.01,.01],"rsi14":[48,50],"distance_sma_50":[.02,.03],"sma50_slope_10":[.01,.01],"adx14":[20,21],"plus_di14":[30,31],"minus_di14":[20,20],"corporate_action_window":[False,True]})
  self.assertFalse(bool(make_reentry_signal(d)[1]))
if __name__=="__main__":unittest.main()
