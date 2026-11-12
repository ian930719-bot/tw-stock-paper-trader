import unittest
import pandas as pd
from tsmc_2330_trend_reentry_exploration import reentry_candidates

class TrendReentrySignalTests(unittest.TestCase):
 def frame(self,n=5):
  return pd.DataFrame({
   "raw_high":[100,101,102,103,104],"raw_close":[99,100,101,102,103],
   "distance_ema_20":[.01]*n,"distance_ema_50":[.01]*n,
   "ema20_slope_5":[.01]*n,"rsi14":[51,51,51,51,51],
   "macd_hist_pct":[.01]*n,"macd_hist_change_3":[.01]*n,
   "adx14":[20]*n,"corporate_action_window":[False]*n})
 def test_breakout_uses_only_previous_session_high(self):
  d=self.frame()
  sig=reentry_candidates(d)["prevhigh_breakout_ema20_rsi50_macd_cd10"][0]
  self.assertFalse(bool(sig[0]))
  self.assertFalse(bool(sig[1]))
  self.assertFalse(bool(sig[2]))
 def test_rsi_cross_is_causal(self):
  d=self.frame()
  d["rsi14"]=[49,51,51,51,51]
  sig=reentry_candidates(d)["rsi50_cross_ema20_macd_cd10"][0]
  self.assertTrue(bool(sig[1]))
  self.assertFalse(bool(sig[2]))
 def test_corporate_action_windows_are_excluded(self):
  d=self.frame()
  d.loc[2,"corporate_action_window"]=True
  sig=reentry_candidates(d)["ema20_rsi50_macdpositive_cd10"][0]
  self.assertFalse(bool(sig[2]))

if __name__=="__main__":unittest.main()
