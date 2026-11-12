import unittest
from tsmc_2330_regime_high_hold_annual_oof import select_policy,WIN_FLOOR
class OOFSelectionTests(unittest.TestCase):
 def metric(self,trades,win,cagr,mdd=-.2):
  return {"completed_trades":trades,"win_rate_completed_only":win,"cagr_mark_to_market":cagr,"max_drawdown":mdd}
 def test_fallback_when_floor_not_met(self):
  m={"baseline":self.metric(10,.9,.3),"strong_long_trend_pullback":self.metric(10,.85,.4)}
  self.assertEqual(select_policy(m),"baseline")
 def test_choose_higher_cagr_only_when_both_eligible(self):
  m={"baseline":self.metric(10,.9,.3),"strong_long_trend_pullback":self.metric(10,.9,.35)}
  self.assertEqual(select_policy(m),"strong_long_trend_pullback")
 def test_minimum_training_trades(self):
  m={"baseline":self.metric(4,1,.5),"strong_long_trend_pullback":self.metric(10,.9,.3)}
  self.assertEqual(select_policy(m),"strong_long_trend_pullback")
if __name__=="__main__":unittest.main()
