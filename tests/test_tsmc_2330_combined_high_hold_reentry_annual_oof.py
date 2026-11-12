import unittest
from tsmc_2330_combined_high_hold_reentry_annual_oof import select_policy,WIN_FLOOR
class CombinedOOFTests(unittest.TestCase):
 def metric(self,n,w,c,m=-.2):return {"completed_trades":n,"win_rate_completed_only":w,"cagr_mark_to_market":c,"max_drawdown":m}
 def test_rejects_below_win_floor(self):
  self.assertEqual(select_policy({"baseline":self.metric(10,.9,.3),"ema20_reclaim_reentry":self.metric(10,WIN_FLOOR-.01,.5)}),"baseline")
 def test_selects_highest_eligible_cagr(self):
  x={"baseline":self.metric(10,.9,.3),"ema20_reclaim_reentry":self.metric(10,.9,.35),"combined_high_hold_and_reentry":self.metric(10,.9,.34)}
  self.assertEqual(select_policy(x),"ema20_reclaim_reentry")
 def test_minimum_trade_count(self):
  x={"baseline":self.metric(10,.9,.3),"strong_trend_high_hold":self.metric(4,1,.8)}
  self.assertEqual(select_policy(x),"baseline")
if __name__=="__main__":unittest.main()
