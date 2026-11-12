import unittest
import numpy as np
import pandas as pd
from tsmc_2330_foreign_trust_entry_oof import build_policies, eligible, choose

class ForeignTrustEntryOOFTests(unittest.TestCase):
 def test_flow_gates_only_use_foreign_and_trust(self):
  d=pd.DataFrame({"institutional_data_available":[1,1,1,0],
   "foreign_net_5d_lots":[1,-1,1,1],"foreign_net_20d_lots":[1,1,-1,1],
   "trust_net_5d_lots":[1,1,-1,1],"trust_net_20d_lots":[-1,1,1,1]})
  low=np.array([1,1,1,1],dtype=bool);high=np.zeros(4,dtype=bool);hold=np.ones(4,dtype=bool)
  p=build_policies(d,low,high,hold)
  self.assertEqual(p["foreign_5d_positive"][0].tolist(),[True,False,True,True])
  self.assertEqual(p["foreign_and_trust_5d_positive"][0].tolist(),[True,False,False,True])
  self.assertEqual(p["foreign_or_trust_5d_positive"][0].tolist(),[True,True,True,True])
  self.assertEqual(p["baseline"][0].tolist(),[True,True,True,True])
  self.assertIn("foreign_20d_positive_high_hold",p)
  self.assertEqual(p["high_hold_foreign_5d_positive"][0].tolist(),[True,True,True,True])
  self.assertEqual(p["high_hold_foreign_5d_positive"][1].tolist(),[True,False,True,True])
  self.assertEqual(p["trust_20d_positive"][0][3],True)
 def test_win_floor_and_trade_count_are_both_required(self):
  self.assertFalse(eligible({"completed_trades":4,"win_rate_completed_only":1.0,"cagr_mark_to_market":.5}))
  self.assertFalse(eligible({"completed_trades":5,"win_rate_completed_only":.86,"cagr_mark_to_market":.5}))
  self.assertTrue(eligible({"completed_trades":5,"win_rate_completed_only":.87,"cagr_mark_to_market":.5}))
 def test_selector_falls_back_to_baseline_if_none_eligible(self):
  self.assertEqual(choose({"x":{"completed_trades":3,"win_rate_completed_only":1,"cagr_mark_to_market":1}}),"baseline")
if __name__=="__main__":unittest.main()
