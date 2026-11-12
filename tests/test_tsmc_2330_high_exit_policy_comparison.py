import unittest
import numpy as np
import pandas as pd
from tsmc_2330_4019_replication import simulate

class HighExitPolicyTests(unittest.TestCase):
 def test_all_policies_run_without_positions(self):
  d=pd.DataFrame({
   "date":pd.to_datetime(["2020-01-02","2020-01-03"]),
   "raw_open":[100.,101.],"raw_high":[101.,102.],"raw_low":[99.,100.],"raw_close":[100.,101.],
   "distance_sma_20":[.01,.01],"distance_ema_20":[.01,.01],"rsi14":[55.,55.]})
  z=np.zeros(2,dtype=bool)
  for policy in ["handoff_or","and_sma20_rsi55","ema20_rsi55","prior5low"]:
   result=simulate(d,z,z,z,{}, {},10,0,1,high_exit_policy=policy)
   self.assertEqual(result["high_exit_policy"],policy)
 def test_unknown_policy_rejected(self):
  d=pd.DataFrame({"date":pd.to_datetime(["2020-01-02","2020-01-03"]),"raw_open":[100.,101.],"raw_high":[101.,102.],"raw_low":[99.,100.],"raw_close":[100.,101.],"distance_sma_20":[.01,.01],"distance_ema_20":[.01,.01],"rsi14":[55.,55.]})
  z=np.zeros(2,dtype=bool)
  with self.assertRaises(ValueError):
   simulate(d,z,z,z,{}, {},10,0,1,high_exit_policy="invented")
if __name__=="__main__":unittest.main()
