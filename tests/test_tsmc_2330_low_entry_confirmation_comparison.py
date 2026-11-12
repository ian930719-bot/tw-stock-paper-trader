import unittest
import numpy as np
import pandas as pd
from tsmc_2330_4019_replication import simulate

class LowEntryConfirmationTests(unittest.TestCase):
 def frame(self,closes,opens=None):
  n=len(closes);opens=opens or closes
  return pd.DataFrame({"date":pd.date_range("2024-01-02",periods=n,freq="B"),
   "raw_open":[float(x) for x in opens],"raw_high":[max(float(o),float(c))+2 for o,c in zip(opens,closes)],
   "raw_low":[min(float(o),float(c))-2 for o,c in zip(opens,closes)],
   "raw_close":[float(x) for x in closes],"distance_sma_20":[0.]*n,
   "distance_ema_20":[0.]*n,"rsi14":[50.]*n})
 def run_sim(self,d,policy,low):
  z=np.zeros(len(d),dtype=bool)
  limits={x.strftime("%Y-%m-%d"):(150.,50.) for x in d.date}
  return simulate(d,np.array(low,dtype=bool),z,z,limits,{},10,0,len(d)-1,low_entry_policy=policy)
 def test_bullish_confirmation_delays_buy_by_one_session(self):
  d=self.frame([100,101,103,104,105],[105,100,102,103,104])
  low=[True,False,False,False,False]
  r=self.run_sim(d,"bullish_day_3",low)
  c=pd.DataFrame(r["equity_curve"])
  self.assertEqual(int(c.shares.iloc[1]),0)
  self.assertGreater(int(c.shares.iloc[2]),0)
 def test_immediate_policy_keeps_next_session_entry(self):
  d=self.frame([100,101,103,104,105],[105,100,102,103,104])
  r=self.run_sim(d,"immediate",[True,False,False,False,False])
  c=pd.DataFrame(r["equity_curve"])
  self.assertGreater(int(c.shares.iloc[1]),0)
 def test_confirmation_timeout_does_not_force_entry(self):
  d=self.frame([100,99,98,97,96,95,94],[105,100,99,98,97,96,95])
  r=self.run_sim(d,"bullish_day_3",[True,False,False,False,False,False,False])
  c=pd.DataFrame(r["equity_curve"])
  self.assertEqual(int(c.shares.sum()),0)

 def test_mfi40_reclaim_waits_for_cross_and_buys_next_open(self):
  d=self.frame([100,99,100,102,103,104],[105,100,99,100,102,103])
  d["mfi14"]=[30.,35.,41.,42.,43.,44.]
  r=self.run_sim(d,"mfi40_reclaim_5",[True,False,False,False,False,False])
  c=pd.DataFrame(r["equity_curve"])
  self.assertEqual(int(c.shares.iloc[2]),0)
  self.assertGreater(int(c.shares.iloc[3]),0)

if __name__=="__main__":unittest.main()
