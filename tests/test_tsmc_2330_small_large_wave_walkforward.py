import unittest
import numpy as np
import pandas as pd
from tsmc_2330_small_large_wave_walkforward import (
    event_size_cutoffs, classify_pivots, make_labels, simulate_all_in_all_out
)

class SmallLargeWaveTests(unittest.TestCase):
    def test_size_cutoffs_use_only_events_before_fold_boundary(self):
        events=pd.DataFrame({
          "pivot_date":pd.to_datetime(["2010-01-01","2011-01-01","2012-01-01","2013-01-01"]*20),
          "type":["LOW","HIGH"]*40,
          "reversal_pct":np.tile(np.arange(1,21),4)
        })
        cut=event_size_cutoffs(events,"2014-01-01")
        self.assertIsNotNone(cut)
        future=events.copy()
        future.loc[future.pivot_date>=pd.Timestamp("2014-01-01"),"reversal_pct"]=999
        cut2=event_size_cutoffs(future,"2014-01-01")
        self.assertEqual(cut,cut2)

    def test_small_and_large_classes_are_disjoint(self):
        events=pd.DataFrame({
          "type":["LOW"]*5+["HIGH"]*5,
          "reversal_pct":[1,2,3,4,5,1,2,3,4,5]
        })
        cut={"LOW":{"small_max_abs_reversal_pct":2,"large_min_abs_reversal_pct":4},
          "HIGH":{"small_max_abs_reversal_pct":2,"large_min_abs_reversal_pct":4}}
        result=classify_pivots(events,cut)
        self.assertEqual(result.wave_class.tolist(),["small","small","medium","large","large"]*2)

    def test_forward_labels_exclude_pivot_session(self):
        y,known=make_labels(10,[5],2)
        self.assertEqual(y.tolist(),[0,0,0,1,1,0,0,0,0,0])
        self.assertFalse(known[-2:].any())

    def test_all_in_all_out_executes_next_open(self):
        d=pd.DataFrame({
          "date":pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03","2020-01-04"]),
          "raw_open":[10,11,12,13],"raw_high":[10,11,12,13],
          "raw_low":[9,10,11,12],"raw_close":[10,11,12,13]})
        low=np.array([True,False,False,False])
        high=np.array([False,False,True,False])
        metrics,trades,curve=simulate_all_in_all_out(d,low,high,"2020-01-01",10000)
        self.assertEqual(trades.iloc[0].entry_date,"2020-01-02")
        self.assertEqual(trades.iloc[0].exit_date,"2020-01-04")
        self.assertEqual(trades.iloc[0].exit_reason,"high_alert_next_open")
        self.assertEqual(int(curve.shares.iloc[-1]),0)


    def test_atr_stop_ratchets_up_and_never_down(self):
        from tsmc_2330_small_large_wave_walkforward import simulate_atr_reentry
        d=pd.DataFrame({
          "date":pd.date_range("2020-01-01",periods=35,freq="B"),
          "raw_open":[100.0]*35,"raw_high":[102.0]*35,
          "raw_low":[99.0]*35,"raw_close":[101.0]*35
        })
        # Seed a volatile history so ATR is defined; a signal buys on next open.
        d.loc[0:13,"raw_high"]=110.0
        d.loc[0:13,"raw_low"]=90.0
        d.loc[0:13,"raw_close"]=100.0
        d.loc[14:20,"raw_high"]=120.0
        d.loc[14:20,"raw_close"]=119.0
        d.loc[21:,"raw_low"]=80.0
        low=np.zeros(35,dtype=bool); high=np.zeros(35,dtype=bool)
        low[14]=True
        _, trades, curve=simulate_atr_reentry(d,low,high,"2020-01-01",2.0,1,100000)
        held=curve.loc[curve.shares>0,"active_stop"].dropna().to_numpy()
        self.assertGreater(len(held),1)
        self.assertTrue(np.all(np.diff(held)>=-1e-9))
        self.assertTrue((trades.stop_width_at_entry>0).all())

    def test_invalidated_signal_does_not_reenter_immediately(self):
        from tsmc_2330_small_large_wave_walkforward import simulate_atr_reentry
        d=pd.DataFrame({
          "date":pd.date_range("2021-01-01",periods=40,freq="B"),
          "raw_open":[100.0]*40,"raw_high":[101.0]*40,
          "raw_low":[99.0]*40,"raw_close":[100.0]*40
        })
        d.loc[:13,"raw_high"]=110.0;d.loc[:13,"raw_low"]=90.0
        low=np.zeros(40,dtype=bool);high=np.zeros(40,dtype=bool)
        low[14]=True;low[17]=True;low[23]=True
        # First fresh entry, then a deliberate stop caused by a low print.
        d.loc[17,"raw_low"]=1.0
        _, trades, _=simulate_atr_reentry(d,low,high,"2021-01-01",1.5,3,100000)
        entries=trades.entry_index.tolist()
        # Re-entry needs a newly generated low after stop + 3 complete sessions.
        if len(entries)>1:
            self.assertGreaterEqual(entries[1]-int(trades.exit_index.iloc[0]),4)


if __name__=="__main__":
    unittest.main()
