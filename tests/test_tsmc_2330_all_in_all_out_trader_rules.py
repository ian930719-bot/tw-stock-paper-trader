import unittest
import pandas as pd
from tsmc_2330_all_in_all_out_trader_rules import simulate, forward_label, build_rules

class AllInAllOutTests(unittest.TestCase):
    def test_signals_execute_next_open_all_in_all_out(self):
        d=pd.DataFrame({
          "date":pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03","2020-01-04","2020-01-05"]),
          "raw_open":[10,10,11,12,12],"raw_high":[10,11,12,12,13],
          "raw_low":[9,10,10,11,11],"raw_close":[10,11,11,12,13]})
        low=pd.Series([True,False,False,False,False])
        high=pd.Series([False,False,True,False,False])
        metrics,trades,curve=simulate(d,low,high,start="2020-01-01",end="2020-01-05",capital=10000)
        self.assertEqual(len(trades),1)
        self.assertEqual(trades.iloc[0].entry_date,"2020-01-02")
        self.assertEqual(trades.iloc[0].exit_date,"2020-01-04")
        self.assertEqual(trades.iloc[0].exit_reason,"high_signal_next_open")
        self.assertGreater(int(curve.shares.max()),0)
        self.assertEqual(int(curve.shares.iloc[-1]),0)

    def test_new_capitulation_climax_rule_is_registered(self):
        frame=pd.DataFrame({
          "relative_volume20":[1.5],"distance_prior_low20":[.02],"obv_flow_ratio_5":[-.2],
          "return_1":[-.02],"close_location":[.2],"bearish_rsi_divergence20":[0],
          "bearish_macd_divergence20":[0],"bearish_volume_divergence20":[0],
          "stoch_k14":[15],"distance_sma_200":[.15],"intraday_range_pct":[.02],
          "macd_hist_change_3":[-.001],"sma50_slope_10":[.002]})
        rules=build_rules(frame)
        self.assertIn("capitulation_OBV_vs_climax_extension",rules)
        low,high=rules["capitulation_OBV_vs_climax_extension"]
        self.assertTrue(bool(low.iloc[0]))
        self.assertFalse(bool(high.iloc[0]))

    def test_forward_label_excludes_pivot_day(self):
        self.assertEqual(forward_label(8,[4],2).tolist(),
          [False,False,True,True,False,False,False,False])

if __name__=="__main__":
    unittest.main()
