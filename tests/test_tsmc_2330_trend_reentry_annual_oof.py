import unittest
import math
import pandas as pd

from tsmc_2330_trend_reentry_annual_oof import (
    adverse_fill, tick_size, _signals, _simulate, INITIAL_CAPITAL
)


class TrendReentryExecutionTests(unittest.TestCase):
    def test_twse_tick_bands(self):
        self.assertEqual(tick_size(9.99), 0.01)
        self.assertEqual(tick_size(10), 0.05)
        self.assertEqual(tick_size(50), 0.10)
        self.assertEqual(tick_size(100), 0.50)
        self.assertEqual(tick_size(500), 1.00)
        self.assertEqual(tick_size(1000), 5.00)

    def test_adverse_fill_is_tick_valid_and_adverse(self):
        buy = adverse_fill(1000.0, "BUY", 5)
        sell = adverse_fill(1000.0, "SELL", 5)
        self.assertGreaterEqual(buy, 1000.0)
        self.assertLessEqual(sell, 1000.0)
        self.assertAlmostEqual(buy / 5, round(buy / 5), places=8)
        self.assertAlmostEqual(sell / 1, round(sell / 1), places=8)

    def test_signal_array_alignment_and_no_lookahead(self):
        d = pd.DataFrame({
            "corporate_action_window":[False]*5,
            "relative_volume20":[1.3]*5,
            "return_1":[-0.01]*5,
            "stoch_k14":[20]*5,
            "close_location":[0.1]*5,
            "obv_flow_ratio_5":[-0.1]*5,
            "intraday_range_pct":[0.02]*5,
            "distance_sma_200":[0.2]*5,
            "macd_hist_change_3":[-0.1]*5,
            "sma50_slope_10":[0.1]*5,
            "macd_hist_pct":[-0.1,-0.05,0,0.1,0.2],
            "rsi14":[40,45,50,55,60],
            "distance_ema_20":[-0.1,-0.02,0.01,0.02,0.03],
            "raw_high":[10,11,12,13,14],
            "raw_open":[10]*5,
            "raw_close":[9,10,11,12,13],
        })
        low, high, re = _signals(d)
        self.assertEqual(len(low),5)
        self.assertTrue(bool(re["ema20_rsi_macd"][2]))
        self.assertFalse(bool(re["ema20_rsi_macd"][0]))

    def test_no_order_without_official_price_limit(self):
        d = pd.DataFrame({
            "date":pd.to_datetime(["2024-01-02","2024-01-03"]),
            "raw_open":[10.0,10.0],"raw_high":[10.1,10.1],
            "raw_low":[9.9,9.9],"raw_close":[10.0,10.0]
        })
        low=[True,False]; high=[False,False]
        re={"break_prev_high":[False,False]}
        sim=_simulate(d,low,high,re,["break_prev_high"]*2,{}, {},10,0,1)
        self.assertEqual(sim["blocked_missing_price_limit"],1)
        self.assertEqual(sim["trade_count"],0)


if __name__ == "__main__":
    unittest.main()
