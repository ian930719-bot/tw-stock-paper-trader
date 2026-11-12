import unittest
import pandas as pd
from tsmc_2330_regime_conditioned_high_hold import gates
class RegimeGateTests(unittest.TestCase):
 def frame(self):
  return pd.DataFrame({"corporate_action_window":[False,False,False],"distance_ema_20":[.01,.01,-.01],"rsi14":[55,70,55],"plus_di14":[30,30,30],"minus_di14":[20,20,20],"distance_sma_200":[.30,.30,.30]})
 def test_strong_regime_gate(self):
  g=gates(self.frame())
  self.assertTrue(bool(g["strong_long_trend_pullback"][0]))
  self.assertFalse(bool(g["strong_long_trend_pullback"][1]))
  self.assertFalse(bool(g["strong_long_trend_pullback"][2]))
 def test_general_gate_does_not_require_sma200_distance(self):
  d=self.frame();d.loc[0,"distance_sma_200"]=.10
  g=gates(d)
  self.assertFalse(bool(g["strong_long_trend_pullback"][0]))
  self.assertTrue(bool(g["general_ema20_directional_pullback"][0]))
if __name__=="__main__":unittest.main()
