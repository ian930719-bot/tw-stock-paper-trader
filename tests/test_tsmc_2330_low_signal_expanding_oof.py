import unittest
import numpy as np
from tsmc_2330_low_signal_expanding_oof import dedup_indices,select_policy if False else dedup_indices

class LowSignalOOFTests(unittest.TestCase):
 def test_dedup_respects_fixed_ten_session_gap(self):
  self.assertEqual(dedup_indices(np.array([1,1,0,1,0,0,0,0,0,0,1,0],dtype=bool),10),[0,10])
if __name__=="__main__":unittest.main()
