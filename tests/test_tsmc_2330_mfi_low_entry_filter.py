import unittest
import pandas as pd
class MfiFilterTests(unittest.TestCase):
 def test_mfi_threshold_boundary(self):
  s=pd.Series([39.9,40.0,45.0,None])
  ok=(s>=40).fillna(False)
  self.assertEqual(ok.tolist(),[False,True,True,False])
if __name__=="__main__":unittest.main()
