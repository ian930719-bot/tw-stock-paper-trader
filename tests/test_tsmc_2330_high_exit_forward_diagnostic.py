import unittest
import pandas as pd
from tsmc_2330_high_exit_forward_diagnostic import clean
class DiagnosticTests(unittest.TestCase):
 def test_nonfinite_becomes_none(self):
  self.assertIsNone(clean(float("nan")))
  self.assertIsNone(clean(float("inf")))
 def test_numeric_is_preserved(self):
  self.assertEqual(clean(0.125),0.125)
if __name__=="__main__":unittest.main()
