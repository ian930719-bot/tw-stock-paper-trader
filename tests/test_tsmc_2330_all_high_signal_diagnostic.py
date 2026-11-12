import unittest
from tsmc_2330_all_high_signal_diagnostic import val
class AllHighTests(unittest.TestCase):
 def test_nonfinite(self):self.assertIsNone(val(float("nan")))
 def test_float(self):self.assertEqual(val(.2),.2)
if __name__=="__main__":unittest.main()
