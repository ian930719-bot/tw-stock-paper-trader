import unittest
from tsmc_2330_all_low_signal_diagnostic import val
class LowDiagnosticTests(unittest.TestCase):
 def test_numeric(self):self.assertEqual(val(.5),.5)
 def test_nan(self):self.assertIsNone(val(float("nan")))
if __name__=="__main__":unittest.main()
