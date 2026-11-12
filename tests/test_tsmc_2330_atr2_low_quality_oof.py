import unittest, tempfile
from pathlib import Path
import pandas as pd
from tsmc_2330_atr2_low_quality_oof import FEATURES, TARGET

class ContractTests(unittest.TestCase):
    def test_features_are_confirmation_only(self):
        self.assertTrue(all(x.startswith("confirm_") or x in {"confirmation_lag_sessions","reversal_at_confirmation_pct"} for x in FEATURES))
        self.assertFalse(any("pivot_" in x for x in FEATURES))
    def test_target_is_forward_label_name(self):
        self.assertEqual(TARGET,"quality_label")

if __name__=="__main__": unittest.main()
