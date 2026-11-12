import unittest
import pandas as pd
from tsmc_2330_atr2_low_rule_oof import select_rule, summarize_rows, FEATURES

class ATR2LowRuleOOFTests(unittest.TestCase):
    def test_rule_selection_uses_training_data_and_supported_coverage(self):
        rows=[]
        for i in range(60):
            x=float(i)
            rows.append({"confirm_rsi14":x,"quality_label":int(x<20)})
        d=pd.DataFrame(rows)
        for feature in FEATURES:
            if feature not in d:
                d[feature]=0.0
        rule=select_rule(d)
        self.assertIsNotNone(rule)
        self.assertEqual(rule["feature"],"confirm_rsi14")
        self.assertEqual(rule["direction"],"low")
        self.assertGreaterEqual(rule["train_n"],8)

    def test_oof_summary_counts_only_selected_events(self):
        s=summarize_rows([
            {"actual_quality_label":1,"selected_by_rule":True},
            {"actual_quality_label":0,"selected_by_rule":True},
            {"actual_quality_label":1,"selected_by_rule":False},
            {"actual_quality_label":0,"selected_by_rule":False},
        ])
        self.assertEqual(s["n"],4)
        self.assertEqual(s["selected_n"],2)
        self.assertAlmostEqual(s["precision"],0.5)
        self.assertAlmostEqual(s["recall"],0.5)

if __name__=="__main__":
    unittest.main()
