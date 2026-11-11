import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from tsmc_2330_trader_strategy_event_audit import paired_metrics, rules, run

class TraderStrategyAuditTests(unittest.TestCase):
    def test_low_volume_rule_prefers_event_over_control(self):
        frame = pd.DataFrame({
            "relative_volume20": [0.5, 1.2, 0.6, 1.1],
            "distance_prior_low20": [0.02, 0.02, 0.04, 0.04],
            "control_relative_volume20": [1.1, 1.1, 1.0, 1.0],
            "control_distance_prior_low20": [0.02, 0.02, 0.04, 0.04],
        })
        fn = rules()["LOW_volume_dryup_near_low"][1]
        metrics = paired_metrics(frame, fn, np.random.default_rng(1), 100)
        self.assertEqual(metrics["event_hit_rate"], 0.5)
        self.assertEqual(metrics["matched_control_hit_rate"], 0.0)
        self.assertGreater(metrics["paired_risk_difference"], 0)

    def test_run_writes_outputs(self):
        source = Path(os.environ.get("TSMC_500_SNAPSHOTS", "/mnt/data/500study_final/tsmc_2330_500_pivot_snapshots.csv"))
        if not source.exists():
            self.skipTest("frozen 500-pivot snapshot artifact unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            summary = run(str(source), tmp, n_boot=20)
            self.assertEqual(len(summary), 55)
            self.assertIn("q_bh_all_rule_lead_tests", summary.columns)
            self.assertIn("n_control_date_clusters", summary.columns)
            self.assertIn("cluster_sign_test_p_unadjusted", summary.columns)
            self.assertTrue((Path(tmp) / "tsmc_2330_trader_rule_event_control_summary.csv").exists())
            self.assertTrue((Path(tmp) / "tsmc_2330_trader_rule_year_stability.csv").exists())

if __name__ == "__main__":
    unittest.main()
