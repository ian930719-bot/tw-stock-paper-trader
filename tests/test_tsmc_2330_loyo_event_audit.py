import tempfile
import unittest
from pathlib import Path
import pandas as pd
from tsmc_2330_loyo_event_audit import run_audit


class LeaveOneYearOutTests(unittest.TestCase):
    def test_years_are_held_out_and_outputs_written(self):
        rows = []
        for kind, feats in {
            "HIGH": ["distance_sma_200", "sma50_slope_10", "overnight_gap",
                     "bearish_volume_divergence20", "return_1", "foreign_net_to_volume",
                     "trust_net_to_volume", "trust_net_negative_streak"],
            "LOW": ["distance_sma_200", "minus_di14", "sma50_slope_10",
                    "overnight_gap", "foreign_net_to_volume", "trust_net_to_volume",
                    "trust_net_negative_streak"],
        }.items():
            for year in range(2012, 2021):
                for event in range(6):
                    row = {"event_no": f"{kind}-{year}-{event}",
                           "pivot_type": kind, "pivot_date": f"{year}-06-20",
                           "feature_date": f"{year}-06-10",
                           "control_date": f"{year}-06-11", "lead_sessions": 5}
                    for j, feature in enumerate(feats):
                        value = event * 0.1 + j + (year - 2012) * 0.01
                        row[feature] = value + 0.5
                        row[f"control_{feature}"] = value
                    rows.append(row)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "snapshots.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            result = run_audit(source, root / "out")
            self.assertEqual(result["input_rows"], len(rows))
            detail = pd.read_csv(root / "out/tsmc_2330_loyo_fold_metrics.csv")
            self.assertEqual(detail.heldout_year.nunique(), 9)
            self.assertEqual(set(detail.model), {
                "technical", "technical_plus_institutional"})
            self.assertTrue((detail.train_events > 0).all())
            self.assertTrue((detail.test_events == 6).all())


if __name__ == "__main__":
    unittest.main()
