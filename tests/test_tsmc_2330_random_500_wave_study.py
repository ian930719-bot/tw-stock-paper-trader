import unittest

import numpy as np
import pandas as pd

from tsmc_2330_random_500_wave_study import (
    deterministic_sample, chronological_split, descriptive_metrics, attach_causal_features,
)


class Random500WaveStudyTests(unittest.TestCase):
    def test_sample_is_deterministic_and_without_replacement(self):
        dates = pd.bdate_range("2010-01-01", periods=20)
        waves = pd.DataFrame({
            "start_date": dates.strftime("%Y-%m-%d"),
            "start_confirmation_date": (dates + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "end_date": (dates + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
            "direction": ["UP", "DOWN"] * 10,
        })
        a = deterministic_sample(waves, n=10, seed=7)
        b = deterministic_sample(waves, n=10, seed=7)
        self.assertEqual(a[["start_date", "end_date"]].values.tolist(),
                         b[["start_date", "end_date"]].values.tolist())
        self.assertEqual(len(a), 10)
        self.assertEqual(len(a.drop_duplicates(["start_date", "end_date"])), 10)

    def test_split_is_chronological_and_purges_overlap(self):
        dates = pd.bdate_range("2010-01-01", periods=100)
        frame = pd.DataFrame({
            "start_confirmation_date": dates.strftime("%Y-%m-%d"),
            "end_date": (dates + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "direction": ["UP", "DOWN"] * 50,
            "scale": ["small", "medium"] * 50,
            "amplitude_pct": np.linspace(.03, .2, 100),
            "duration_sessions": np.arange(1, 101),
        })
        research, validation, start = chronological_split(frame)
        self.assertTrue((research["end_date"] < start).all())
        self.assertTrue((validation["start_confirmation_date"] >= start).all())
        self.assertGreaterEqual(len(research), 25)
        self.assertEqual(len(validation), 30)


    def test_forward_label_uses_confirmation_close_and_rejects_late_confirmation(self):
        features = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "close": [100.0, 101.0, 102.0],
            "raw_close": [100.0, 101.0, 102.0],
            "corporate_action_window": [False, False, False],
        })
        waves = pd.DataFrame([
            {
                "start_confirmation_index": 0,
                "start_confirmation_date": "2020-01-01",
                "end_date": "2020-01-03",
                "end_price": 110.0,
                "direction": "UP",
            },
            {
                "start_confirmation_index": 2,
                "start_confirmation_date": "2020-01-03",
                "end_date": "2020-01-02",
                "end_price": 90.0,
                "direction": "DOWN",
            },
        ])
        result, _ = attach_causal_features(waves, features)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["post_confirmation_direction"], "UP")
        self.assertAlmostEqual(result.iloc[0]["post_confirmation_return_pct"], 0.10)

    def test_descriptive_metrics(self):
        frame = pd.DataFrame({
            "direction": ["UP", "DOWN"],
            "scale": ["small", "large"],
            "amplitude_pct": [.05, .25],
            "duration_sessions": [3, 20],
        })
        result = descriptive_metrics(frame)
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["up_wave_share"], .5)


if __name__ == "__main__":
    unittest.main()
