import unittest

import numpy as np
import pandas as pd

from tsmc_2330_random_500_wave_study import (
    deterministic_sample, chronological_split, descriptive_metrics,
)


class Random500WaveStudyTests(unittest.TestCase):
    def test_sample_is_deterministic_and_without_replacement(self):
        dates = pd.bdate_range("2010-01-01", periods=20)
        waves = pd.DataFrame({
            "start_date": dates.strftime("%Y-%m-%d"),
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
        self.assertGreaterEqual(len(research), 50)
        self.assertEqual(len(validation), 30)

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
