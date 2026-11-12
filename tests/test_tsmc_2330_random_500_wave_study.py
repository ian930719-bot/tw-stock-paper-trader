import unittest

import numpy as np
import pandas as pd

from tsmc_2330_random_500_wave_study import (
    deterministic_sample, chronological_split, descriptive_metrics,
)


class Random500WaveStudyTests(unittest.TestCase):
    def test_sample_is_deterministic_and_without_replacement(self):
        waves = pd.DataFrame({
            "start_date": [f"2010-01-{i:02d}" for i in range(1, 21)],
            "end_date": [f"2010-02-{i:02d}" for i in range(1, 21)],
            "direction": ["UP", "DOWN"] * 10,
        })
        a = deterministic_sample(waves, n=10, seed=7)
        b = deterministic_sample(waves, n=10, seed=7)
        self.assertEqual(a[["start_date", "end_date"]].values.tolist(),
                         b[["start_date", "end_date"]].values.tolist())
        self.assertEqual(len(a), 10)
        self.assertEqual(len(a.drop_duplicates(["start_date", "end_date"])), 10)

    def test_split_is_chronological_and_purges_overlap(self):
        frame = pd.DataFrame({
            "start_confirmation_date": [f"2010-01-{i:02d}" for i in range(1, 101)],
            "end_date": [f"2010-01-{min(i+1, 31):02d}" for i in range(1, 101)],
            "direction": ["UP", "DOWN"] * 50,
            "scale": ["small", "medium"] * 50,
            "amplitude_pct": np.linspace(.03, .2, 100),
            "duration_sessions": np.arange(1, 101),
        })
        research, validation, start = chronological_split(frame)
        self.assertTrue((research["end_date"] < start).all())
        self.assertTrue((validation["start_confirmation_date"] >= start).all())

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
