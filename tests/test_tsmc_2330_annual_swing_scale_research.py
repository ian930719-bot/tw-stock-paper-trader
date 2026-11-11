import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_annual_swing_scale_research import (
    build_wave_legs, build_pivot_characteristics, run,
)


class AnnualSwingScaleResearchTests(unittest.TestCase):
    def setUp(self):
        n = 30
        close = np.linspace(100, 120, n)
        self.features = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=n),
            "open": close * .999, "high": close * 1.01,
            "low": close * .99, "close": close,
            "corporate_action_window": False,
            "rsi14": np.linspace(30, 70, n),
            "atr_pct14": np.full(n, .02),
            "broker_net_buy": [np.nan] * n,
        })
        self.events = pd.DataFrame([
            {"threshold_id":"p0.03","type":"LOW","pivot_index":2,"confirmation_index":4,
             "pivot_date":self.features.date.iloc[2].date().isoformat(),
             "confirmation_date":self.features.date.iloc[4].date().isoformat(),
             "pivot_price":100.0,"confirmation_close":104.0,"reversal_pct":.04,"confirmation_lag":2},
            {"threshold_id":"p0.03","type":"HIGH","pivot_index":10,"confirmation_index":12,
             "pivot_date":self.features.date.iloc[10].date().isoformat(),
             "confirmation_date":self.features.date.iloc[12].date().isoformat(),
             "pivot_price":110.0,"confirmation_close":105.0,"reversal_pct":.045,"confirmation_lag":2},
            {"threshold_id":"p0.03","type":"LOW","pivot_index":20,"confirmation_index":22,
             "pivot_date":self.features.date.iloc[20].date().isoformat(),
             "confirmation_date":self.features.date.iloc[22].date().isoformat(),
             "pivot_price":80.0,"confirmation_close":84.0,"reversal_pct":.05,"confirmation_lag":2},
        ])

    def test_wave_legs_assign_small_medium_large_scale_by_realized_amplitude(self):
        events = self.events.copy()
        events.loc[1, "pivot_price"] = 106.0
        legs = build_wave_legs(events, self.features)
        self.assertEqual(legs["scale"].tolist(), ["small", "large"])
        self.assertAlmostEqual(legs.iloc[0]["amplitude_pct"], .06)
        self.assertGreater(legs.iloc[1]["amplitude_pct"], .20)

    def test_pivot_characteristics_separate_pivot_lead_and_confirmation_features(self):
        legs = build_wave_legs(self.events, self.features)
        pivots = build_pivot_characteristics(self.events, self.features, legs)
        low = pivots.loc[pivots["type"] == "LOW"].iloc[0]
        self.assertEqual(low["lead1_rsi14"], self.features.loc[1, "rsi14"])
        self.assertEqual(low["pivot_rsi14"], self.features.loc[2, "rsi14"])
        self.assertEqual(low["confirm_rsi14"], self.features.loc[4, "rsi14"])
        self.assertTrue(pd.isna(low["pivot_broker_net_buy"]))

    def test_run_writes_annual_outputs_and_partial_year_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            feature_path = Path(tmp) / "features.csv"
            event_path = Path(tmp) / "events.csv"
            out = Path(tmp) / "out"
            self.features.to_csv(feature_path, index=False)
            self.events.to_csv(event_path, index=False)
            summary = run(str(feature_path), str(event_path), str(out))
            self.assertEqual(summary["wave_legs"], 2)
            annual = pd.read_csv(out / "tsmc_2330_annual_scale_summary.csv")
            medium_up = annual.loc[annual["pivot_type"] == "WAVE_MEDIUM_UP"]
            large_down = annual.loc[annual["pivot_type"] == "WAVE_LARGE_DOWN"]
            high_pivots = annual.loc[annual["pivot_type"] == "HIGH"]
            self.assertEqual(int(medium_up.iloc[0]["pivot_count"]), 1)
            self.assertEqual(int(large_down.iloc[0]["pivot_count"]), 1)
            self.assertEqual(int(high_pivots.iloc[0]["pivot_count"]), 1)
            self.assertTrue((out / "tsmc_2330_annual_indicator_patterns.csv").exists())


if __name__ == "__main__":
    unittest.main()
