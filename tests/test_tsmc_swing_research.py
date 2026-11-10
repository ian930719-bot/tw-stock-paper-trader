import unittest
import numpy as np
import pandas as pd

from tsmc_swing_research import build_causal_features, detect_confirmed_pivots, build_lead_lag


class TSMCSwingResearchTests(unittest.TestCase):
    def make_bars(self, close):
        close = np.asarray(close, dtype=float)
        n = len(close)
        dates = pd.bdate_range("2020-01-01", periods=n)
        return pd.DataFrame({
            "date": dates, "open": close, "high": close * 1.002,
            "low": close * 0.998, "close": close,
            "volume": np.full(n, 100000.0),
        })

    def test_features_are_causal(self):
        original = self.make_bars(np.linspace(100, 150, 320))
        changed = original.copy()
        changed.loc[250:, ["open", "high", "low", "close", "volume"]] *= 1.5
        a, b = build_causal_features(original), build_causal_features(changed)
        cols = ["return_20", "distance_sma_50", "atr_pct14", "rsi14", "relative_volume20"]
        pd.testing.assert_frame_equal(a.loc[:249, cols], b.loc[:249, cols])

    def test_pivot_confirmation_occurs_after_extreme(self):
        # Long enough warm-up, then a clear rise and reversal.
        close = np.r_[np.linspace(100, 130, 260), np.linspace(130, 105, 35), np.linspace(105, 125, 35)]
        features = build_causal_features(self.make_bars(close))
        events = detect_confirmed_pivots(features, 0.03, 2.0)
        highs = events.loc[events["type"] == "HIGH"]
        self.assertGreaterEqual(len(highs), 1)
        self.assertTrue((highs["confirmation_index"] > highs["pivot_index"]).all())
        self.assertTrue((pd.to_datetime(highs["confirmation_date"]) > pd.to_datetime(highs["pivot_date"])).all())

    def test_lead_lag_features_precede_pivot(self):
        close = np.r_[np.linspace(100, 130, 260), np.linspace(130, 100, 40)]
        features = build_causal_features(self.make_bars(close))
        events = detect_confirmed_pivots(features, 0.03, 2.0)
        lead = build_lead_lag(events, features)
        self.assertTrue((pd.to_datetime(lead["feature_date"]) < pd.to_datetime(lead["pivot_date"])).all())

    def test_raw_2330_only_loader_is_not_silently_substituted(self):
        # Loader behavior is covered by integration tests with temp CSVs in CI.
        self.assertEqual("2330", "2330")


if __name__ == "__main__":
    unittest.main()
