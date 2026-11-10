import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from tsmc_swing_research import build_causal_features, detect_confirmed_pivots, build_lead_lag, load_tsmc


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

    def test_loader_filters_to_2330_and_rejects_other_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = pd.bdate_range("2020-01-01", periods=260)
            frames = []
            for symbol, base in (("2330", 100.0), ("2317", 50.0)):
                close = np.linspace(base, base * 1.2, len(dates))
                frames.append(pd.DataFrame({
                    "symbol": symbol, "date": dates, "open": close,
                    "high": close * 1.01, "low": close * 0.99,
                    "close": close, "volume": 100000,
                }))
            pd.concat(frames).to_csv(root / "panel.csv", index=False)
            result = load_tsmc(str(root / "panel.csv"))
            self.assertEqual(set(result["symbol"]), {"2330"})
            self.assertEqual(len(result), 260)
            pd.concat(frames[1:]).to_csv(root / "other.csv", index=False)
            with self.assertRaisesRegex(ValueError, "No TSMC 2330"):
                load_tsmc(str(root / "other.csv"))


if __name__ == "__main__":
    unittest.main()
