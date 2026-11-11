import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from tsmc_swing_research import build_causal_features, detect_confirmed_pivots, build_lead_lag, load_tsmc
from corporate_action_adjustments import apply_back_adjustments


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
        cols = ["return_20", "distance_sma_50", "atr_pct14", "rsi14", "relative_volume20",
                "bearish_rsi_divergence20", "bullish_rsi_divergence20",
                "bearish_macd_divergence20", "bullish_macd_divergence20",
                "rsi_divergence_strength_high20", "macd_divergence_strength_low20"]
        pd.testing.assert_frame_equal(a.loc[:249, cols], b.loc[:249, cols])

    def test_expanded_indicator_panel_and_ranges(self):
        close = 100 + np.cumsum(np.sin(np.arange(340) / 7) * 1.2 + 0.15)
        bars = self.make_bars(close)
        bars["volume"] = 80000 + (np.arange(len(bars)) % 17) * 3500
        features = build_causal_features(bars)
        required = [
            "ema_5", "ema_20", "ema_50", "ema_200", "stoch_k14",
            "stoch_d3", "cci20", "williams_r14", "adx14", "plus_di14",
            "minus_di14", "obv", "mfi14", "chaikin_money_flow20",
            "volume_sma5_ratio", "volume_sma20_ratio", "volume_zscore20",
            "rolling_vwap_proxy_20", "roc_20", "ppo_hist",
            "donchian_position20",
        ]
        self.assertTrue(set(required).issubset(features.columns))
        valid_stoch = features["stoch_k14"].dropna()
        self.assertTrue(((valid_stoch >= 0) & (valid_stoch <= 100)).all())
        valid_williams = features["williams_r14"].dropna()
        self.assertTrue(((valid_williams >= -100) & (valid_williams <= 0)).all())
        valid_mfi = features["mfi14"].dropna()
        self.assertTrue(((valid_mfi >= 0) & (valid_mfi <= 100)).all())
        self.assertTrue(np.isfinite(features["ema_200"].dropna()).all())

    def test_expanded_indicators_are_causal(self):
        original = self.make_bars(100 + np.cumsum(np.sin(np.arange(340) / 9)))
        original["volume"] = 50000 + (np.arange(len(original)) % 13) * 2000
        changed = original.copy()
        changed.loc[280:, ["open", "high", "low", "close", "volume"]] *= 1.7
        a, b = build_causal_features(original), build_causal_features(changed)
        cols = [
            "ema_20", "distance_ema_50", "stoch_k14", "cci20", "williams_r14",
            "adx14", "obv", "mfi14", "chaikin_money_flow20",
            "volume_zscore20", "rolling_vwap_proxy_20", "ppo_hist",
            "donchian_position20",
        ]
        pd.testing.assert_frame_equal(a.loc[:279, cols], b.loc[:279, cols])

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

    def test_corporate_action_adjustment_only_changes_pre_event_prices(self):
        bars = self.make_bars(np.linspace(100, 150, 300))
        events = pd.DataFrame([{
            "symbol": "2330", "date": bars.loc[200, "date"].date().isoformat(),
            "adjustment_ratio": 0.95, "conflict": False,
        }])
        adjusted = apply_back_adjustments(bars, events)
        self.assertAlmostEqual(adjusted.loc[199, "adj_close"], bars.loc[199, "close"] * 0.95)
        self.assertAlmostEqual(adjusted.loc[200, "adj_close"], bars.loc[200, "close"])
        self.assertTrue(adjusted.loc[200:220, "corporate_action_window"].all())
        self.assertFalse(adjusted.loc[221, "corporate_action_window"])

    def test_volume_features_are_masked_after_corporate_action(self):
        bars = self.make_bars(np.linspace(100, 150, 320))
        bars["corporate_action_window"] = False
        bars.loc[250:270, "corporate_action_window"] = True
        features = build_causal_features(bars)
        self.assertTrue(features.loc[250:270, "relative_volume20"].isna().all())
        self.assertTrue(features.loc[250:270, "turnover_ratio20"].isna().all())

    def test_all_share_volume_features_mask_corporate_action_window(self):
        close = 100 + np.cumsum(np.sin(np.arange(340) / 8) * 0.8)
        bars = self.make_bars(close)
        bars["volume"] = 60000 + (np.arange(len(bars)) % 19) * 2500
        bars["corporate_action_window"] = False
        bars.loc[250:270, "corporate_action_window"] = True
        features = build_causal_features(bars)
        cols = [
            "obv_change_5", "obv_flow_ratio_5", "obv_flow_ratio_20", "mfi14",
            "chaikin_ad_line", "chaikin_money_flow20", "volume_sma5_ratio",
            "volume_sma20_ratio", "volume_zscore20", "rolling_vwap_proxy_5",
            "rolling_vwap_proxy_10", "rolling_vwap_proxy_20",
        ]
        self.assertTrue(features.loc[250:270, cols].isna().all().all())

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
