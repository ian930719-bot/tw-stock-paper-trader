import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_engine import STRATEGIES
from walk_forward import rolling_walk_forward, classify_market_regime, regime_performance, parameter_sensitivity


def sample_frame(n=820):
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = np.full(n, 100.0)
    frame = pd.DataFrame({
        "date": dates, "open": close.copy(), "high": close + 1.0,
        "low": close - 1.0, "close": close.copy(), "volume": np.full(n, 1_000_000),
    })
    frame["atr"] = 1.0
    return frame


class RollingWalkForwardTests(unittest.TestCase):
    def test_windows_are_non_overlapping_and_no_pretest_trades(self):
        frame = sample_frame(820)
        frame["symbol"] = "WF"
        frame["open"] = 100.0
        frame["high"] = 101.0
        frame["low"] = 99.0
        frame["close"] = 100.0
        frame["atr"] = 1.0

        def controlled_signals(z):
            entry = pd.Series(False, index=z.index)
            for idx in (299, 399, 499, 599, 699):
                if idx < len(z):
                    entry.iloc[idx] = True
            return {STRATEGIES[0]: (entry, pd.Series(False, index=z.index))}

        with patch("research_engine._signals", side_effect=controlled_signals):
            result = rolling_walk_forward(
                {"WF": frame}, STRATEGIES[0], initial_capital=1_000_000,
                lookback_days=300, test_days=100, step_days=100,
                warmup_bars=200, min_test_days=40, max_positions=1,
            )

        self.assertGreaterEqual(len(result["windows"]), 3)
        starts = pd.to_datetime(result["windows"]["測試開始"])
        ends = pd.to_datetime(result["windows"]["測試結束"])
        self.assertTrue((starts.iloc[1:].reset_index(drop=True) >
                         ends.iloc[:-1].reset_index(drop=True)).all())
        if not result["trades"].empty:
            entry_dates = pd.to_datetime(result["trades"]["進場日"])
            for _, window in result["windows"].iterrows():
                selected = result["trades"].loc[
                    result["trades"]["樣本外區段"] == window["區段"]
                ]
                if not selected.empty:
                    self.assertTrue((pd.to_datetime(selected["進場日"]) >=
                                     pd.Timestamp(window["測試開始"])).all())

    def test_rejects_overlapping_test_windows(self):
        frame = sample_frame(500)
        with self.assertRaisesRegex(ValueError, "不可小於"):
            rolling_walk_forward(
                {"A": frame}, STRATEGIES[0],
                lookback_days=300, test_days=100, step_days=50,
                warmup_bars=200,
            )

    def test_empty_input_returns_empty_result(self):
        result = rolling_walk_forward({}, STRATEGIES[0])
        self.assertTrue(result["windows"].empty)
        self.assertEqual(result["coverage"], 0)


    def test_market_regime_classification_and_regime_report(self):
        dates = pd.bdate_range("2020-01-01", periods=280)
        close = np.r_[np.linspace(80, 140, 220), np.linspace(140, 90, 60)]
        benchmark = pd.DataFrame({"date": dates, "close": close})
        labels = classify_market_regime(benchmark, ema_window=30, slope_window=5)
        self.assertIn("多頭", set(labels))
        self.assertIn("空頭", set(labels))

        equity = pd.DataFrame({
            "日期": dates[-40:],
            "資產": np.linspace(1_000_000, 1_040_000, 40),
            "區段": [1] * 40,
            "區段資產": np.linspace(1_000_000, 1_040_000, 40),
        })
        report = regime_performance(
            equity, benchmark, initial_capital=1_000_000,
            ema_window=30, slope_window=5,
        )
        self.assertGreater(len(report), 0)
        self.assertTrue(set(report["盤勢"]).issubset({"多頭", "空頭", "盤整／轉換"}))

    def test_parameter_sensitivity_runs_predeclared_grid(self):
        from unittest.mock import patch
        fake = {
            "windows": pd.DataFrame([{"區段": 1}]),
            "trades": pd.DataFrame([{"損益": 10}]),
            "metrics": {"總報酬（鏈結）": 0.1, "最大回撤（鏈結）": -0.05, "Sharpe（鏈結）": 1.2},
        }
        with patch("walk_forward.rolling_walk_forward", return_value=fake) as runner:
            result = parameter_sensitivity(
                {"A": sample_frame(500)}, STRATEGIES[0],
                risk_fractions=(0.0025, 0.005), exit_profiles=("baseline_atr2", "atr_3"),
                lookback_days=300, test_days=100, warmup_bars=200,
            )
        self.assertEqual(len(result), 4)
        self.assertEqual(runner.call_count, 4)
        self.assertEqual(set(result["出場規則"]), {"baseline_atr2", "atr_3"})

    def test_parameter_sensitivity_rejects_invalid_risk(self):
        with self.assertRaisesRegex(ValueError, "風險比例"):
            parameter_sensitivity(
                {}, STRATEGIES[0], risk_fractions=(0.5,),
                exit_profiles=("baseline_atr2",),
            )


if __name__ == "__main__":
    unittest.main()
