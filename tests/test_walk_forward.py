import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_engine import STRATEGIES
from walk_forward import rolling_walk_forward


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


if __name__ == "__main__":
    unittest.main()
