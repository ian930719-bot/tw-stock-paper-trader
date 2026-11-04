"""Unit tests for research_engine portfolio mechanics."""
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch

from research_engine import STRATEGIES, daily_strategy_returns, performance, shared_capital_backtest


def sample_frame(n=620, drift=0.0008, phase=0.0):
    dates = pd.bdate_range("2022-01-03", periods=n)
    t = np.arange(n, dtype=float)
    close = 50 * np.exp(drift * t + 0.015 * np.sin(t / 9 + phase))
    open_ = close * (1 - 0.001 * np.sin(t / 5))
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = 2_000_000 + (t % 17) * 15_000
    frame = pd.DataFrame({
        "date": dates, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })
    frame["ema21"] = frame.close.ewm(span=21, adjust=False).mean()
    frame["ema50"] = frame.close.ewm(span=50, adjust=False).mean()
    frame["up"] = frame.ema21 > frame.ema21.shift(5)
    frame["h20"] = frame.high.shift(1).rolling(20).max()
    frame["v20"] = frame.volume.rolling(20).mean()
    tr = pd.concat([
        frame.high-frame.low,
        (frame.high-frame.close.shift()).abs(),
        (frame.low-frame.close.shift()).abs(),
    ], axis=1).max(axis=1)
    frame["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    return frame


class ResearchEngineTests(unittest.TestCase):
    def test_empty_performance(self):
        result = performance(pd.Series(dtype=float), 100000)
        self.assertEqual(result["期末資產"], 100000)
        self.assertEqual(result["交易日數"], 0)

    def test_single_strategy_returns_are_finite(self):
        frame = sample_frame()
        for strategy in STRATEGIES:
            returns = daily_strategy_returns(frame, strategy)
            self.assertTrue(np.isfinite(returns.to_numpy()).all())

    def test_shared_capital_respects_position_limit(self):
        frames = {f"S{i}": sample_frame(phase=i * .3) for i in range(7)}
        result = shared_capital_backtest(
            frames, STRATEGIES[2], initial_capital=1_000_000,
            max_positions=3, warmup=260,
        )
        self.assertEqual(result["coverage"]["symbols_used"], 7)
        self.assertIsInstance(result["equity"], pd.DataFrame)
        self.assertGreaterEqual(result["metrics"]["期末資產"], 0)
        if not result["equity"].empty:
            self.assertLessEqual(int(result["equity"]["持倉數"].max()), 3)
            self.assertTrue((result["equity"]["現金"] >= -1e-6).all())
            self.assertTrue(np.isfinite(result["equity"]["資產"].to_numpy()).all())
        self.assertTrue(np.isfinite(list(result["metrics"].values())).all())
        if not result["trades"].empty:
            self.assertTrue((result["trades"]["股數"] > 0).all())
            signal_dates = pd.to_datetime(result["trades"]["訊號日"])
            entry_dates = pd.to_datetime(result["trades"]["進場日"])
            self.assertTrue((signal_dates < entry_dates).all())

    def test_no_data_returns_empty_result(self):
        result = shared_capital_backtest({}, STRATEGIES[0])
        self.assertEqual(result["coverage"]["symbols_used"], 0)
        self.assertTrue(result["equity"].empty)

    def test_symbol_is_liquidated_on_its_own_last_available_date(self):
        short = sample_frame(25)
        long = sample_frame(28, phase=0.4)
        short["symbol"] = "SHORT"
        long["symbol"] = "LONG"

        def controlled_signals(frame):
            entry = pd.Series(False, index=frame.index)
            if frame["symbol"].iloc[0] == "SHORT":
                entry.iloc[20] = True
            exit_ = pd.Series(False, index=frame.index)
            return {STRATEGIES[0]: (entry, exit_)}

        with patch("research_engine._signals", side_effect=controlled_signals):
            result = shared_capital_backtest(
                {"SHORT": short, "LONG": long}, STRATEGIES[0],
                initial_capital=1_000_000, max_positions=1, warmup=0,
            )

        short_trades = result["trades"].loc[result["trades"]["代號"] == "SHORT"]
        self.assertEqual(len(short_trades), 1)
        self.assertEqual(
            pd.Timestamp(short_trades.iloc[0]["出場日"]),
            pd.Timestamp(short["date"].iloc[-1]),
        )
        self.assertLess(
            pd.Timestamp(short_trades.iloc[0]["出場日"]),
            pd.Timestamp(long["date"].iloc[-1]),
        )


if __name__ == "__main__":
    unittest.main()
