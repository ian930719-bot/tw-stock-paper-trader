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
    def test_rejects_invalid_risk_budget(self):
        with self.assertRaises(ValueError):
            shared_capital_backtest({}, STRATEGIES[0], risk_fraction=0.5)
        with self.assertRaises(ValueError):
            shared_capital_backtest({}, STRATEGIES[0], initial_capital=0)

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

    def test_first_report_day_includes_initial_capital_in_drawdown(self):
        frame = sample_frame(25)
        frame["symbol"] = "ENTRY"
        frame["open"] = frame["close"] * 1.01
        frame["high"] = frame["open"] * 1.002
        frame["low"] = frame["close"] * 0.998

        def controlled_signals(z):
            entry = pd.Series(False, index=z.index)
            entry.iloc[0] = True
            return {STRATEGIES[0]: (entry, pd.Series(False, index=z.index))}

        with patch("research_engine._signals", side_effect=controlled_signals):
            result = shared_capital_backtest(
                {"ENTRY": frame}, STRATEGIES[0],
                initial_capital=1_000_000, max_positions=1, warmup=0,
                report_start_date=frame["date"].iloc[1],
            )

        first_equity = float(result["equity"]["資產"].iloc[0])
        self.assertLess(first_equity, 1_000_000)
        self.assertLessEqual(
            result["metrics"]["最大回撤"],
            first_equity / 1_000_000 - 1 + 1e-9,
        )

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


    def test_exit_profiles_share_entry_logic_but_change_initial_stop(self):
        frame = sample_frame(30)
        frame["symbol"] = "PROFILE"
        frame.loc[21, "low"] = float(frame.loc[21, "open"]) - 1.0

        def controlled_signals(z):
            entry = pd.Series(False, index=z.index)
            entry.iloc[20] = True
            return {STRATEGIES[0]: (entry, pd.Series(False, index=z.index))}

        with patch("research_engine._signals", side_effect=controlled_signals):
            atr_result = shared_capital_backtest(
                {"PROFILE": frame}, STRATEGIES[0], warmup=0,
                max_positions=1, exit_profile="atr_2",
            )
            oneil_result = shared_capital_backtest(
                {"PROFILE": frame}, STRATEGIES[0], warmup=0,
                max_positions=1, exit_profile="oneil_8pct",
            )

        self.assertEqual(atr_result["exit_profile"], "atr_2")
        self.assertEqual(oneil_result["exit_profile"], "oneil_8pct")
        self.assertEqual(len(atr_result["trades"]), 1)
        self.assertEqual(len(oneil_result["trades"]), 1)
        self.assertEqual(atr_result["trades"]["出場原因"].iloc[0], "進場日觸及初始停損")
        self.assertIn("該標的資料期末平倉", oneil_result["trades"]["出場原因"].iloc[0])

    def test_minervini_structural_stop_uses_only_completed_bars_before_signal(self):
        frame = sample_frame(26)
        frame["open"] = 100.0
        frame["high"] = 100.5
        frame["low"] = 99.5
        frame["close"] = 100.0
        frame["atr"] = 2.0
        # The signal bar's low is deliberately extreme; it must not become
        # the stop reference because the signal is only known at that close.
        frame.loc[20, "low"] = 70.0
        frame.loc[10:19, "low"] = 98.0

        def controlled_signals(z):
            entry = pd.Series(False, index=z.index)
            entry.iloc[20] = True
            return {STRATEGIES[0]: (entry, pd.Series(False, index=z.index))}

        with patch("research_engine._signals", side_effect=controlled_signals):
            result = shared_capital_backtest(
                {"MINERVINI_LOOKAHEAD": frame}, STRATEGIES[0], warmup=0,
                max_positions=1, risk_fraction=0.02,
                exit_profile="minervini_structure",
            )
        # The 98 support from prior completed bars is used; the signal-day 70
        # print is excluded. With 2% account risk, shares are sized on 2/share.
        trades = result["trades"]
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(float(trades.iloc[0]["進場成本"]) / float(trades.iloc[0]["股數"]), 100.0 * 1.001 * 1.001425, places=3)

    def test_entry_day_high_updates_atr_trailing_stop_for_next_session(self):
        frame = sample_frame(26)
        frame["open"] = 100.0
        frame["high"] = 100.5
        frame["low"] = 99.5
        frame["close"] = 100.0
        frame["atr"] = 2.0
        # Entry fills at row 21. Its high raises the ATR(2) trail to 106.
        frame.loc[21, ["high", "close"]] = [110.0, 108.0]
        # The next session opens above the stop, then trades below it.
        frame.loc[22, ["open", "high", "low", "close"]] = [108.0, 109.0, 105.0, 107.0]

        def controlled_signals(z):
            entry = pd.Series(False, index=z.index)
            entry.iloc[20] = True
            return {STRATEGIES[0]: (entry, pd.Series(False, index=z.index))}

        with patch("research_engine._signals", side_effect=controlled_signals):
            result = shared_capital_backtest(
                {"ENTRY_TRAIL": frame}, STRATEGIES[0], warmup=0,
                max_positions=1, risk_fraction=0.02, exit_profile="atr_2",
                fee=0.0, sell_tax=0.0, slippage=0.0,
            )
        trades = result["trades"]
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["出場日"], str(pd.Timestamp(frame.loc[22, "date"]).date()))
        self.assertEqual(trades.iloc[0]["出場原因"], "停損／移動停損")
        self.assertAlmostEqual(float(trades.iloc[0]["出場收入"]) / float(trades.iloc[0]["股數"]), 106.0)

    def test_unknown_exit_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知出場設定"):
            shared_capital_backtest({}, STRATEGIES[0], exit_profile="magic")


    def test_oneil_profit_review_activates_wide_trailing_stop_not_fixed_target(self):
        frame = sample_frame(26)
        frame["open"] = 100.0
        frame["high"] = 100.5
        frame["low"] = 99.5
        frame["close"] = 100.0
        frame["atr"] = 2.0
        frame.loc[22, ["high", "close"]] = [125.0, 120.0]
        frame.loc[23, ["high", "close", "low"]] = [130.0, 126.0, 124.0]

        def controlled_signals(z):
            entry = pd.Series(False, index=z.index)
            entry.iloc[20] = True
            return {STRATEGIES[0]: (entry, pd.Series(False, index=z.index))}

        with patch("research_engine._signals", side_effect=controlled_signals):
            result = shared_capital_backtest(
                {"ONEIL": frame}, STRATEGIES[0], warmup=0,
                max_positions=1, risk_fraction=0.02, exit_profile="oneil_8pct",
            )
        trades = result["trades"]
        self.assertEqual(len(trades), 1)
        self.assertTrue(bool(trades.iloc[0]["曾觸發獲利檢視"]))
        self.assertEqual(trades.iloc[0]["出場原因"], "停損／移動停損")
        self.assertGreater(float(trades.iloc[0]["最高風險倍數"]), 0)

    def test_minervini_two_r_moves_stop_to_breakeven_and_three_r_trails(self):
        frame = sample_frame(26)
        frame["open"] = 100.0
        frame["high"] = 100.5
        frame["low"] = 99.5
        frame["close"] = 100.0
        frame["atr"] = 2.0
        # Structural stop at the row-20 signal uses rows 10..19 only.
        frame.loc[10:19, "low"] = 98.0
        frame.loc[22, ["high", "close"]] = [105.0, 104.0]
        frame.loc[23, ["high", "close", "low"]] = [108.0, 106.0, 103.0]
        frame.loc[24, "low"] = 102.0

        def controlled_signals(z):
            entry = pd.Series(False, index=z.index)
            entry.iloc[20] = True
            return {STRATEGIES[0]: (entry, pd.Series(False, index=z.index))}

        with patch("research_engine._signals", side_effect=controlled_signals):
            result = shared_capital_backtest(
                {"MINERVINI": frame}, STRATEGIES[0], warmup=0,
                max_positions=1, risk_fraction=0.02,
                exit_profile="minervini_structure",
            )
        trades = result["trades"]
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["出場原因"], "停損／移動停損")
        # Row 22 closes at +2R; row 23 reaches +3R and activates the ATR trail.
        self.assertGreaterEqual(float(trades.iloc[0]["最高風險倍數"]), 3.0)


if __name__ == "__main__":
    unittest.main()
