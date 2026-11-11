import unittest
import numpy as np
import pandas as pd

from top10_shared_capital_research import (
    prepare_bars, apply_universe_breadth, build_simulation_cache, simulate_portfolio, equal_weight_buy_hold,
    _trade_cost_return, BUY_FEE, SELL_FEE_TAX,
)


def toy_frame(n=6, symbol="2330", price=100.0, adv20=100_000.0):
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = np.full(n, price, dtype=float)
    return pd.DataFrame({
        "date": dates, "symbol": symbol, "name": symbol,
        "open": close.copy(), "high": close + 1,
        "low": close - 1, "close": close.copy(),
        "volume": np.full(n, 100_000.0),
        "atr": np.full(n, 1.0),
        "momentum20": np.zeros(n),
        "adv20": np.full(n, adv20),
        "sig_KD_low": [True] + [False] * (n - 1),
    })


class Top10SharedCapitalTests(unittest.TestCase):
    def test_signal_enters_next_available_open_not_signal_close(self):
        frame = toy_frame()
        _, trades, _ = simulate_portfolio(
            {"2330": frame}, "KD_low", "fixed_20",
            str(frame.date.iloc[0].date()), str(frame.date.iloc[-1].date()),
            max_positions=1,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(pd.Timestamp(trades.iloc[0].signal_date), pd.Timestamp(frame.date.iloc[0]))
        self.assertEqual(pd.Timestamp(trades.iloc[0].entry_date), pd.Timestamp(frame.date.iloc[1]))
        self.assertEqual(pd.Timestamp(trades.iloc[0].exit_date), pd.Timestamp(frame.date.iloc[-1]))
        self.assertEqual(trades.iloc[0].exit_reason, "period_end_close")

    def test_closed_position_cannot_reenter_same_symbol_on_same_open(self):
        frame = toy_frame(6)
        frame.loc[1, "sig_KD_low"] = True
        frame.loc[2, ["open", "high", "low", "close"]] = [90.0, 91.0, 89.0, 90.0]
        _, trades, _ = simulate_portfolio(
            {"2330": frame}, "KD_low", "fixed_20",
            str(frame.date.iloc[0].date()), str(frame.date.iloc[-1].date()), max_positions=1,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0].exit_reason, "gap_stop")

    def test_cached_and_uncached_simulations_are_identical(self):
        frame = toy_frame(8)
        frames = {"2330": frame}
        uncached = simulate_portfolio(
            frames, "KD_low", "fixed_20", "2024-01-02", "2024-01-11", max_positions=1,
        )
        cache = build_simulation_cache(frames)
        cached = simulate_portfolio(
            frames, "KD_low", "fixed_20", "2024-01-02", "2024-01-11", max_positions=1, cache=cache,
        )
        self.assertEqual(uncached[0], cached[0])
        pd.testing.assert_frame_equal(uncached[1], cached[1])
        pd.testing.assert_frame_equal(uncached[2], cached[2])

    def test_integer_shares_and_adv_participation_cap(self):
        frame = toy_frame(6, adv20=100_000.0)
        _, trades, _ = simulate_portfolio(
            {"2330": frame}, "KD_low", "fixed_20",
            str(frame.date.iloc[0].date()), str(frame.date.iloc[-1].date()),
            max_positions=1, max_participation=0.01,
        )
        self.assertEqual(len(trades), 1)
        shares = int(trades.iloc[0].shares)
        self.assertGreater(shares, 0)
        self.assertLessEqual(shares, int(100_000 * 0.01 / 100))

    def test_maximum_concurrent_positions_is_respected(self):
        frames = {str(i): toy_frame(8, symbol=str(i), adv20=1_000_000) for i in range(7)}
        _, trades, equity = simulate_portfolio(
            frames, "KD_low", "fixed_20",
            "2024-01-02", "2024-01-11", max_positions=3,
        )
        self.assertLessEqual(int(equity.positions.max()), 3)
        self.assertLessEqual(trades.symbol.nunique(), 3)

    def test_transaction_cost_formula_applies_each_side_once(self):
        result = _trade_cost_return(100.0, 110.0)
        expected = (110 / 100) * (1 - BUY_FEE) * (1 - SELL_FEE_TAX) - 1
        self.assertAlmostEqual(result, expected, places=12)

    def test_indicators_are_causal_when_future_bars_change(self):
        n = 350
        t = np.arange(n, dtype=float)
        close = 100 * np.exp(.0005 * t + .03 * np.sin(t / 8))
        frame = pd.DataFrame({
            "date": pd.bdate_range("2020-01-02", periods=n),
            "open": close * .999, "high": close * 1.01,
            "low": close * .99, "close": close,
            "volume": 1_000_000 + (t % 11) * 10_000,
        })
        original = prepare_bars(frame)
        changed = frame.copy()
        changed.loc[260:, ["open", "high", "low", "close", "volume"]] *= 2
        revised = prepare_bars(changed)
        cols = ["rsi", "k", "d", "atr", "bb_pct", "momentum20", "adv20",
                "sig_KD_low", "sig_Oversold3", "sig_Trend_RSI",
                "sig_TrendBreakout", "sig_SqueezeBreakout", "sig_TrendPullback"]
        pd.testing.assert_frame_equal(original.loc[:259, cols], revised.loc[:259, cols])

    def test_breadth_filter_uses_same_day_completed_closes_and_thresholds(self):
        dates = pd.bdate_range("2020-01-02", periods=3)
        frames = {}
        for i in range(10):
            above = i < 5
            close = 110.0 if above else 90.0
            d = pd.DataFrame({
                "date": dates, "close": [close] * 3, "sma200": [100.0] * 3,
                "sig_KD_low": [True] * 3, "sig_Oversold3": [True] * 3,
                "sig_Trend_RSI": [True] * 3, "sig_TrendBreakout": [True] * 3,
                "sig_SqueezeBreakout": [True] * 3,
            })
            frames[str(i)] = d
        apply_universe_breadth(frames, min_symbols=8)
        self.assertTrue((frames["0"].universe_breadth == 0.5).all())
        self.assertTrue(frames["0"].sig_KD_low_Breadth50.all())
        self.assertFalse(frames["0"].sig_KD_low_Breadth70.any())
        self.assertTrue(frames["0"].sig_TrendBreakout_Breadth50.all())
        self.assertFalse(frames["0"].sig_TrendBreakout_Breadth70.any())

    def test_equal_weight_benchmark_charges_round_trip_costs(self):
        a = toy_frame(5, symbol="A")
        b = toy_frame(5, symbol="B")
        metrics, curve = equal_weight_buy_hold({"A": a, "B": b}, "2024-01-02", "2024-01-08", 100_000)
        self.assertLess(metrics["total_return"], 0)
        self.assertEqual(metrics["stocks"], 2)
        self.assertEqual(float(curve.equity.iloc[-1]), metrics["ending_equity"])


if __name__ == "__main__":
    unittest.main()
