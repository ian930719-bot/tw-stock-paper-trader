import unittest
import numpy as np
import pandas as pd

from swing_turning_point_research import (
    FEATURES, apply_cooldown, build_feature_panel, future_extreme,
    simulate_portfolio,
)


class SwingTurningPointTests(unittest.TestCase):
    def test_future_extreme_excludes_current_bar(self):
        s = pd.Series([100.0, 110.0, 90.0, 120.0, 80.0])
        future = future_extreme(s, 2, "max")
        self.assertEqual(future.iloc[0], 110.0)
        self.assertEqual(future.iloc[1], 120.0)
        self.assertTrue(np.isnan(future.iloc[-1]))

    def test_cooldown_counts_each_symbol_trading_bars_not_panel_rows(self):
        dates = pd.bdate_range("2024-01-02", periods=100)
        rows = []
        for symbol in ("A", "B"):
            for i, date in enumerate(dates):
                rows.append({"symbol": symbol, "date": date,
                             "signal": i in (0, 20, 40, 80)})
        panel = pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)
        selected = apply_cooldown(panel, "signal", cooldown=40)
        got = panel.loc[selected].groupby("symbol").date.apply(
            lambda x: list((x - dates[0]).dt.days)
        )
        self.assertEqual(len(got["A"]), 3)
        self.assertEqual(len(got["B"]), 3)

    def test_same_close_features_do_not_change_when_future_bars_change(self):
        n = 360
        t = np.arange(n, dtype=float)
        close = 100*np.exp(.0004*t + .025*np.sin(t/9))
        frame = pd.DataFrame({
            "date": pd.bdate_range("2020-01-02", periods=n),
            "open": close*.999, "high": close*1.01,
            "low": close*.99, "close": close,
            "volume": 1_000_000+(t%13)*20_000,
        })
        original = build_feature_panel({"2330": frame})
        changed = frame.copy()
        changed.loc[270:, ["open","high","low","close","volume"]] *= 2
        revised = build_feature_panel({"2330": changed})
        pd.testing.assert_frame_equal(
            original.loc[:269, FEATURES+["atr","sma20","sma50","sma100","sma200"]],
            revised.loc[:269, FEATURES+["atr","sma20","sma50","sma100","sma200"]],
        )

    def test_signal_enters_next_available_open_and_period_end_closes(self):
        dates = pd.bdate_range("2024-01-02", periods=6)
        panel = pd.DataFrame({
            "date": dates, "symbol": "2330",
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104, 105],
            "atr": [1.0]*6, "adv20_proxy": [1e9]*6,
            "low_score": [.9, .1, .1, .1, .1, .1],
            "signal_low_cooldown": [True, False, False, False, False, False],
        })
        metrics, trades, equity = simulate_portfolio(
            panel, str(dates[0].date()), str(dates[-1].date()),
            hold_bars=50, trail_atr=5.0, risk_fraction=.08,
            max_positions=1, max_weight=.4, adv_participation=.01,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(pd.Timestamp(trades.iloc[0].entry_date), dates[1])
        self.assertEqual(pd.Timestamp(trades.iloc[0].exit_date), dates[-1])
        self.assertEqual(trades.iloc[0].exit_reason, "period_end_close")
        self.assertGreater(metrics["trades"], 0)
        self.assertEqual(int(equity.positions.max()), 1)

    def test_uncensored_forward_labels_are_unknown_not_negative(self):
        n = 80
        t = np.arange(n, dtype=float)
        close = 100 * np.exp(0.0005 * t + 0.02 * np.sin(t / 5))
        frame = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=n),
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 100_000.0),
        })
        panel = build_feature_panel({"2330": frame}).sort_values("date")
        tail = panel.tail(20)
        self.assertTrue(tail["future_max20"].isna().all())
        self.assertTrue(tail["future_min20"].isna().all())
        self.assertTrue(tail["low_label20"].isna().all())
        self.assertTrue(tail["high_label20"].isna().all())
        mature = panel.loc[panel["future_max20"].notna()]
        self.assertTrue(mature["low_label20"].notna().all())
        self.assertTrue(mature["high_label20"].notna().all())


if __name__ == "__main__":
    unittest.main()
