import unittest
import numpy as np
import pandas as pd
from sector_rotation_backtest import build_monthly_targets, metrics, run_portfolio

class SectorRotationBacktestTests(unittest.TestCase):
    def test_sector_selection_is_cross_sectional_and_bounded(self):
        rows = []
        day = pd.Timestamp("2024-01-31")
        for industry, base in [("A", 0.30), ("B", 0.10)]:
            for i in range(6):
                rows.append({
                    "date": day, "symbol": f"{industry}{i}", "industry": industry,
                    "close": 100.0, "ret5": -0.02 + i * 0.001,
                    "ret20": 0.01, "ret60": base + i * 0.001,
                    "ret120": base, "distance_high55": -0.03,
                    "atr_pct14": 0.02 + i * 0.001,
                    "avg_turnover20": 50_000_000.0, "eligible": True,
                })
        targets, sectors, picks = build_monthly_targets(pd.DataFrame(rows),
            top_sectors=1, stocks_per_sector=2, min_sector_names=5)
        self.assertEqual(len(targets[day]), 2)
        self.assertTrue(all(x.startswith("A") for x in targets[day]))
        self.assertEqual(int(sectors.loc[sectors.selected, "selected"].sum()), 1)
        self.assertEqual(len(picks), 2)

    def test_portfolio_fills_next_open_and_exits_on_next_open(self):
        dates = np.array(["2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[ns]")
        data = {"A": {
            "industry": "測試業", "dates": dates,
            "open": np.array([10, 10, 11], dtype=np.float32),
            "close": np.array([10, 10, 11], dtype=np.float32),
            "sma50": np.array([9, 9, 9], dtype=np.float32),
            "high20": np.array([10, 10, 11], dtype=np.float32),
            "avg_volume20": np.array([100, 1_000_000, 1_000_000], dtype=np.float32),
            "ret5": np.zeros(3, dtype=np.float32), "ret20": np.zeros(3, dtype=np.float32),
            "ret60": np.zeros(3, dtype=np.float32), "ret120": np.zeros(3, dtype=np.float32),
        }}
        targets = {pd.Timestamp(dates[0]): {"A"}, pd.Timestamp(dates[1]): set()}
        eq, trades, costs = run_portfolio(data, pd.DatetimeIndex(dates), targets,
            initial_capital=100_000, max_positions=1)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0].entry_date, "2024-01-03")
        self.assertEqual(trades.iloc[0].exit_date, "2024-01-04")
        self.assertEqual(trades.iloc[0].shares, 1)  # Uses prior-session ADV, not same-day volume
        self.assertGreater(costs["fees"], 0)
        self.assertEqual(len(eq), 3)

    def test_period_metrics_use_prior_equity_and_period_trades(self):
        eq = pd.DataFrame({
            "date": pd.to_datetime(["2021-12-30", "2022-01-03", "2022-01-04"]),
            "equity": [100.0, 105.0, 110.0], "exposure": [0.0, 0.5, 0.5],
        })
        trades = pd.DataFrame({
            "exit_date": ["2021-12-30", "2022-01-04"],
            "pnl": [5.0, 5.0], "return": [0.05, 0.05],
        })
        result = metrics(eq, trades, 100.0, "2022-01-01")
        self.assertAlmostEqual(result["total_return"], 0.10)
        self.assertEqual(result["closed_trades"], 1)

if __name__ == "__main__":
    unittest.main()
