"""Tests for historical data audit and factor research."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from historical_research import (
    audit_historical_panel, build_factor_panel, evaluate_factor_panel, run_research,
)


def sample_panel(n=360, symbols=("2330", "2603", "6488")):
    rows = []
    dates = pd.bdate_range("2023-01-02", periods=n)
    for j, symbol in enumerate(symbols):
        t = np.arange(n, dtype=float)
        close = (50 + j * 20) * np.exp((0.0005 + j * 0.0002) * t + 0.015 * np.sin(t / (6+j)))
        for i, day in enumerate(dates):
            rows.append({
                "symbol": symbol, "date": day, "open": close[i] * 0.999,
                "high": close[i] * 1.01, "low": close[i] * 0.99,
                "close": close[i], "volume": 100000 + (i % 13) * 5000,
            })
    return pd.DataFrame(rows)


class HistoricalResearchTests(unittest.TestCase):
    def test_audit_blocks_research_grade_without_pit_and_delisted_coverage(self):
        panel = sample_panel()
        coverage, summary = audit_historical_panel(
            panel, source="fixture", price_basis="raw",
            point_in_time_universe=False, delisted_coverage=False,
        )
        self.assertEqual(len(coverage), 3)
        self.assertFalse(summary["research_grade_eligible"])
        self.assertTrue(any("point-in-time" in x for x in summary["limitations"]))
        self.assertTrue(any("下市" in x for x in summary["limitations"]))

    def test_audit_detects_duplicate_and_invalid_ohlc(self):
        panel = sample_panel(n=30, symbols=("2330",))
        panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
        panel.loc[1, "high"] = panel.loc[1, "low"] - 1
        _, summary = audit_historical_panel(panel, price_basis="raw")
        self.assertGreater(summary["duplicate_rows"], 0)
        self.assertEqual(summary["invalid_rows"], 1)

    def test_factor_panel_uses_next_open_forward_return(self):
        panel = sample_panel(n=260, symbols=("2330",))
        result = build_factor_panel(panel)
        row = result.iloc[100]
        expected = (
            panel.sort_values("date").iloc[101]["close"]
            / panel.sort_values("date").iloc[101]["open"] - 1
        )
        self.assertAlmostEqual(row["fwd_1d_next_open"], expected, places=10)
        self.assertIn("breakout_55", result.columns)
        self.assertIn("volatility_compression", result.columns)

    def test_factor_panel_is_causal(self):
        original = sample_panel(n=260, symbols=("2330",))
        changed = original.copy()
        changed.loc[200:, ["open", "high", "low", "close", "volume"]] *= 2
        a = build_factor_panel(original)
        b = build_factor_panel(changed)
        cols = ["ret_20", "distance_sma_50", "breakout_20", "relative_volume_20"]
        pd.testing.assert_frame_equal(
            a.loc[:199, cols].reset_index(drop=True),
            b.loc[:199, cols].reset_index(drop=True),
        )

    def test_research_runner_reads_collector_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "download"
            bars = root / "bars"
            bars.mkdir(parents=True)
            sample_panel(n=260, symbols=tuple(f"{i:04d}" for i in range(10))).to_csv(
                bars / "fixture.csv", index=False
            )
            output = Path(tmp) / "artifacts"
            summary = run_research(str(root), str(output), price_basis="raw")
            self.assertEqual(summary["symbols"], 10)
            self.assertTrue((output / "data_audit_summary.json").exists())
            self.assertTrue((output / "factor_research_results.csv").exists())

    def test_factor_evaluation_returns_registry_without_auto_selecting(self):
        panel = build_factor_panel(sample_panel(n=300, symbols=tuple(f"{i:04d}" for i in range(10))))
        results, registry = evaluate_factor_panel(panel)
        self.assertFalse(registry.empty)
        self.assertTrue(registry["selection_status"].eq("UNSELECTED").all())
        self.assertIn("mean_rank_ic", results.columns)


if __name__ == "__main__":
    unittest.main()
