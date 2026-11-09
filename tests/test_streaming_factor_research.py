import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from streaming_factor_research import (
    benjamini_hochberg, build_symbol_factors, hac_t_stat, run_streaming_research,
)


class StreamingFactorResearchTests(unittest.TestCase):
    def make_bars(self, symbol, n=280, seed=7):
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0004, 0.015, n)
        close = 50 * np.cumprod(1 + returns)
        open_ = np.r_[close[0], close[:-1] * (1 + rng.normal(0, 0.003, n-1))]
        high = np.maximum(open_, close) * 1.01
        low = np.minimum(open_, close) * 0.99
        return pd.DataFrame({
            "symbol": symbol, "date": pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d"),
            "open": open_, "high": high, "low": low, "close": close,
            "volume": rng.integers(1000, 100000, n), "turnover": close * rng.integers(1000, 100000, n),
        })

    def test_next_open_forward_return_and_warmup(self):
        bars = self.make_bars("2330", n=280)
        f = build_symbol_factors(bars, "2330")
        self.assertTrue(pd.isna(f.iloc[0]["ret_5"]))
        expected = bars.iloc[5]["close"] / bars.iloc[1]["open"] - 1
        self.assertAlmostEqual(f.iloc[0]["fwd_5d_next_open"], expected, places=10)

    def test_hac_stat_and_bh_adjustment(self):
        self.assertTrue(np.isfinite(hac_t_stat([0.1, 0.2, 0.15, 0.12, 0.18, 0.11], 2)))
        q = benjamini_hochberg([0.001, 0.02, 0.5])
        self.assertLess(q[0], q[1])
        self.assertLessEqual(q[1], q[2])

    def test_end_to_end_writes_quality_and_exploratory_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            (root / "bars").mkdir(parents=True)
            for i in range(12):
                self.make_bars(str(1000+i), n=280, seed=10+i).to_csv(root / "bars" / f"{1000+i}.csv", index=False)
            import json
            pd.DataFrame({"symbol": [str(1000+i) for i in range(12)] + ["9999"], "name": ["test"] * 13}).to_csv(root / "candidate_symbols.csv", index=False)
            (root / "download_manifest.json").write_text(json.dumps({
                "source": "test", "price_dataset": "raw", "start_date": "2020-01-01",
                "collection_complete": True, "data_files_sha256": {},
                "point_in_time_universe_complete": False, "delisted_price_coverage_verified": False,
            }), encoding="utf-8")
            out = Path(tmp) / "out"
            summary = run_streaming_research(str(root), str(out), split_date="2021-01-01")
            self.assertEqual(summary["symbols_with_valid_rows"], 12)
            self.assertEqual(summary["missing_candidate_files"], 1)
            self.assertTrue((out / "candidate_coverage.csv").exists())
            self.assertFalse(summary["research_grade_eligible"])
            self.assertTrue((out / "data_quality.csv").exists())
            self.assertTrue((out / "factor_research_results.csv").exists())
            self.assertIn("POST_SPLIT_DESCRIPTIVE", set(pd.read_csv(out / "factor_research_results.csv")["segment"]))


    def test_factor_builder_uses_adjusted_prices_for_signals_and_forward_returns(self):
        bars = self.make_bars("2330", n=280, seed=44)
        bars["adj_open"] = bars["open"] * 0.5
        bars["adj_high"] = bars["high"] * 0.5
        bars["adj_low"] = bars["low"] * 0.5
        bars["adj_close"] = bars["close"] * 0.5
        f = build_symbol_factors(bars, "2330")
        expected = bars.iloc[5]["adj_close"] / bars.iloc[1]["adj_open"] - 1
        raw_expected = bars.iloc[5]["close"] / bars.iloc[1]["open"] - 1
        self.assertAlmostEqual(f.iloc[0]["fwd_5d_next_open"], expected, places=10)
        self.assertNotAlmostEqual(f.iloc[0]["fwd_5d_next_open"], raw_expected, places=8)


if __name__ == "__main__":
    unittest.main()
