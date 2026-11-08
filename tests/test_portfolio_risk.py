import unittest
import numpy as np
import pandas as pd
from portfolio_risk import portfolio_risk_report, risk_parity_weights, dynamic_strategy_ensemble


def fixture(n=320):
    dates = pd.bdate_range("2023-01-02", periods=n)
    t = np.arange(n, dtype=float)
    out = {}
    for i, code in enumerate(("A", "B", "C", "D")):
        r = 0.0002 + 0.008 * np.sin(t / (7 + i)) + 0.003 * np.cos(t / (13 + i))
        close = 50 * np.cumprod(1 + r)
        out[code] = pd.DataFrame({"date": dates, "close": close})
    return out


class PortfolioRiskTests(unittest.TestCase):
    def test_weights_respect_exposure_and_name_caps(self):
        r = portfolio_risk_report(fixture())
        self.assertEqual(r["status"], "ok")
        self.assertAlmostEqual(float(r["weights"].sum()), 0.90, places=6)
        self.assertLessEqual(float(r["weights"].max()), 0.30 + 1e-8)
        self.assertGreaterEqual(r["cash_weight"], 0)

    def test_risk_contributions_are_percentages(self):
        r = portfolio_risk_report(fixture())
        self.assertTrue(np.isfinite(r["risk_contributions"]).all())
        self.assertAlmostEqual(float(r["risk_contributions"].sum()), 1.0, places=5)

    def test_correlation_is_symmetric(self):
        corr = portfolio_risk_report(fixture())["correlation"].to_numpy()
        self.assertTrue(np.allclose(corr, corr.T))
        self.assertTrue(np.allclose(np.diag(corr), 1))

    def test_infeasible_caps_are_rejected(self):
        returns = pd.DataFrame(np.random.default_rng(2).normal(0, .01, (80, 2)))
        with self.assertRaisesRegex(ValueError, "無法在單檔上限"):
            risk_parity_weights(returns, max_weight=.30, target_exposure=.90)

    def test_short_history_fails_closed(self):
        self.assertEqual(portfolio_risk_report(fixture(50))["status"], "insufficient_history")

    def test_dynamic_ensemble_uses_only_prior_returns(self):
        dates = pd.bdate_range("2024-01-02", periods=150)
        curves = {}
        for i, name in enumerate(("trend", "breakout", "reversion")):
            r = 0.0003 + (0.004 + i * 0.002) * np.sin(np.arange(150) / (6 + i))
            curves[name] = pd.DataFrame({
                "日期": dates,
                "資產": 100000 * np.cumprod(1 + r),
            })
        first = dynamic_strategy_ensemble(curves, lookback=40, rebalance_frequency=10)
        changed = {k: v.copy() for k, v in curves.items()}
        for v in changed.values():
            v.loc[100:, "資產"] *= 4
        second = dynamic_strategy_ensemble(changed, lookback=40, rebalance_frequency=10)
        self.assertEqual(first["status"], "ok")
        self.assertTrue(np.allclose(
            first["weights"].loc[:dates[99]].to_numpy(),
            second["weights"].loc[:dates[99]].to_numpy(),
        ))
        self.assertLessEqual(float(first["weights"].to_numpy().max()), 0.40 + 1e-8)
        self.assertTrue((first["weights"].sum(axis=1) <= 0.90 + 1e-8).all())

    def test_dynamic_ensemble_requires_enough_strategy_history(self):
        dates = pd.bdate_range("2024-01-02", periods=45)
        curve = pd.DataFrame({"日期": dates, "資產": np.linspace(100000, 105000, 45)})
        result = dynamic_strategy_ensemble({"a": curve, "b": curve}, lookback=60)
        self.assertEqual(result["status"], "insufficient_strategies")

    def test_snapshot_is_asof_and_does_not_use_later_rows(self):
        frames = fixture(340)
        cutoff = frames["A"]["date"].iloc[319]
        earlier = {k: v.loc[v.date <= cutoff].copy() for k, v in frames.items()}
        a = portfolio_risk_report(earlier)
        for v in frames.values():
            v.loc[v.date > cutoff, "close"] *= 10
        b = portfolio_risk_report(earlier)
        self.assertTrue(np.allclose(a["weights"].sort_index(), b["weights"].sort_index()))


if __name__ == "__main__":
    unittest.main()
