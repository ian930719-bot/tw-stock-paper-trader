import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_engine import STRATEGIES, shared_capital_backtest
from continuous_walk_forward import continuous_walk_forward


def flat_frame(n=450):
    dates = pd.bdate_range("2021-01-04", periods=n)
    close = np.full(n, 100.0)
    return pd.DataFrame({
        "date": dates, "open": close, "high": close + 0.05,
        "low": close - 0.05, "close": close,
        "volume": np.full(n, 1_000_000), "atr": np.full(n, 0.1),
    })


class ContinuousWalkForwardTests(unittest.TestCase):
    def test_fixed_strategy_uses_one_continuous_portfolio_run(self):
        frame = flat_frame()

        def controlled_signals(z):
            entry = pd.Series(False, index=z.index)
            entry.iloc[299] = True
            exit_signal = pd.Series(False, index=z.index)
            return {STRATEGIES[0]: (entry, exit_signal)}

        with patch("research_engine._signals", side_effect=controlled_signals), \
             patch("continuous_walk_forward.shared_capital_backtest",
                   wraps=shared_capital_backtest) as simulator:
            result = continuous_walk_forward(
                {"TEST": frame}, STRATEGIES[0],
                initial_capital=100_000, max_positions=1,
                lookback_days=300, test_days=60, warmup_bars=200,
                min_test_days=40, risk_fraction=0.005,
                fee=0, sell_tax=0, slippage=0,
            )

        self.assertEqual(simulator.call_count, 1)
        self.assertEqual(len(result["windows"]), 2)
        self.assertEqual(
            len(result["equity"]),
            int(result["windows"]["交易日數"].sum()),
        )
        self.assertEqual(
            pd.Timestamp(result["equity"]["日期"].iloc[-1]),
            pd.Timestamp(result["windows"]["測試結束"].iloc[-1]),
        )
        self.assertGreaterEqual(int(result["equity"]["持倉數"].max()), 1)
        # The first position survives the first OOS boundary; the account is
        # not liquidated and restarted when the next test window begins.
        boundary = pd.Timestamp(result["windows"].iloc[0]["測試結束"])
        next_day = result["equity"].loc[
            pd.to_datetime(result["equity"]["日期"]) > boundary
        ].iloc[0]
        self.assertEqual(int(next_day["持倉數"]), 1)


if __name__ == "__main__":
    unittest.main()
