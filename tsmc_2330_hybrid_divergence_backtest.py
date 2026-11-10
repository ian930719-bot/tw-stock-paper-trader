"""Hybrid TSMC 2330 strategy: reversal-confirmed lows + MACD-divergence highs.

Both direction rules are selected independently using the same 2017-2021
validation period: low entry uses the selected prior-range break confirmation;
high exit uses the selected base-high plus bearish-MACD divergence. This script
does not optimize the 2022+ period; it evaluates the frozen hybrid with the
existing execution/cost/limit/dividend simulator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_execution_backtest import (
    SLIPPAGE_BPS, INITIAL_CAPITAL, load_inputs, _simulate_period,
)
from tsmc_2330_reversal_confirmation_backtest import build_confirmed_signals
from tsmc_2330_divergence_research import candidate_signals

VALIDATION = ("2017-01-01", "2021-12-31")
HOLDOUT_START = "2022-01-01"


def compact(result: dict) -> dict:
    return {k: result[k] for k in (
        "period_start", "period_end", "initial_capital", "final_equity",
        "total_return", "cagr", "max_drawdown", "trade_count", "win_rate",
        "average_trade_return", "exposure_ratio", "total_cash_dividends",
        "total_commission", "total_sell_tax", "estimated_slippage_cost",
        "blocked_limit_orders", "missing_price_limit_orders",
    )}


def run(features_path: str, rules_path: str, limits_path: str, dividends_path: str,
        output_dir: str, initial_capital: float = INITIAL_CAPITAL) -> dict:
    bars, rules, limits, dividends = load_inputs(
        features_path, rules_path, limits_path, dividends_path
    )
    low_signal, _ = build_confirmed_signals(bars, rules, "break_prior_range")
    base_low, base_high = __import__(
        "tsmc_2330_execution_backtest", fromlist=["build_signals"]
    ).build_signals(bars, rules)
    candidates = candidate_signals(bars, base_low, base_high)
    high_signal = candidates["HIGH"]["base_high_and_bearish_macd"]
    limit_lookup = {
        row.date.strftime("%Y-%m-%d"): (float(row.limit_up), float(row.limit_down))
        for row in limits.itertuples(index=False)
        if np.isfinite(row.limit_up) and np.isfinite(row.limit_down)
    }

    validation = {}
    for slip in SLIPPAGE_BPS:
        result = _simulate_period(
            bars, low_signal, high_signal, limit_lookup, dividends,
            *VALIDATION, slip, initial_capital, True,
        )
        validation[f"hybrid_{slip}bps"] = result

    holdout = {}
    for slip in SLIPPAGE_BPS:
        result = _simulate_period(
            bars, low_signal, high_signal, limit_lookup, dividends,
            HOLDOUT_START, bars["date"].max().date().isoformat(),
            slip, initial_capital, True,
        )
        holdout[f"hybrid_{slip}bps"] = result

    baseline_low, baseline_high = build_confirmed_signals(bars, rules, "break_prior_range")
    baseline = _simulate_period(
        bars, baseline_low, baseline_high, limit_lookup, dividends,
        HOLDOUT_START, bars["date"].max().date().isoformat(),
        10, initial_capital, True,
    )
    buy_hold = _simulate_period(
        bars, low_signal, high_signal, limit_lookup, dividends,
        HOLDOUT_START, bars["date"].max().date().isoformat(),
        10, initial_capital, False,
    )
    summary = {
        "symbol": "2330",
        "entry_rule": "break_prior_range low setup + close above prior session high + close_location >= 0.55; selected in 2017-2021",
        "exit_rule": "base high setup + bearish MACD histogram divergence20; selected in 2017-2021",
        "validation_period": list(VALIDATION),
        "validation_results": {k: compact(v) for k, v in validation.items()},
        "historical_holdout_period": [HOLDOUT_START, bars["date"].max().date().isoformat()],
        "historical_holdout_results": {k: compact(v) for k, v in holdout.items()},
        "historical_holdout_prior_range_baseline_10bps": compact(baseline),
        "historical_holdout_buy_hold_10bps": compact(buy_hold),
        "test_period_used_for_selection": False,
        "holdout_is_pristine": False,
        "warning": "Both rules were chosen on 2017-2021 only. 2022+ has been inspected in previous diagnostics and is not a pristine blind test. No 2022+ parameter tuning was performed.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_hybrid_divergence_backtest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for key, value in {**validation, **holdout}.items():
        pd.DataFrame(value["trades"]).to_csv(out / f"{key}_trades.csv", index=False)
        pd.DataFrame(value["equity_curve"]).to_csv(out / f"{key}_equity.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--limits", required=True)
    parser.add_argument("--dividends", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.rules, args.limits, args.dividends,
                         args.output, args.initial_capital), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
