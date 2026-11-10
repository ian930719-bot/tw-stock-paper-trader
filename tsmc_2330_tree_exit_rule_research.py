"""Compare predeclared high-point exits with the frozen TSMC tree low entry.

The entry rule is fixed from the prior tree study. Exit variants are predeclared:
base high setup, bearish MACD divergence, and five stateful confirmations.
Only 2017-2021 selects the exit using CAGR - 0.5*abs(max drawdown).
2022+ is reported as a descriptive historical check, not a pristine blind test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_execution_backtest import (
    SLIPPAGE_BPS, INITIAL_CAPITAL, load_inputs, _simulate_period, build_signals,
)
from tsmc_2330_tree_rule_execution_backtest import tree_low_setup
from tsmc_2330_divergence_research import candidate_signals
from tsmc_2330_reversal_confirmation_backtest import build_confirmed_signals, VARIANTS

VALIDATION = ("2017-01-01", "2021-12-31")
HOLDOUT_START = "2022-01-01"
SELECTION_SLIPPAGE_BPS = 10


def selection_score(result: dict) -> float:
    return float(result["cagr"]) - 0.5 * abs(float(result["max_drawdown"]))


def compact(result: dict) -> dict:
    keys = (
        "period_start", "period_end", "initial_capital", "final_equity",
        "total_return", "cagr", "max_drawdown", "trade_count", "win_rate",
        "average_trade_return", "exposure_ratio", "total_cash_dividends",
        "total_commission", "total_sell_tax", "estimated_slippage_cost",
        "blocked_limit_orders", "missing_price_limit_orders",
    )
    return {k: result[k] for k in keys}


def run(features_path: str, rules_path: str, limits_path: str,
        dividends_path: str, output_dir: str,
        initial_capital: float = INITIAL_CAPITAL) -> dict:
    bars, rules, limits, dividends = load_inputs(
        features_path, rules_path, limits_path, dividends_path
    )
    low_signal = tree_low_setup(bars)
    base_low, base_high = build_signals(bars, rules)
    divergence = candidate_signals(bars, base_low, base_high)
    exits = {
        "base_high_setup": base_high,
        "base_high_bearish_macd": divergence["HIGH"]["base_high_and_bearish_macd"],
    }
    for variant in VARIANTS:
        if variant == "unconfirmed_baseline":
            continue
        _, high_confirmed = build_confirmed_signals(bars, rules, variant)
        exits[f"confirmed_{variant}"] = high_confirmed

    limit_lookup = {
        row.date.strftime("%Y-%m-%d"): (float(row.limit_up), float(row.limit_down))
        for row in limits.itertuples(index=False)
        if np.isfinite(row.limit_up) and np.isfinite(row.limit_down)
    }
    results = {}
    for exit_name, high_signal in exits.items():
        for period_name, (start, end) in {
            "validation_2017_2021": VALIDATION,
            "historical_check_2022_plus": (
                HOLDOUT_START, bars["date"].max().date().isoformat()
            ),
        }.items():
            for slip in SLIPPAGE_BPS:
                result = _simulate_period(
                    bars, low_signal, high_signal, limit_lookup, dividends,
                    start, end, slip, initial_capital, True,
                )
                results[f"{period_name}_{exit_name}_{slip}bps"] = result

    validation_10 = {
        name: value for name, value in results.items()
        if name.startswith("validation_2017_2021_") and name.endswith("_10bps")
    }
    selected_key = max(validation_10, key=lambda k: (
        selection_score(validation_10[k]),
        validation_10[k]["cagr"],
    ))
    selected_exit = selected_key.removeprefix("validation_2017_2021_").removesuffix("_10bps")
    selected_holdout = results[f"historical_check_2022_plus_{selected_exit}_10bps"]
    buy_hold = _simulate_period(
        bars, low_signal, base_high, limit_lookup, dividends,
        HOLDOUT_START, bars["date"].max().date().isoformat(),
        10, initial_capital, False,
    )
    summary = {
        "symbol": "2330",
        "frozen_entry_rule": "return_1 <= -0.0025693492 AND close_location <= 0.25657895; next-session open entry",
        "exit_candidates": list(exits),
        "confirmation_max_wait_sessions": 5,
        "selection_metric": "CAGR - 0.5 * abs(max_drawdown), at 10 bps one-way slippage",
        "selection_period": list(VALIDATION),
        "selected_exit": selected_exit,
        "selected_validation_result": compact(validation_10[selected_key]),
        "selected_historical_check_result": compact(selected_holdout),
        "historical_check_buy_hold_10bps": compact(buy_hold),
        "all_results": {k: compact(v) for k, v in results.items()},
        "test_period_used_for_exit_selection": False,
        "historical_check_is_pristine": False,
        "warning": "Exit selection uses 2017-2021 only. 2022+ was previously inspected and is descriptive, not a pristine blind test. Results do not establish 200% CAGR."
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_tree_exit_rule_research.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for key, value in results.items():
        pd.DataFrame(value["trades"]).to_csv(out / f"{key}_trades.csv", index=False)
        pd.DataFrame(value["equity_curve"]).to_csv(out / f"{key}_equity.csv", index=False)
    return summary


def main():
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
