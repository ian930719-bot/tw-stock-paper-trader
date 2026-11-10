"""ATR trailing-stop sensitivity for the frozen TSMC tree entry + SMA20 exit.

Entry: fixed one-day tree low rule.
Exit: fixed high setup armed, then close below SMA20 with negative daily return.
Only the ATR trailing-stop multiple is selected on 2017-2021. 2022+ is a
descriptive historical check and has already been inspected in prior research.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tsmc_2330_execution_backtest import (
    SLIPPAGE_BPS, INITIAL_CAPITAL, load_inputs, _simulate_period, build_signals,
)
from tsmc_2330_tree_rule_execution_backtest import tree_low_setup
from tsmc_2330_reversal_confirmation_backtest import build_confirmed_signals

VALIDATION = ("2017-01-01", "2021-12-31")
HOLDOUT_START = "2022-01-01"
ATR_STOPS = (None, 2.5, 3.0, 3.5, 4.0, 5.0)
SELECTION_SLIPPAGE_BPS = 10


def score(result: dict) -> float:
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
    _, high_signal = build_confirmed_signals(bars, rules, "sma20_reclaim")
    limit_lookup = {
        row.date.strftime("%Y-%m-%d"): (float(row.limit_up), float(row.limit_down))
        for row in limits.itertuples(index=False)
        if np.isfinite(row.limit_up) and np.isfinite(row.limit_down)
    }
    results = {}
    for stop in ATR_STOPS:
        stop_name = "none" if stop is None else f"{stop:g}atr"
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
                    atr_stop_multiple=stop,
                )
                results[f"{period_name}_{stop_name}_{slip}bps"] = result

    validation_10 = {
        k: v for k, v in results.items()
        if k.startswith("validation_2017_2021_") and k.endswith("_10bps")
    }
    selected_key = max(validation_10, key=lambda k: (score(validation_10[k]), validation_10[k]["cagr"]))
    selected_name = selected_key.removeprefix("validation_2017_2021_").removesuffix("_10bps")
    holdout_key = f"historical_check_2022_plus_{selected_name}_10bps"
    buy_hold = _simulate_period(
        bars, low_signal, high_signal, limit_lookup, dividends,
        HOLDOUT_START, bars["date"].max().date().isoformat(),
        10, initial_capital, False,
    )
    summary = {
        "symbol": "2330",
        "frozen_entry": "return_1 <= -0.0025693492 AND close_location <= 0.25657895; next-session open",
        "frozen_exit": "high setup, then close below SMA20 AND return_1 < 0 within 5 sessions; next-session open",
        "atr_stop_grid": [None if x is None else float(x) for x in ATR_STOPS],
        "selection_metric": "CAGR - 0.5 * abs(max_drawdown), at 10 bps one-way slippage",
        "selection_period": list(VALIDATION),
        "selected_atr_stop": selected_name,
        "selected_validation_result": compact(validation_10[selected_key]),
        "selected_historical_check_result": compact(results[holdout_key]),
        "historical_check_buy_hold_10bps": compact(buy_hold),
        "all_results": {k: compact(v) for k, v in results.items()},
        "test_period_used_for_stop_selection": False,
        "historical_check_is_pristine": False,
        "warning": "ATR stop selection uses 2017-2021 only. 2022+ was previously inspected and is descriptive, not a pristine blind test. Stops trigger at close and execute at the next tradable open; limit-locked exits remain pending."
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_tree_atr_risk_research.json").write_text(
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
