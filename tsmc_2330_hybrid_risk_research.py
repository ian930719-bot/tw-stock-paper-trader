"""Trend/ATR risk study for the frozen TSMC MACD-divergence hybrid.

The low entry and high exit rules are frozen from prior 2017-2021 selection.
This stage compares only predeclared entry trend filters and ATR trailing stops,
selecting on 2017-2021 and evaluating the frozen choice on 2022+.
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
from tsmc_2330_reversal_confirmation_backtest import build_confirmed_signals
from tsmc_2330_divergence_research import candidate_signals
from tsmc_2330_trend_atr_risk_research import apply_entry_filter

VALIDATION = ("2017-01-01", "2021-12-31")
HOLDOUT_START = "2022-01-01"
ENTRY_FILTERS = ("none", "above_sma200", "positive_sma50_slope", "trend_stack")
ATR_STOPS = (None, 2.5, 4.0)
SELECTION_SLIPPAGE_BPS = 10


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
    low, _ = build_confirmed_signals(bars, rules, "break_prior_range")
    base_low, base_high = build_signals(bars, rules)
    high = candidate_signals(bars, base_low, base_high)["HIGH"]["base_high_and_bearish_macd"]
    limit_lookup = {
        row.date.strftime("%Y-%m-%d"): (float(row.limit_up), float(row.limit_down))
        for row in limits.itertuples(index=False)
        if np.isfinite(row.limit_up) and np.isfinite(row.limit_down)
    }
    candidates = {}
    for entry_filter in ENTRY_FILTERS:
        filtered_low = apply_entry_filter(bars, low, entry_filter)
        for stop in ATR_STOPS:
            val = _simulate_period(
                bars, filtered_low, high, limit_lookup, dividends,
                *VALIDATION, SELECTION_SLIPPAGE_BPS, initial_capital, True,
                atr_stop_multiple=stop,
            )
            score = float(val["cagr"] - 0.5 * abs(val["max_drawdown"])) if val["cagr"] is not None else -1.0
            if val["trade_count"] < 5:
                score = -1.0
            key = f"{entry_filter}_atr_{'none' if stop is None else str(stop).replace('.', 'p')}"
            candidates[key] = {
                "entry_filter": entry_filter, "atr_stop_multiple": stop,
                "selection_score": score, "validation": compact(val),
                "entry_signal_days": int(filtered_low.sum()),
                "exit_signal_days": int(high.sum()),
            }
    selected = max(candidates, key=lambda k: (
        candidates[k]["selection_score"],
        candidates[k]["validation"]["cagr"] or -1.0,
    ))
    chosen = candidates[selected]
    filtered_low = apply_entry_filter(bars, low, chosen["entry_filter"])
    holdout = {}
    for slip in SLIPPAGE_BPS:
        result = _simulate_period(
            bars, filtered_low, high, limit_lookup, dividends,
            HOLDOUT_START, bars["date"].max().date().isoformat(),
            slip, initial_capital, True,
            atr_stop_multiple=chosen["atr_stop_multiple"],
        )
        holdout[f"hybrid_{slip}bps"] = result
    baseline = _simulate_period(
        bars, low, high, limit_lookup, dividends,
        HOLDOUT_START, bars["date"].max().date().isoformat(),
        10, initial_capital, True,
    )
    buy_hold = _simulate_period(
        bars, filtered_low, high, limit_lookup, dividends,
        HOLDOUT_START, bars["date"].max().date().isoformat(),
        10, initial_capital, False,
    )
    summary = {
        "symbol": "2330",
        "frozen_entry_rule": "prior-range breakout confirmation",
        "frozen_exit_rule": "base high setup plus bearish MACD divergence20",
        "entry_filters": list(ENTRY_FILTERS),
        "atr_stop_candidates": list(ATR_STOPS),
        "selection_period": list(VALIDATION),
        "selection_slippage_bps_per_side": SELECTION_SLIPPAGE_BPS,
        "selection_objective": "validation CAGR - 0.5*abs(max drawdown), minimum five trades",
        "candidate_validation_results": candidates,
        "selected_variant": selected,
        "selected_parameters": {
            "entry_filter": chosen["entry_filter"],
            "atr_stop_multiple": chosen["atr_stop_multiple"],
        },
        "historical_holdout_period": [HOLDOUT_START, bars["date"].max().date().isoformat()],
        "historical_holdout_results": {k: compact(v) for k, v in holdout.items()},
        "historical_holdout_hybrid_without_risk_filter_10bps": compact(baseline),
        "historical_holdout_buy_hold_10bps": compact(buy_hold),
        "test_period_used_for_selection": False,
        "holdout_is_pristine": False,
        "warning": "2022+ has been inspected in previous diagnostics and is not a pristine blind holdout. Candidate selection uses 2017-2021 only.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_hybrid_risk_research.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([
        {"variant": name, "entry_filter": value["entry_filter"],
         "atr_stop_multiple": value["atr_stop_multiple"],
         "selection_score": value["selection_score"], **value["validation"]}
        for name, value in candidates.items()
    ]).to_csv(out / "tsmc_2330_hybrid_risk_validation.csv", index=False)
    for key, value in holdout.items():
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
