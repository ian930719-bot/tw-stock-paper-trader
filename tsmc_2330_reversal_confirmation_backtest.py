"""Develop stateful reversal-confirmation variants for TSMC 2330.

Frozen low/high setups come from the 2010-2016 train / 2017-2021 validation
rule study. A setup arms for at most five sessions; an entry/exit is triggered
only after a separately specified reversal confirmation. Candidate confirmation
types are predeclared. The chosen variant is selected on 2017-2021 only and
then evaluated on the previously inspected 2022+ historical holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_execution_backtest import (
    SLIPPAGE_BPS, INITIAL_CAPITAL, load_inputs, build_signals,
    _simulate_period,
)

VALIDATION = ("2017-01-01", "2021-12-31")
HOLDOUT_START = "2022-01-01"
VARIANTS = (
    "unconfirmed_baseline",
    "positive_negative_close",
    "break_prior_range",
    "rsi_reversal",
    "sma5_reclaim",
    "sma20_reclaim",
)
MAX_WAIT = 5
SELECTION_SLIPPAGE_BPS = 10


def build_confirmed_signals(features: pd.DataFrame, rules: dict,
                            variant: str, max_wait: int = MAX_WAIT) -> tuple[np.ndarray, np.ndarray]:
    base_low, base_high = build_signals(features, rules)
    if variant == "unconfirmed_baseline":
        return base_low, base_high
    if variant not in VARIANTS:
        raise ValueError(f"Unknown confirmation variant: {variant}")
    required = {"return_1", "close_location", "rsi14", "distance_sma_5",
                "distance_sma_20", "high", "low", "close"}
    if not required.issubset(features.columns):
        raise ValueError(f"Missing reversal features: {sorted(required - set(features.columns))}")

    close = pd.to_numeric(features["close"], errors="coerce")
    prior_high = pd.to_numeric(features["high"], errors="coerce").shift(1)
    prior_low = pd.to_numeric(features["low"], errors="coerce").shift(1)
    ret1 = pd.to_numeric(features["return_1"], errors="coerce")
    location = pd.to_numeric(features["close_location"], errors="coerce")
    rsi = pd.to_numeric(features["rsi14"], errors="coerce")
    rsi_change = rsi.diff(3)
    dist5 = pd.to_numeric(features["distance_sma_5"], errors="coerce")
    dist20 = pd.to_numeric(features["distance_sma_20"], errors="coerce")

    if variant == "positive_negative_close":
        low_confirm = ret1.gt(0) & location.ge(0.60)
        high_confirm = ret1.lt(0) & location.le(0.40)
    elif variant == "break_prior_range":
        low_confirm = close.gt(prior_high) & location.ge(0.55)
        high_confirm = close.lt(prior_low) & location.le(0.45)
    elif variant == "rsi_reversal":
        low_confirm = ret1.gt(0) & rsi_change.gt(0) & location.ge(0.55)
        high_confirm = ret1.lt(0) & rsi_change.lt(0) & location.le(0.45)
    elif variant == "sma5_reclaim":
        low_confirm = dist5.gt(0) & location.ge(0.55)
        high_confirm = dist5.lt(0) & location.le(0.45)
    else:  # sma20_reclaim
        low_confirm = dist20.gt(0) & ret1.gt(0)
        high_confirm = dist20.lt(0) & ret1.lt(0)

    low_confirm = low_confirm.fillna(False).to_numpy()
    high_confirm = high_confirm.fillna(False).to_numpy()
    n = len(features)
    low_signal = np.zeros(n, dtype=bool)
    high_signal = np.zeros(n, dtype=bool)
    low_armed_until = -1
    high_armed_until = -1
    for i in range(n):
        if base_low[i] and low_armed_until < i:
            low_armed_until = i + max_wait
        if base_high[i] and high_armed_until < i:
            high_armed_until = i + max_wait
        if low_armed_until >= i and i > low_armed_until - max_wait and low_confirm[i]:
            low_signal[i] = True
            low_armed_until = -1
        if high_armed_until >= i and i > high_armed_until - max_wait and high_confirm[i]:
            high_signal[i] = True
            high_armed_until = -1
        if low_armed_until < i:
            low_armed_until = -1
        if high_armed_until < i:
            high_armed_until = -1

    if "corporate_action_window" in features:
        action = features["corporate_action_window"].fillna(True).astype(bool).to_numpy()
        low_signal &= ~action
        high_signal &= ~action
    return low_signal, high_signal


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
    limit_lookup = {
        row.date.strftime("%Y-%m-%d"): (float(row.limit_up), float(row.limit_down))
        for row in limits.itertuples(index=False)
        if np.isfinite(row.limit_up) and np.isfinite(row.limit_down)
    }
    variants = {}
    for variant in VARIANTS:
        low, high = build_confirmed_signals(bars, rules, variant)
        val = _simulate_period(
            bars, low, high, limit_lookup, dividends,
            *VALIDATION, SELECTION_SLIPPAGE_BPS, initial_capital, True,
        )
        # Penalize drawdown while retaining CAGR as the main selection objective.
        score = float(val["cagr"] - 0.5 * abs(val["max_drawdown"])) if val["cagr"] is not None else -1.0
        if val["trade_count"] < 5:
            score = -1.0
        variants[variant] = {
            "selection_score": score,
            "validation": compact(val),
            "signals": {"low_signal_days": int(low.sum()), "high_signal_days": int(high.sum())},
        }
    selected = max(VARIANTS, key=lambda name: (
        variants[name]["selection_score"],
        variants[name]["validation"]["cagr"] or -1.0,
    ))
    selected_low, selected_high = build_confirmed_signals(bars, rules, selected)
    holdout = {}
    for slip in SLIPPAGE_BPS:
        result = _simulate_period(
            bars, selected_low, selected_high, limit_lookup, dividends,
            HOLDOUT_START, bars["date"].max().date().isoformat(),
            slip, initial_capital, True,
        )
        holdout[f"strategy_{slip}bps"] = result
    benchmark = _simulate_period(
        bars, selected_low, selected_high, limit_lookup, dividends,
        HOLDOUT_START, bars["date"].max().date().isoformat(),
        SELECTION_SLIPPAGE_BPS, initial_capital, False,
    )
    summary = {
        "symbol": "2330",
        "frozen_setup_rules": rules.get("rules", {}),
        "candidate_confirmation_variants": list(VARIANTS),
        "max_wait_sessions_after_setup": MAX_WAIT,
        "selection_period": list(VALIDATION),
        "selection_slippage_bps_per_side": SELECTION_SLIPPAGE_BPS,
        "selection_objective": "validation CAGR minus 0.5 times absolute maximum drawdown; require at least five completed trades",
        "candidate_validation_results": variants,
        "selected_variant": selected,
        "historical_holdout_period": [HOLDOUT_START, bars["date"].max().date().isoformat()],
        "historical_holdout_results": {
            key: compact(value) for key, value in holdout.items()
        },
        "historical_holdout_buy_hold_10bps": compact(benchmark),
        "test_period_used_for_selection": False,
        "holdout_is_pristine": False,
        "strategy_backtest": True,
        "warning": "The 2022+ period was already examined in prior diagnostics and is not a pristine blind holdout. These candidate comparisons are exploratory; forward paper validation is required. No parameter was selected using 2022+.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_reversal_confirmation_backtest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([
        {"variant": name, "selection_score": value["selection_score"],
         **value["validation"]}
        for name, value in variants.items()
    ]).to_csv(out / "tsmc_2330_reversal_confirmation_validation.csv", index=False)
    for key, value in holdout.items():
        pd.DataFrame(value["trades"]).to_csv(out / f"reversal_{selected}_{key}_trades.csv", index=False)
        pd.DataFrame(value["equity_curve"]).to_csv(out / f"reversal_{selected}_{key}_equity.csv", index=False)
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
