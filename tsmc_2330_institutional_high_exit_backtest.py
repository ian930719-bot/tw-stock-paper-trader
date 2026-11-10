"""Execution-aware TSMC 2330 exit backtest using high-pivot model probabilities.

The high-pivot model is fit on 2010-2016 only. Its probability cutoff is the
90th percentile of 2017-2021 validation probabilities and is frozen unchanged
for the 2022+ descriptive historical check. Signal at close t executes next
available session open. Entry remains the frozen tree low setup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_execution_backtest import (
    SLIPPAGE_BPS, INITIAL_CAPITAL, load_inputs, _simulate_period,
    build_signals,
)
from tsmc_2330_tree_rule_execution_backtest import tree_low_setup
from tsmc_2330_reversal_confirmation_backtest import build_confirmed_signals
from tsmc_2330_chip_factor_ablation import feature_groups, make_model, evaluate
from tsmc_2330_selected_threshold_signal_diagnostics import period_events, build_target
from tsmc_swing_research import detect_confirmed_pivots

TRAIN_START, TRAIN_END = "2010-01-01", "2016-12-31"
VALIDATION_START, VALIDATION_END = "2017-01-01", "2021-12-31"
TEST_START = "2022-01-01"
HIGH_HORIZON = 1
VALIDATION_QUANTILE = 0.90


def frozen_probability_threshold(probabilities: np.ndarray, quantile: float = VALIDATION_QUANTILE) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or not 0 < quantile < 1:
        raise ValueError("Need finite validation probabilities and a quantile between 0 and 1")
    return float(np.quantile(values, quantile))


def train_high_models(bars: pd.DataFrame, oracle_path: str) -> tuple[dict, dict]:
    oracle = json.loads(Path(oracle_path).read_text(encoding="utf-8"))
    selected = oracle.get("threshold_selected_using_training_period_only", {})
    min_reversal, atr_multiple = selected.get("min_reversal_pct"), selected.get("atr_multiple")
    if min_reversal is None or atr_multiple is None:
        raise ValueError("Missing frozen training-selected pivot threshold")
    events = detect_confirmed_pivots(bars, float(min_reversal), float(atr_multiple))
    if events.empty:
        raise ValueError("No high pivots for frozen threshold")
    dates = bars["date"]
    idx = np.arange(len(bars))
    train_mask = ((dates >= TRAIN_START) & (dates <= TRAIN_END)).to_numpy(copy=True)
    validation_mask = ((dates >= VALIDATION_START) & (dates <= VALIDATION_END)).to_numpy(copy=True)
    safe = ~bars.get("corporate_action_window", pd.Series(False, index=bars.index)).fillna(True).astype(bool).to_numpy(copy=True)
    train_start_idx = int(np.flatnonzero((dates >= pd.Timestamp(TRAIN_START)).to_numpy())[0])
    val_start_idx = int(np.flatnonzero((dates >= pd.Timestamp(VALIDATION_START)).to_numpy())[0])
    train_mask &= safe & (idx >= train_start_idx + HIGH_HORIZON)
    validation_mask &= safe & (idx >= val_start_idx + HIGH_HORIZON)
    train_events = period_events(events, dates, TRAIN_START, TRAIN_END, HIGH_HORIZON)
    validation_events = period_events(events, dates, VALIDATION_START, VALIDATION_END, HIGH_HORIZON)
    y_train = build_target(dates, train_events, HIGH_HORIZON, "HIGH")
    y_validation = build_target(dates, validation_events, HIGH_HORIZON, "HIGH")
    if len(np.unique(y_train[train_mask])) < 2 or len(np.unique(y_validation[validation_mask])) < 2:
        raise ValueError("High-pivot target lacks both classes in train or validation")

    groups = feature_groups(bars)
    outputs = {}
    for group_name in ("technical_only", "technical_plus_institutional"):
        cols = groups[group_name]
        X = bars[cols].replace([np.inf, -np.inf], np.nan)
        train_cols = [c for c in cols if X.loc[train_mask, c].notna().any()]
        X = X[train_cols]
        model = make_model()
        model.fit(X.loc[train_mask], y_train[train_mask])
        val_prob = model.predict_proba(X.loc[validation_mask])[:, 1]
        threshold = frozen_probability_threshold(val_prob)
        all_prob = model.predict_proba(X)[:, 1]
        signals = all_prob >= threshold
        if "corporate_action_window" in bars:
            signals &= ~bars["corporate_action_window"].fillna(True).astype(bool).to_numpy(copy=True)
        validation_metrics = evaluate(model, X, validation_mask, y_validation)
        test_mask = ((dates >= TEST_START) & safe).to_numpy(copy=True)
        test_metrics = evaluate(model, X, test_mask, build_target(
            dates,
            period_events(events, dates, TEST_START, dates.max().date().isoformat(), HIGH_HORIZON),
            HIGH_HORIZON, "HIGH",
        ))
        outputs[group_name] = {
            "model": model,
            "signals": signals.astype(bool),
            "threshold": threshold,
            "feature_count": len(train_cols),
            "feature_columns": train_cols,
            "validation_signal_rate": float(signals[validation_mask].mean()),
            "test_signal_rate": float(signals[test_mask].mean()),
            "validation_metrics": validation_metrics,
            "test_metrics_2022_plus_descriptive": test_metrics,
        }
    metadata = {
        "pivot_threshold": {"min_reversal_pct": float(min_reversal), "atr_multiple": float(atr_multiple)},
        "training_period": [TRAIN_START, TRAIN_END],
        "validation_period": [VALIDATION_START, VALIDATION_END],
        "test_period": [TEST_START, dates.max().date().isoformat()],
        "high_pivot_horizon_sessions": HIGH_HORIZON,
        "probability_cutoff": "90th percentile of validation predictions; frozen for test",
        "signal_timing": "features known at close t; exit at next available session open",
        "model_refit_after_validation": False,
    }
    return outputs, metadata


def score(result: dict) -> float:
    cagr = result.get("cagr")
    drawdown = result.get("max_drawdown")
    if cagr is None or drawdown is None:
        return -1e9
    return float(cagr) - 0.5 * abs(float(drawdown))


def run(features_path: str, rules_path: str, limits_path: str,
        dividends_path: str, oracle_path: str, output_dir: str,
        initial_capital: float = INITIAL_CAPITAL) -> dict:
    bars, rules, limits, dividends = load_inputs(features_path, rules_path, limits_path, dividends_path)
    low_signal = tree_low_setup(bars)
    model_outputs, model_meta = train_high_models(bars, oracle_path)
    _, sma20_exit = build_confirmed_signals(bars, rules, "sma20_reclaim")
    exits = {
        "sma20_reclaim_baseline": sma20_exit,
        "technical_high_probability": model_outputs["technical_only"]["signals"],
        "institutional_high_probability": model_outputs["technical_plus_institutional"]["signals"],
    }
    limit_lookup = {
        row.date.strftime("%Y-%m-%d"): (float(row.limit_up), float(row.limit_down))
        for row in limits.itertuples(index=False)
        if np.isfinite(row.limit_up) and np.isfinite(row.limit_down)
    }
    periods = {
        "validation_2017_2021": (VALIDATION_START, VALIDATION_END),
        "historical_check_2022_plus": (TEST_START, bars["date"].max().date().isoformat()),
    }
    results = {}
    for exit_name, high_signal in exits.items():
        for period_name, (start, end) in periods.items():
            for slip in SLIPPAGE_BPS:
                results[f"{period_name}_{exit_name}_{slip}bps"] = _simulate_period(
                    bars, low_signal, high_signal, limit_lookup, dividends,
                    start, end, slip, initial_capital, True,
                )
    buy_hold = {}
    for period_name, (start, end) in periods.items():
        for slip in SLIPPAGE_BPS:
            buy_hold[f"{period_name}_{slip}bps"] = _simulate_period(
                bars, low_signal, sma20_exit, limit_lookup, dividends,
                start, end, slip, initial_capital, False,
            )
    validation_10 = {
        name: result for name, result in results.items()
        if name.startswith("validation_2017_2021_") and name.endswith("_10bps")
    }
    selected_key = max(validation_10, key=lambda name: (score(validation_10[name]), validation_10[name]["cagr"] or -1e9))
    selected_exit = selected_key.removeprefix("validation_2017_2021_").removesuffix("_10bps")
    summary = {
        "symbol": "2330",
        "entry_rule": "frozen tree low setup: return_1 <= -0.0025693492 AND close_location <= 0.25657895; next-session open",
        "exit_candidates": list(exits),
        "selected_exit_by_validation_score": selected_exit,
        "selection_metric": "CAGR - 0.5 * abs(max_drawdown), at 10 bps one-way slippage; 2017-2021 only",
        "model_metadata": model_meta,
        "model_classification": {
            name: {k: v for k, v in output.items() if k not in {"model", "signals"}}
            for name, output in model_outputs.items()
        },
        "results": results,
        "buy_hold": buy_hold,
        "slippage_scenarios_bps_per_side": list(SLIPPAGE_BPS),
        "test_period_is_pristine": False,
        "test_period_used_for_selection": False,
        "note": "High-pivot model is trained only through 2016 and not refit; threshold is fixed from 2017-2021 validation. Test-period results are descriptive. Execution uses next-session open, official daily price limits, dividends, commissions, sell tax and adverse slippage.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_institutional_high_exit_backtest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    flat = []
    for name, result in results.items():
        flat.append({
            "period_strategy": name, "cagr": result.get("cagr"),
            "total_return": result.get("total_return"), "max_drawdown": result.get("max_drawdown"),
            "trade_count": result.get("trade_count"), "win_rate": result.get("win_rate"),
            "exposure": result.get("exposure"),
        })
    pd.DataFrame(flat).to_csv(out / "tsmc_2330_institutional_high_exit_backtest.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--limits", required=True)
    parser.add_argument("--dividends", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.rules, args.limits, args.dividends,
                         args.oracle, args.output, args.initial_capital),
                     ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
