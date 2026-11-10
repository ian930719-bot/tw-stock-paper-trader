"""Probability-ranked TSMC 2330 swing strategy with validation-only signal calibration.

A single 5-session-ahead pivot classifier is trained on 2010-2016.
Model family and low/high probability cutoffs are selected on 2017-2021 only.
The frozen models and cutoffs are then evaluated on 2022+ as descriptive data;
that period has previously been inspected and is not a pristine blind holdout.

Signals are computed after close t and traded no earlier than next session open.
The execution simulator includes commissions, sell tax, slippage, dividends,
historical price limits, and one-share cash constraints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from tsmc_swing_research import detect_confirmed_pivots
from tsmc_2330_selected_threshold_signal_diagnostics import (
    period_events, build_target,
)
from tsmc_2330_execution_backtest import (
    SLIPPAGE_BPS, INITIAL_CAPITAL, load_inputs, _simulate_period,
)

TRAIN_START, TRAIN_END = "2010-01-01", "2016-12-31"
VALID_START, VALID_END = "2017-01-01", "2021-12-31"
TEST_START = "2022-01-01"
HORIZON = 5
MODEL_NAMES = ("logistic", "random_forest", "hist_gradient_boosting")
QUANTILES = (0.70, 0.80, 0.85, 0.90, 0.95)
EXCLUDE = {
    "obv", "symbol", "date", "open", "high", "low", "close", "volume",
    "turnover", "corporate_action_window", "atr14",
}


def make_model(name: str):
    if name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               C=0.2, random_state=0),
        )
    if name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=400, max_depth=6, min_samples_leaf=15,
                max_features=0.6, class_weight="balanced_subsample",
                random_state=0, n_jobs=1,
            ),
        )
    if name == "hist_gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=250, max_leaf_nodes=15, learning_rate=0.04,
                l2_regularization=5.0, max_bins=127,
                early_stopping=False, random_state=0,
            ),
        )
    raise ValueError(f"Unknown model: {name}")


def fit_model(model, name: str, X: pd.DataFrame, y: np.ndarray):
    if name == "hist_gradient_boosting":
        model.fit(X, y, histgradientboostingclassifier__sample_weight=
                  compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def probability_signal(probability: np.ndarray, threshold: float) -> np.ndarray:
    p = np.asarray(probability, dtype=float)
    return np.isfinite(p) & (p >= float(threshold))


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


def run(features_path: str, oracle_path: str, limits_path: str,
        dividends_path: str, output_dir: str,
        initial_capital: float = INITIAL_CAPITAL) -> dict:
    bars, _, limits, dividends = load_inputs(
        features_path, oracle_path, limits_path, dividends_path
    )
    oracle = json.loads(Path(oracle_path).read_text(encoding="utf-8"))
    selected = oracle.get("threshold_selected_using_training_period_only", {})
    min_reversal, atr_multiple = selected.get("min_reversal_pct"), selected.get("atr_multiple")
    if min_reversal is None or atr_multiple is None:
        raise ValueError("Missing training-only pivot threshold")
    events = detect_confirmed_pivots(bars, float(min_reversal), float(atr_multiple))
    if events.empty:
        raise ValueError("No pivot events")

    feature_cols = [
        c for c in bars.columns
        if c not in EXCLUDE and not c.startswith("raw_")
        and pd.api.types.is_numeric_dtype(bars[c])
        and bars[c].notna().any()
    ]
    X = bars[feature_cols].replace([np.inf, -np.inf], np.nan)
    dates = bars["date"]
    idx = np.arange(len(bars))
    train_start_idx = int(np.flatnonzero((dates >= pd.Timestamp(TRAIN_START)).to_numpy())[0])
    valid_start_idx = int(np.flatnonzero((dates >= pd.Timestamp(VALID_START)).to_numpy())[0])
    test_start_idx = int(np.flatnonzero((dates >= pd.Timestamp(TEST_START)).to_numpy())[0])
    valid_rows = X.notna().any(axis=1).to_numpy()
    action = bars["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    valid_rows &= ~action

    train_events = period_events(events, dates, TRAIN_START, TRAIN_END, HORIZON)
    valid_events = period_events(events, dates, VALID_START, VALID_END, HORIZON)
    test_events = period_events(events, dates, TEST_START, dates.max().date().isoformat(), HORIZON)
    train_mask = (idx >= train_start_idx + HORIZON) & (idx <= valid_start_idx - HORIZON - 1) & valid_rows
    valid_mask = ((dates >= VALID_START) & (dates <= VALID_END)).to_numpy()
    valid_mask &= (idx >= valid_start_idx + HORIZON) & (idx <= test_start_idx - HORIZON - 1) & valid_rows
    test_mask = (idx >= test_start_idx + HORIZON) & valid_rows

    models, probabilities, model_metrics = {}, {}, {}
    for kind in ("LOW", "HIGH"):
        y_train = build_target(dates, train_events, HORIZON, kind)
        y_valid = build_target(dates, valid_events, HORIZON, kind)
        if len(np.unique(y_train[train_mask])) != 2 or len(np.unique(y_valid[valid_mask])) != 2:
            raise ValueError(f"{kind}: training or validation target has one class")
        candidates = {}
        for name in MODEL_NAMES:
            model = fit_model(make_model(name), name, X.loc[train_mask], y_train[train_mask])
            p_valid = model.predict_proba(X.loc[valid_mask])[:, 1]
            candidates[name] = {
                "pr_auc": float(average_precision_score(y_valid[valid_mask], p_valid)),
                "validation_prevalence": float(y_valid[valid_mask].mean()),
            }
        chosen = max(MODEL_NAMES, key=lambda n: candidates[n]["pr_auc"])
        model = fit_model(make_model(chosen), chosen, X.loc[train_mask], y_train[train_mask])
        p_all = np.full(len(bars), np.nan, dtype=float)
        p_all[valid_mask] = model.predict_proba(X.loc[valid_mask])[:, 1]
        p_all[test_mask] = model.predict_proba(X.loc[test_mask])[:, 1]
        models[kind] = model
        probabilities[kind] = p_all
        model_metrics[kind] = {
            "selected_model": chosen, "validation_candidates": candidates,
            "validation_pr_auc": candidates[chosen]["pr_auc"],
            "validation_event_prevalence": candidates[chosen]["validation_prevalence"],
        }

    # Select a pair of probability cutoffs using only validation execution results.
    limit_lookup = {
        r.date.strftime("%Y-%m-%d"): (float(r.limit_up), float(r.limit_down))
        for r in limits.itertuples(index=False)
        if np.isfinite(r.limit_up) and np.isfinite(r.limit_down)
    }
    candidates = []
    for q_low in QUANTILES:
        low_threshold = float(np.nanquantile(probabilities["LOW"][valid_mask], q_low))
        low_signal = probability_signal(probabilities["LOW"], low_threshold)
        for q_high in QUANTILES:
            high_threshold = float(np.nanquantile(probabilities["HIGH"][valid_mask], q_high))
            high_signal = probability_signal(probabilities["HIGH"], high_threshold)
            result = _simulate_period(
                bars, low_signal, high_signal, limit_lookup, dividends,
                VALID_START, VALID_END, 10, initial_capital, True,
            )
            candidates.append({
                "low_quantile": q_low, "high_quantile": q_high,
                "low_threshold": low_threshold, "high_threshold": high_threshold,
                "validation": compact(result), "score": score(result),
            })
    eligible = [c for c in candidates if c["validation"]["trade_count"] >= 5]
    if not eligible:
        raise ValueError("No probability-threshold pair produced five validation trades")
    chosen = max(eligible, key=lambda c: (c["score"], c["validation"]["cagr"]))
    low_signal = probability_signal(probabilities["LOW"], chosen["low_threshold"])
    high_signal = probability_signal(probabilities["HIGH"], chosen["high_threshold"])

    results = {}
    for slip in SLIPPAGE_BPS:
        result = _simulate_period(
            bars, low_signal, high_signal, limit_lookup, dividends,
            TEST_START, dates.max().date().isoformat(), slip, initial_capital, True,
        )
        results[f"strategy_{slip}bps"] = compact(result)
    buy_hold = _simulate_period(
        bars, low_signal, high_signal, limit_lookup, dividends,
        TEST_START, dates.max().date().isoformat(), 10, initial_capital, False,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date": dates.dt.strftime("%Y-%m-%d"),
        "low_probability": probabilities["LOW"],
        "high_probability": probabilities["HIGH"],
        "low_signal": low_signal,
        "high_signal": high_signal,
    }).to_csv(out / "tsmc_2330_probability_strategy_signals.csv", index=False)
    summary = {
        "symbol": "2330", "horizon_sessions": HORIZON,
        "pivot_threshold": {"min_reversal_pct": min_reversal, "atr_multiple": atr_multiple},
        "feature_count": len(feature_cols), "feature_columns": feature_cols,
        "train_period": [TRAIN_START, TRAIN_END],
        "validation_period": [VALID_START, VALID_END],
        "historical_check_period": [TEST_START, dates.max().date().isoformat()],
        "models": model_metrics,
        "selected_probability_cutoffs": {
            "low_quantile": chosen["low_quantile"], "high_quantile": chosen["high_quantile"],
            "low_threshold": chosen["low_threshold"], "high_threshold": chosen["high_threshold"],
            "validation_result": chosen["validation"], "validation_score": chosen["score"],
        },
        "validation_candidate_count": len(candidates),
        "historical_check_results": results,
        "historical_check_buy_hold_10bps": compact(buy_hold),
        "execution": {
            "signal_timing": "probability at close t; next available session open",
            "commission_rate_per_side": 0.001425, "minimum_commission_ntd": 20,
            "sell_transaction_tax_rate": 0.003, "slippage_bps_per_side": list(SLIPPAGE_BPS),
            "price_limits": "historical TWSE price limits and tick sizes; locked orders handled conservatively",
            "dividends": "cash dividends credited to shares held at prior session close",
            "capital": initial_capital, "leverage": 1.0, "short_selling": False,
        },
        "test_period_used_for_selection": False,
        "historical_check_is_pristine": False,
        "warning": "2022+ was inspected in earlier research and is descriptive only. Model families and probability thresholds are selected on 2017-2021; no 200% CAGR claim is implied. Broker-branch data unavailable remains null, not zero.",
    }
    (out / "tsmc_2330_probability_strategy_backtest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--limits", required=True)
    parser.add_argument("--dividends", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.oracle, args.limits, args.dividends,
                         args.output, args.initial_capital), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
