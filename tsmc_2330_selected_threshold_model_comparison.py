"""Compare predeclared causal models for the training-selected TSMC pivot threshold.

Models are selected by 2017-2021 validation PR-AUC, then refit through 2021
before the 2022+ descriptive historical check. No test-period model selection.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from tsmc_swing_research import detect_confirmed_pivots
from tsmc_2330_selected_threshold_signal_diagnostics import period_events, build_target

HORIZONS = (1, 3, 5)
VALIDATION_START = "2017-01-01"
VALIDATION_END = "2021-12-31"
TEST_START = "2022-01-01"
MODEL_NAMES = ("logistic", "random_forest", "hist_gradient_boosting")
EXCLUDE = {
    "symbol", "date", "open", "high", "low", "close", "volume",
    "turnover", "corporate_action_window", "atr14",
}


def make_model(name: str):
    if name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced", C=0.2, random_state=0
            ),
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
        weights = compute_sample_weight("balanced", y)
        model.fit(X, y, histgradientboostingclassifier__sample_weight=weights)
    else:
        model.fit(X, y)
    return model


def evaluate(model, X: pd.DataFrame, mask: np.ndarray, y: np.ndarray) -> dict:
    probability = model.predict_proba(X.loc[mask])[:, 1]
    truth = y[mask]
    prevalence = float(truth.mean())
    out = {
        "rows": int(mask.sum()),
        "event_days": int(truth.sum()),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(truth, probability)),
        "pr_auc": float(average_precision_score(truth, probability)),
    }
    for fraction in (0.01, 0.05, 0.10, 0.20):
        count = max(1, int(np.ceil(len(probability) * fraction)))
        top = np.argsort(probability)[-count:]
        precision = float(truth[top].mean())
        suffix = str(int(fraction * 100))
        out[f"top_{suffix}_precision"] = precision
        out[f"top_{suffix}_lift"] = precision / prevalence if prevalence else None
        out[f"top_{suffix}_recall"] = float(truth[top].sum() / truth.sum()) if truth.sum() else None
    return out


def run(features_path: str, oracle_path: str, output_dir: str) -> dict:
    x = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    oracle = json.loads(Path(oracle_path).read_text(encoding="utf-8"))
    selected = oracle.get("threshold_selected_using_training_period_only", {})
    min_reversal = selected.get("min_reversal_pct")
    atr_multiple = selected.get("atr_multiple")
    if min_reversal is None or atr_multiple is None:
        raise ValueError("Missing training-selected pivot threshold")
    events = detect_confirmed_pivots(x, float(min_reversal), float(atr_multiple))
    if events.empty:
        raise ValueError("No pivot events for selected threshold")
    feature_cols = [
        col for col in x.columns
        if col not in EXCLUDE and not col.startswith("raw_")
        and pd.api.types.is_numeric_dtype(x[col])
    ]
    X = x[feature_cols].replace([np.inf, -np.inf], np.nan)
    valid = X.notna().any(axis=1).to_numpy(copy=True)
    if "corporate_action_window" in x:
        valid &= ~x["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    dates = x["date"]
    train_start_idx = int(np.flatnonzero((dates >= pd.Timestamp("2010-01-01")).to_numpy())[0])
    val_start_idx = int(np.flatnonzero((dates >= pd.Timestamp(VALIDATION_START)).to_numpy())[0])
    test_start_idx = int(np.flatnonzero((dates >= pd.Timestamp(TEST_START)).to_numpy())[0])
    rows = []
    selections = {}
    for horizon in HORIZONS:
        idx = np.arange(len(x))
        train_mask = (idx >= train_start_idx + horizon) & (idx <= val_start_idx - horizon - 1) & valid
        val_mask = (
            (dates >= VALIDATION_START).to_numpy()
            & (dates <= VALIDATION_END).to_numpy()
            & (idx >= val_start_idx + horizon)
            & (idx <= test_start_idx - horizon - 1)
            & valid
        )
        test_mask = (idx >= test_start_idx + horizon) & valid
        train_events = period_events(events, dates, "2010-01-01", "2016-12-31", horizon)
        val_events = period_events(events, dates, VALIDATION_START, VALIDATION_END, horizon)
        test_events = period_events(events, dates, TEST_START, dates.max().date().isoformat(), horizon)
        train_val_events = pd.concat([train_events, val_events], ignore_index=True)
        train_val_mask = (idx >= train_start_idx + horizon) & (idx <= test_start_idx - horizon - 1) & valid
        for kind in ("HIGH", "LOW"):
            y_train = build_target(dates, train_events, horizon, kind)
            y_val = build_target(dates, val_events, horizon, kind)
            y_test = build_target(dates, test_events, horizon, kind)
            y_train_val = build_target(dates, train_val_events, horizon, kind)
            if len(np.unique(y_train[train_mask])) < 2:
                continue
            if not val_mask.any() or len(np.unique(y_val[val_mask])) < 2:
                selections[f"{kind}_{horizon}"] = {
                    "status": "skipped",
                    "reason": "validation segment lacks rows or both target classes",
                    "validation_rows": int(val_mask.sum()),
                    "validation_event_days": int(y_val[val_mask].sum()),
                }
                continue
            candidate_results = {}
            for name in MODEL_NAMES:
                model = fit_model(make_model(name), name, X.loc[train_mask], y_train[train_mask])
                metrics = evaluate(model, X, val_mask, y_val)
                candidate_results[name] = {"validation": metrics}
            chosen = max(MODEL_NAMES, key=lambda name: (
                candidate_results[name]["validation"]["pr_auc"],
                candidate_results[name]["validation"]["top_10_precision"],
            ))
            final_model = fit_model(
                make_model(chosen), chosen, X.loc[train_val_mask], y_train_val[train_val_mask]
            )
            test_metrics = (
                evaluate(final_model, X, test_mask, y_test)
                if test_mask.any() and len(np.unique(y_test[test_mask])) == 2
                else {
                    "error": "test segment lacks rows or both target classes",
                    "rows": int(test_mask.sum()),
                    "event_days": int(y_test[test_mask].sum()),
                }
            )
            selections[f"{kind}_{horizon}"] = {
                "selected_model": chosen,
                "validation_candidates": candidate_results,
                "test_metrics": test_metrics,
            }
            rows.append({
                "pivot_type": kind, "horizon_sessions": horizon,
                "selected_model": chosen,
                "validation_pr_auc": candidate_results[chosen]["validation"].get("pr_auc"),
                "validation_auc": candidate_results[chosen]["validation"].get("roc_auc"),
                "test_auc": test_metrics.get("roc_auc"),
                "test_pr_auc": test_metrics.get("pr_auc"),
                "test_top_1_precision": test_metrics.get("top_1_precision"),
                "test_top_5_precision": test_metrics.get("top_5_precision"),
                "test_top_10_precision": test_metrics.get("top_10_precision"),
                "test_top_10_lift": test_metrics.get("top_10_lift"),
                "test_top_10_recall": test_metrics.get("top_10_recall"),
                "train_rows": int(train_mask.sum()),
                "validation_rows": int(val_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "feature_count": len(feature_cols),
            })
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "tsmc_2330_selected_threshold_model_comparison.csv", index=False)
    summary = {
        "symbol": "2330",
        "pivot_threshold": {"min_reversal_pct": min_reversal, "atr_multiple": atr_multiple},
        "models": list(MODEL_NAMES),
        "model_selection_metric": "2017-2021 validation PR-AUC; tie-break top-10 precision",
        "train_period": ["2010-01-01", "2016-12-31"],
        "validation_period": [VALIDATION_START, VALIDATION_END],
        "test_period": [TEST_START, x["date"].max().date().isoformat()],
        "feature_count": len(feature_cols), "feature_columns": feature_cols,
        "results": selections,
        "test_period_used_for_model_selection": False,
        "test_period_is_pristine": False,
        "strategy_backtest": False,
        "warning": "Model families and hyperparameters were predeclared; selection is on validation PR-AUC only. 2022+ was previously inspected and is descriptive. No trades or CAGR are simulated.",
    }
    (out / "tsmc_2330_selected_threshold_model_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.oracle, args.output), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
