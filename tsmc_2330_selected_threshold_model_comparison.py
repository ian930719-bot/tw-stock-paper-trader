"""Compare fixed model families using annual expanding walk-forward folds.

Every calendar year is scored by models fit only on earlier data. Models are
reported side-by-side; no single winner is selected using pooled future folds.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from tsmc_swing_research import detect_confirmed_pivots
from tsmc_2330_walk_forward import (
    annual_fold_years, event_target, fold_events, fold_masks, score_predictions,
)

HORIZONS = (1, 3, 5)
MIN_REVERSAL_PCT = 0.05
ATR_MULTIPLE = 2.0
MODEL_NAMES = ("logistic", "random_forest", "hist_gradient_boosting")
EXCLUDE = {
    "obv", "symbol", "date", "open", "high", "low", "close", "volume",
    "turnover", "corporate_action_window", "atr14",
}


def make_model(name: str):
    if name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.2, random_state=0),
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
        model.fit(X, y, histgradientboostingclassifier__sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model



# Compatibility evaluator retained for existing unit tests and dependent studies.
def evaluate(model, X: pd.DataFrame, mask: np.ndarray, y: np.ndarray) -> dict:
    probability = model.predict_proba(X.loc[mask])[:, 1]
    truth = np.asarray(y[mask], dtype=np.int8)
    prevalence = float(truth.mean()) if len(truth) else 0.0
    out = {
        "rows": int(mask.sum()),
        "event_days": int(truth.sum()),
        "prevalence": prevalence,
        "roc_auc": score_predictions(truth, probability).get("roc_auc"),
        "pr_auc": score_predictions(truth, probability).get("pr_auc"),
    }
    for fraction in (0.01, 0.05, 0.10, 0.20):
        count = max(1, int(np.ceil(len(probability) * fraction)))
        top = np.argsort(probability)[-count:]
        precision = float(truth[top].mean()) if len(truth) else 0.0
        suffix = str(int(fraction * 100))
        out[f"top_{suffix}_precision"] = precision
        out[f"top_{suffix}_lift"] = precision / prevalence if prevalence else None
        out[f"top_{suffix}_recall"] = float(truth[top].sum() / truth.sum()) if truth.sum() else None
    return out

def run(features_path: str, oracle_path: str | None, output_dir: str) -> dict:
    x = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    events = detect_confirmed_pivots(x, MIN_REVERSAL_PCT, ATR_MULTIPLE)
    if events.empty:
        raise ValueError("Fixed pivot definition produced no events")
    feature_cols = [
        c for c in x.columns
        if c not in EXCLUDE and not c.startswith("raw_") and pd.api.types.is_numeric_dtype(x[c])
    ]
    X = x[feature_cols].replace([np.inf, -np.inf], np.nan)
    valid = X.notna().any(axis=1).to_numpy(copy=True)
    if "corporate_action_window" in x:
        valid &= ~x["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    dates = x["date"]
    folds, pooled = [], {}
    for horizon in HORIZONS:
        for kind in ("HIGH", "LOW"):
            for name in MODEL_NAMES:
                pooled[f"{kind}_{horizon}_{name}"] = {"truth": [], "probability": []}
            for year in annual_fold_years(dates):
                fold = fold_masks(dates, valid, year, horizon)
                if fold is None or not fold["validation_mask"].any():
                    continue
                train_events, val_events = fold_events(events, dates, fold, horizon)
                y_train = event_target(dates, train_events, horizon, kind)
                y_val = event_target(dates, val_events, horizon, kind)
                train_mask, val_mask = fold["train_mask"], fold["validation_mask"]
                if len(np.unique(y_train[train_mask])) < 2:
                    folds.append({
                        "pivot_type": kind, "horizon_sessions": horizon,
                        "validation_year": year, "status": "skipped",
                        "reason": "training fold lacks both classes",
                        "training_rows": int(train_mask.sum()),
                        "training_event_days": int(y_train[train_mask].sum()),
                    })
                    continue
                for name in MODEL_NAMES:
                    model = fit_model(make_model(name), name, X.loc[train_mask], y_train[train_mask])
                    probability = model.predict_proba(X.loc[val_mask])[:, 1]
                    truth = y_val[val_mask]
                    metrics = score_predictions(truth, probability)
                    key = f"{kind}_{horizon}_{name}"
                    pooled[key]["truth"].extend(truth.astype(int).tolist())
                    pooled[key]["probability"].extend(probability.astype(float).tolist())
                    folds.append({
                        "pivot_type": kind, "horizon_sessions": horizon,
                        "validation_year": year, "model": name, "status": "evaluated",
                        "training_end": fold["training_end"],
                        "validation_start": fold["validation_start"],
                        "validation_end": fold["validation_end"],
                        "partial_validation_year": fold["partial_validation_year"],
                        "training_rows": int(train_mask.sum()),
                        "training_event_days": int(y_train[train_mask].sum()),
                        "validation_metrics": metrics,
                    })
    pooled_metrics = {
        key: score_predictions(np.asarray(v["truth"]), np.asarray(v["probability"]))
        for key, v in pooled.items()
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    flat = []
    for r in folds:
        m = r.get("validation_metrics", {})
        flat.append({
            "pivot_type": r["pivot_type"], "horizon_sessions": r["horizon_sessions"],
            "validation_year": r["validation_year"], "model": r.get("model"),
            "status": r["status"], "training_rows": r.get("training_rows"),
            "training_event_days": r.get("training_event_days"),
            "validation_rows": m.get("rows"), "validation_event_days": m.get("event_days"),
            "validation_roc_auc": m.get("roc_auc"), "validation_pr_auc": m.get("pr_auc"),
            "validation_top_10_precision": m.get("top_10_precision"),
            "validation_top_10_lift": m.get("top_10_lift"),
        })
    pd.DataFrame(flat).to_csv(out / "tsmc_2330_selected_threshold_model_comparison.csv", index=False)
    summary = {
        "symbol": "2330",
        "pivot_threshold": {"min_reversal_pct": MIN_REVERSAL_PCT, "atr_multiple": ATR_MULTIPLE},
        "validation_method": "expanding-window annual walk-forward: train 2010..Y-1, validate Y",
        "models": list(MODEL_NAMES),
        "model_selection": "none across future validation years; fixed model families compared on identical annual folds",
        "validation_years": annual_fold_years(dates),
        "feature_count": len(feature_cols), "feature_columns": feature_cols,
        "annual_folds": folds, "pooled_walk_forward_metrics": pooled_metrics,
        "strategy_backtest": False,
        "warning": "Pooled metrics combine out-of-fold predictions. They measure event ranking, not executable returns or CAGR. Latest calendar year is partial.",
    }
    (out / "tsmc_2330_selected_threshold_model_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--oracle", default=None, help="Retained for workflow compatibility; not used to tune labels.")
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.oracle, args.output), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
