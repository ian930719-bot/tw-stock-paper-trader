"""Annual expanding walk-forward diagnostics for TSMC 2330 causal pivot signals.

Each validation year Y is scored by a model trained only on 2010..Y-1.
The pivot-label definition is fixed at 5% reversal and 2 ATR before folds;
it is not re-optimized against validation years. This is signal prediction,
not a trading-return backtest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tsmc_swing_research import detect_confirmed_pivots
from tsmc_2330_walk_forward import (
    annual_fold_years, event_target, fold_events, fold_masks, score_predictions,
)

HORIZONS = (1, 3, 5)
MIN_REVERSAL_PCT = 0.05
ATR_MULTIPLE = 2.0
EXCLUDE = {
    "obv", "symbol", "date", "open", "high", "low", "close", "volume",
    "turnover", "corporate_action_window", "atr14",
}


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=0.2, random_state=0),
    )


def run(features_path: str, oracle_path: str | None, output_dir: str) -> dict:
    x = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    events = detect_confirmed_pivots(x, MIN_REVERSAL_PCT, ATR_MULTIPLE)
    if events.empty:
        raise ValueError("Fixed pivot definition produced no events")
    events = events.copy()
    events.insert(0, "threshold_id", "p0.050_atr2.0")
    feature_cols = [
        c for c in x.columns
        if c not in EXCLUDE and not c.startswith("raw_") and pd.api.types.is_numeric_dtype(x[c])
    ]
    X = x[feature_cols].replace([np.inf, -np.inf], np.nan)
    valid = X.notna().any(axis=1).to_numpy(copy=True)
    if "corporate_action_window" in x:
        valid &= ~x["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    dates = x["date"]
    fold_rows, pooled = [], {}
    for horizon in HORIZONS:
        for kind in ("HIGH", "LOW"):
            key = f"{kind}_{horizon}"
            pooled[key] = {"truth": [], "probability": []}
            for year in annual_fold_years(dates):
                fold = fold_masks(dates, valid, year, horizon)
                if fold is None or not fold["validation_mask"].any():
                    continue
                train_events, val_events = fold_events(events, dates, fold, horizon)
                y_train = event_target(dates, train_events, horizon, kind)
                y_val = event_target(dates, val_events, horizon, kind)
                train_mask, val_mask = fold["train_mask"], fold["validation_mask"]
                if len(np.unique(y_train[train_mask])) < 2:
                    fold_rows.append({
                        "pivot_type": kind, "horizon_sessions": horizon,
                        "validation_year": year, "status": "skipped",
                        "reason": "training fold lacks both classes",
                        "training_rows": int(train_mask.sum()),
                        "training_event_days": int(y_train[train_mask].sum()),
                        "validation_rows": int(val_mask.sum()),
                    })
                    continue
                model = make_model()
                model.fit(X.loc[train_mask], y_train[train_mask])
                probability = model.predict_proba(X.loc[val_mask])[:, 1]
                truth = y_val[val_mask]
                metrics = score_predictions(truth, probability)
                pooled[key]["truth"].extend(truth.astype(int).tolist())
                pooled[key]["probability"].extend(probability.astype(float).tolist())
                fold_rows.append({
                    "pivot_type": kind, "horizon_sessions": horizon,
                    "validation_year": year, "status": "evaluated",
                    "training_start": dates.iloc[0].date().isoformat(),
                    "training_end": fold["training_end"],
                    "validation_start": fold["validation_start"],
                    "validation_end": fold["validation_end"],
                    "partial_validation_year": fold["partial_validation_year"],
                    "training_rows": int(train_mask.sum()),
                    "training_event_days": int(y_train[train_mask].sum()),
                    "validation_metrics": metrics,
                    "feature_count": len(feature_cols),
                })
    pooled_metrics = {
        key: score_predictions(np.asarray(v["truth"]), np.asarray(v["probability"]))
        for key, v in pooled.items()
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "pivot_type": r["pivot_type"], "horizon_sessions": r["horizon_sessions"],
            "validation_year": r["validation_year"], "status": r["status"],
            "training_rows": r.get("training_rows"), "training_event_days": r.get("training_event_days"),
            "validation_rows": r.get("validation_metrics", {}).get("rows", r.get("validation_rows")),
            "validation_event_days": r.get("validation_metrics", {}).get("event_days"),
            "validation_pr_auc": r.get("validation_metrics", {}).get("pr_auc"),
            "validation_roc_auc": r.get("validation_metrics", {}).get("roc_auc"),
            "validation_top_10_precision": r.get("validation_metrics", {}).get("top_10_precision"),
            "validation_top_10_lift": r.get("validation_metrics", {}).get("top_10_lift"),
        } for r in fold_rows
    ]).to_csv(out / "tsmc_2330_selected_threshold_signal_diagnostics.csv", index=False)
    summary = {
        "symbol": "2330",
        "pivot_threshold": {
            "min_reversal_pct": MIN_REVERSAL_PCT, "atr_multiple": ATR_MULTIPLE,
            "selection": "fixed label definition, not reselected per validation year",
        },
        "validation_method": "expanding-window annual walk-forward: train 2010..Y-1, validate Y",
        "first_training_year": int(pd.to_datetime(dates).dt.year.min()),
        "validation_years": annual_fold_years(dates),
        "model": "logistic regression C=0.2; median imputation and scaling fit inside each fold",
        "target": "HIGH and LOW pivots within next 1/3/5 sessions; close-known features; no same-day execution claim",
        "purging": "training rows exclude final horizon sessions; training events must be confirmed by training cutoff; validation rows require complete forward horizon",
        "feature_count": len(feature_cols), "feature_columns": feature_cols,
        "annual_folds": fold_rows, "pooled_walk_forward_metrics": pooled_metrics,
        "strategy_backtest": False,
        "warning": "Pooled metrics combine strictly out-of-fold annual predictions. They measure event ranking, not executable returns or CAGR. Latest calendar year is partial.",
    }
    (out / "tsmc_2330_selected_threshold_signal_diagnostics.json").write_text(
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
