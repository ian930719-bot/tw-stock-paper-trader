"""Annual expanding walk-forward ranking diagnostics for TSMC 2330 swing pivots.

All pivot definitions are predeclared and reported separately. Each validation
year is scored by a model trained only on prior years with confirmed labels;
no fixed 2022+ holdout is used to choose a threshold or model. This is not a
trading strategy or return backtest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tsmc_2330_walk_forward import annual_fold_years, event_target, fold_events, fold_masks

THRESHOLDS = (
    "p0.030_atr2.0", "p0.030_atr3.0", "p0.050_atr2.0",
    "p0.050_atr3.0", "p0.080_atr2.0",
)
HORIZONS = (1, 3, 5)
EXCLUDE = {
    "obv", "symbol", "date", "open", "high", "low", "close", "volume",
    "turnover", "corporate_action_window", "atr14",
}


def score(truth: np.ndarray, probability: np.ndarray) -> dict:
    truth = np.asarray(truth, dtype=np.int8)
    probability = np.asarray(probability, dtype=float)
    if not len(truth):
        return {"rows": 0, "event_days": 0, "prevalence": None,
                "roc_auc": None, "pr_auc": None}
    prevalence = float(truth.mean())
    n = max(1, int(np.ceil(len(truth) * 0.10)))
    top = np.argsort(probability)[-n:]
    return {
        "rows": int(len(truth)), "event_days": int(truth.sum()),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(truth, probability))
        if len(np.unique(truth)) == 2 else None,
        "pr_auc": float(average_precision_score(truth, probability))
        if truth.sum() else 0.0,
        "top_decile_precision": float(truth[top].mean()),
        "top_decile_lift": float(truth[top].mean() / prevalence) if prevalence else None,
        "top_decile_recall": float(truth[top].sum() / truth.sum()) if truth.sum() else None,
    }


def run(features_path: str, events_path: str, output_dir: str) -> dict:
    features = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    events = pd.read_csv(events_path, parse_dates=["pivot_date", "confirmation_date"])
    required = {"date", "return_1", "rsi14", "bb_position20", "atr_pct14"}
    if not required.issubset(features.columns):
        raise ValueError(f"Missing causal feature columns: {sorted(required - set(features.columns))}")
    if not {"threshold_id", "type", "pivot_date", "confirmation_date"}.issubset(events.columns):
        raise ValueError("Pivot event file is missing required label columns")

    feature_cols = [
        col for col in features.columns
        if col not in EXCLUDE and not col.startswith("raw_")
        and pd.api.types.is_numeric_dtype(features[col])
    ]
    if not feature_cols:
        raise ValueError("No numeric causal predictors available")
    X = features[feature_cols].replace([np.inf, -np.inf], np.nan)
    dates = features["date"]
    valid_rows = X.notna().any(axis=1).to_numpy(copy=True)
    if "corporate_action_window" in features:
        valid_rows &= ~features["corporate_action_window"].fillna(True).astype(bool).to_numpy()

    annual_rows = []
    pooled_rows = []
    years = annual_fold_years(dates)
    for threshold in THRESHOLDS:
        threshold_events = events.loc[events["threshold_id"] == threshold].copy()
        if threshold_events.empty:
            continue
        for horizon in HORIZONS:
            pooled = {(kind): {"truth": [], "probability": []} for kind in ("HIGH", "LOW")}
            for year in years:
                fold = fold_masks(dates, valid_rows, year, horizon)
                if fold is None or not fold["validation_mask"].any():
                    continue
                train_events, val_events = fold_events(threshold_events, dates, fold, horizon)
                train_mask, val_mask = fold["train_mask"], fold["validation_mask"]
                if not train_mask.any() or not val_mask.any():
                    continue
                for kind in ("HIGH", "LOW"):
                    y_train = event_target(dates, train_events, horizon, kind)
                    y_val = event_target(dates, val_events, horizon, kind)
                    train_cols = [
                        col for col in feature_cols
                        if X.loc[train_mask, col].notna().any()
                    ]
                    if not train_cols or len(np.unique(y_train[train_mask])) < 2:
                        annual_rows.append({
                            "threshold_id": threshold, "horizon_sessions": horizon,
                            "pivot_type": kind, "validation_year": year,
                            "status": "skipped_insufficient_training_data",
                            "training_rows": int(train_mask.sum()),
                            "training_events": int(y_train[train_mask].sum()),
                        })
                        continue
                    model = make_pipeline(
                        SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced",
                                           C=0.2, random_state=0),
                    )
                    model.fit(X.loc[train_mask, train_cols], y_train[train_mask])
                    probability = model.predict_proba(X.loc[val_mask, train_cols])[:, 1]
                    truth = y_val[val_mask]
                    metrics = score(truth, probability)
                    annual_rows.append({
                        "threshold_id": threshold, "horizon_sessions": horizon,
                        "pivot_type": kind, "validation_year": year,
                        "status": "evaluated",
                        "training_start": dates.loc[train_mask].min().date().isoformat(),
                        "training_end": dates.loc[train_mask].max().date().isoformat(),
                        "validation_start": dates.loc[val_mask].min().date().isoformat(),
                        "validation_end": dates.loc[val_mask].max().date().isoformat(),
                        "partial_validation_year": year == int(dates.max().year),
                        "training_rows": int(train_mask.sum()),
                        "training_events": int(y_train[train_mask].sum()),
                        "feature_count": len(train_cols), "features": train_cols,
                        **metrics,
                    })
                    pooled[kind]["truth"].extend(truth.tolist())
                    pooled[kind]["probability"].extend(probability.tolist())
            for kind in ("HIGH", "LOW"):
                truth = np.asarray(pooled[kind]["truth"], dtype=np.int8)
                probability = np.asarray(pooled[kind]["probability"], dtype=float)
                pooled_rows.append({
                    "threshold_id": threshold, "horizon_sessions": horizon,
                    "pivot_type": kind, **score(truth, probability),
                })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(annual_rows).to_csv(out / "tsmc_2330_oos_signal_diagnostics.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(out / "tsmc_2330_oos_signal_diagnostics_pooled.csv", index=False)
    summary = {
        "symbol": "2330", "validation_method": "expanding-window annual walk-forward: train through Y-1, validate Y",
        "validation_years": years, "feature_count_max": len(feature_cols),
        "candidate_threshold_definitions": list(THRESHOLDS),
        "horizons": list(HORIZONS), "annual_results": annual_rows,
        "pooled_out_of_fold_results": pooled_rows,
        "threshold_selection_performed": False, "model_selection_performed": False,
        "strategy_backtest": False, "pristine_holdout": False,
        "note": "All predeclared pivot thresholds are reported independently on pooled annual OOF predictions. No threshold winner is selected on validation data. Metrics rank close-known signals for later pivot events; they do not represent trades, costs, or CAGR. Latest calendar year is partial.",
    }
    (out / "tsmc_2330_oos_signal_diagnostics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.events, args.output),
                     ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
