"""Chronological OOS diagnostic for TSMC 2330 swing pivots.

This is a signal-ranking diagnostic, not a trading strategy or return backtest.
Pivot dates are retrospective labels. Predictors are close-known causal features.
A purge equal to the prediction horizon prevents labels crossing the train/test split.
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

THRESHOLDS = (
    "p0.030_atr2.0", "p0.030_atr3.0", "p0.050_atr2.0",
    "p0.050_atr3.0", "p0.080_atr2.0",
)
HORIZONS = (1, 3, 5)
SPLIT_DATE = "2022-01-01"
EXCLUDE = {
    "symbol", "date", "open", "high", "low", "close", "volume",
    "turnover", "corporate_action_window", "atr14",
}


def build_target(dates: pd.Series, events: pd.DataFrame, horizon: int, kind: str) -> np.ndarray:
    selected = events if kind == "ANY" else events.loc[events["type"] == kind]
    y = np.zeros(len(dates), dtype=np.int8)
    lookup = {pd.Timestamp(value): i for i, value in enumerate(dates)}
    for value in pd.to_datetime(selected["pivot_date"]).dropna():
        index = lookup.get(pd.Timestamp(value))
        if index is not None:
            y[max(0, index - horizon):index] = 1
    return y


def run(features_path: str, events_path: str, output_dir: str) -> dict:
    features = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    events = pd.read_csv(events_path, parse_dates=["pivot_date", "confirmation_date"])
    required = {"date", "return_1", "rsi14", "bb_position20", "atr_pct14"}
    if not required.issubset(features.columns):
        raise ValueError(f"Missing causal feature columns: {sorted(required - set(features.columns))}")
    if not {"threshold_id", "type", "pivot_date"}.issubset(events.columns):
        raise ValueError("Pivot event file is missing required label columns")
    feature_cols = [
        col for col in features.columns
        if col not in EXCLUDE and pd.api.types.is_numeric_dtype(features[col])
    ]
    if not feature_cols:
        raise ValueError("No numeric causal predictors available")
    X = features[feature_cols].replace([np.inf, -np.inf], np.nan)
    dates = features["date"]
    split = pd.Timestamp(SPLIT_DATE)
    split_indices = np.flatnonzero((dates >= split).to_numpy())
    if len(split_indices) == 0:
        raise ValueError(f"No test rows at or after {SPLIT_DATE}")
    split_index = int(split_indices[0])
    test_mask = (dates >= split).to_numpy() & X.notna().any(axis=1).to_numpy()
    output = []
    for threshold in THRESHOLDS:
        threshold_events = events.loc[events["threshold_id"] == threshold]
        if threshold_events.empty:
            continue
        for horizon in HORIZONS:
            # Purge the final horizon sessions before the split.
            train_end = max(0, split_index - horizon - 1)
            train_mask = (
                (np.arange(len(features)) <= train_end)
                & (dates < split).to_numpy()
                & X.notna().any(axis=1).to_numpy()
            )
            for kind in ("HIGH", "LOW"):
                y = build_target(dates, threshold_events, horizon, kind)
                if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
                    continue
                model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", C=0.2,
                        random_state=0,
                    ),
                )
                model.fit(X.loc[train_mask], y[train_mask])
                probability = model.predict_proba(X.loc[test_mask])[:, 1]
                truth = y[test_mask]
                top_n = max(1, int(np.ceil(len(probability) * 0.10)))
                top = np.argsort(probability)[-top_n:]
                prevalence = float(truth.mean())
                precision = float(truth[top].mean())
                output.append({
                    "threshold_id": threshold,
                    "horizon_sessions": horizon,
                    "pivot_type": kind,
                    "train_start": dates.loc[train_mask].min().date().isoformat(),
                    "train_end": dates.loc[train_mask].max().date().isoformat(),
                    "test_start": dates.loc[test_mask].min().date().isoformat(),
                    "test_end": dates.loc[test_mask].max().date().isoformat(),
                    "train_rows": int(train_mask.sum()),
                    "test_rows": int(test_mask.sum()),
                    "feature_count": len(feature_cols),
                    "test_event_prevalence": prevalence,
                    "roc_auc": float(roc_auc_score(truth, probability)),
                    "pr_auc": float(average_precision_score(truth, probability)),
                    "top_decile_precision": precision,
                    "top_decile_lift": precision / prevalence if prevalence else None,
                    "top_decile_recall": float(truth[top].sum() / truth.sum()) if truth.sum() else None,
                    "model": "standardized logistic regression; median imputation; balanced classes; C=0.2",
                    "strategy_backtest": False,
                })
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(output)
    result.to_csv(out / "tsmc_2330_oos_signal_diagnostics.csv", index=False)
    summary = {
        "symbol": "2330",
        "split_date": SPLIT_DATE,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "thresholds": list(THRESHOLDS),
        "horizons": list(HORIZONS),
        "results": int(len(result)),
        "test_rows": int(test_mask.sum()),
        "strategy_selected": False,
        "strategy_backtest": False,
        "note": "Exploratory chronological OOS ranking diagnostics only. No trades, execution prices, transaction costs, dividends, slippage, or CAGR are simulated. Thresholds were predeclared but comparative results are exploratory and require a fresh untouched validation segment before model selection.",
    }
    (out / "tsmc_2330_oos_signal_diagnostics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.events, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
