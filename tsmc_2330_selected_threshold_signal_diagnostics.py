"""Causal signal diagnostics for the pivot definition selected on training-only oracle data.

The pivot threshold is read from the oracle grid's 2010-2016-only selection.
A logistic model is fit on 2010-2016 and evaluated on 2017-2021, then refit on
2010-2021 and evaluated on 2022+. The 2022+ period was previously inspected,
so this remains descriptive rather than a pristine blind test.
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

from tsmc_swing_research import detect_confirmed_pivots

HORIZONS = (1, 3, 5)
VALIDATION_START = "2017-01-01"
TEST_START = "2022-01-01"
EXCLUDE = {
    "symbol", "date", "open", "high", "low", "close", "volume",
    "turnover", "corporate_action_window", "atr14",
}


def period_events(events: pd.DataFrame, dates: pd.Series, start: str,
                  end: str, horizon: int) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(start))
    end_idx = max(i for i, d in enumerate(dates) if d <= pd.Timestamp(end))
    piv = pd.to_datetime(events["pivot_date"])
    conf = pd.to_datetime(events["confirmation_date"])
    pivot_indices = piv.map(lambda d: idx.get(pd.Timestamp(d), -1))
    mask = (
        (pivot_indices >= start_idx + horizon)
        & (pivot_indices <= end_idx)
        & (conf >= pd.Timestamp(start))
        & (conf <= pd.Timestamp(end))
    )
    return events.loc[mask].copy()


def build_target(dates: pd.Series, events: pd.DataFrame, horizon: int,
                 kind: str) -> np.ndarray:
    chosen = events.loc[events["type"] == kind]
    lookup = {pd.Timestamp(value): i for i, value in enumerate(dates)}
    y = np.zeros(len(dates), dtype=np.int8)
    for value in pd.to_datetime(chosen["pivot_date"]).dropna():
        index = lookup.get(pd.Timestamp(value))
        if index is not None:
            y[index - horizon:index] = 1
    return y


def evaluate(model, X: pd.DataFrame, mask: np.ndarray, y: np.ndarray) -> dict:
    probability = model.predict_proba(X.loc[mask])[:, 1]
    truth = y[mask]
    top_n = max(1, int(np.ceil(len(probability) * 0.10)))
    top = np.argsort(probability)[-top_n:]
    prevalence = float(truth.mean())
    precision = float(truth[top].mean())
    return {
        "rows": int(mask.sum()),
        "event_prevalence": prevalence,
        "roc_auc": float(roc_auc_score(truth, probability)),
        "pr_auc": float(average_precision_score(truth, probability)),
        "top_decile_precision": precision,
        "top_decile_lift": precision / prevalence if prevalence else None,
        "top_decile_recall": float(truth[top].sum() / truth.sum()) if truth.sum() else None,
    }


def run(features_path: str, oracle_path: str, output_dir: str) -> dict:
    features = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    oracle = json.loads(Path(oracle_path).read_text(encoding="utf-8"))
    selected = oracle.get("threshold_selected_using_training_period_only", {})
    min_reversal = selected.get("min_reversal_pct")
    atr_multiple = selected.get("atr_multiple")
    if min_reversal is None or atr_multiple is None:
        raise ValueError("Oracle grid did not select a valid training-only pivot threshold")
    threshold_id = f"selected_train_p{float(min_reversal):.3f}_atr{float(atr_multiple):.2f}"
    events = detect_confirmed_pivots(features, float(min_reversal), float(atr_multiple))
    if events.empty:
        raise ValueError("Selected training-only pivot threshold produced no events")
    events.insert(0, "threshold_id", threshold_id)

    feature_cols = [
        col for col in features.columns
        if col not in EXCLUDE and not col.startswith("raw_")
        and pd.api.types.is_numeric_dtype(features[col])
    ]
    X = features[feature_cols].replace([np.inf, -np.inf], np.nan)
    dates = features["date"]
    valid_rows = X.notna().any(axis=1).to_numpy(copy=True)
    if "corporate_action_window" in features:
        valid_rows &= ~features["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    train_end = "2016-12-31"
    validation_end = "2021-12-31"
    validation_mask_base = ((dates >= VALIDATION_START) & (dates <= validation_end)).to_numpy()
    test_mask = (dates >= TEST_START).to_numpy() & valid_rows
    validation_start_idx = int(np.flatnonzero((dates >= pd.Timestamp(VALIDATION_START)).to_numpy())[0])
    test_start_idx = int(np.flatnonzero((dates >= pd.Timestamp(TEST_START)).to_numpy())[0])
    train_end_idx = int(np.flatnonzero((dates <= pd.Timestamp(train_end)).to_numpy())[-1])
    rows = []
    for horizon in HORIZONS:
        train_events = period_events(events, dates, "2010-01-01", train_end, horizon)
        validation_events = period_events(events, dates, VALIDATION_START, validation_end, horizon)
        test_events = period_events(events, dates, TEST_START, dates.max().date().isoformat(), horizon)
        train_y_by_kind = {}
        validation_y_by_kind = {}
        test_y_by_kind = {}
        train_mask = (np.arange(len(features)) <= train_end_idx - horizon).astype(bool) & valid_rows
        validation_mask = validation_mask_base & (np.arange(len(features)) <= test_start_idx - horizon - 1) & valid_rows
        for kind in ("HIGH", "LOW"):
            y_train = build_target(dates, train_events, horizon, kind)
            y_val = build_target(dates, validation_events, horizon, kind)
            y_test = build_target(dates, test_events, horizon, kind)
            train_mask_kind = train_mask
            val_mask_kind = validation_mask
            test_mask_kind = test_mask
            if len(np.unique(y_train[train_mask_kind])) < 2:
                continue
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", C=0.2, random_state=0
                ),
            )
            model.fit(X.loc[train_mask_kind], y_train[train_mask_kind])
            if len(np.unique(y_val[val_mask_kind])) == 2:
                val_metrics = evaluate(model, X, val_mask_kind, y_val)
            else:
                val_metrics = {"error": "validation target has one class"}
            # Refit using all labels confirmed by 2021, then evaluate the 2022+ period.
            train_val_events = pd.concat([train_events, validation_events], ignore_index=True)
            y_train_val = build_target(dates, train_val_events, horizon, kind)
            train_val_mask = (np.arange(len(features)) <= test_start_idx - horizon - 1).astype(bool) & valid_rows
            if len(np.unique(y_train_val[train_val_mask])) < 2 or len(np.unique(y_test[test_mask_kind])) < 2:
                test_metrics = {"error": "train/test target has one class"}
            else:
                final_model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", C=0.2, random_state=0
                    ),
                )
                final_model.fit(X.loc[train_val_mask], y_train_val[train_val_mask])
                test_metrics = evaluate(final_model, X, test_mask_kind, y_test)
            rows.append({
                "threshold_id": threshold_id, "min_reversal_pct": min_reversal,
                "atr_multiple": atr_multiple, "horizon_sessions": horizon,
                "pivot_type": kind,
                "train_rows": int(train_mask.sum()),
                "train_event_days": int(y_train[train_mask].sum()),
                "validation_rows": int(validation_mask.sum()),
                "validation_event_days": int(y_val[validation_mask].sum()),
                "test_rows": int(test_mask.sum()),
                "test_event_days": int(y_test[test_mask].sum()),
                "feature_count": len(feature_cols),
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
            })
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    flat = []
    for row in rows:
        flat.append({
            "threshold_id": row["threshold_id"],
            "min_reversal_pct": row["min_reversal_pct"],
            "atr_multiple": row["atr_multiple"],
            "horizon_sessions": row["horizon_sessions"],
            "pivot_type": row["pivot_type"],
            "train_rows": row["train_rows"],
            "train_event_days": row["train_event_days"],
            "validation_rows": row["validation_rows"],
            "validation_event_days": row["validation_event_days"],
            "validation_auc": row["validation_metrics"].get("roc_auc"),
            "validation_pr_auc": row["validation_metrics"].get("pr_auc"),
            "validation_top_decile_precision": row["validation_metrics"].get("top_decile_precision"),
            "validation_top_decile_lift": row["validation_metrics"].get("top_decile_lift"),
            "validation_top_decile_recall": row["validation_metrics"].get("top_decile_recall"),
            "test_rows": row["test_rows"],
            "test_event_days": row["test_event_days"],
            "test_auc": row["test_metrics"].get("roc_auc"),
            "test_pr_auc": row["test_metrics"].get("pr_auc"),
            "test_top_decile_precision": row["test_metrics"].get("top_decile_precision"),
            "test_top_decile_lift": row["test_metrics"].get("top_decile_lift"),
            "test_top_decile_recall": row["test_metrics"].get("top_decile_recall"),
        })
    pd.DataFrame(flat).to_csv(out / "tsmc_2330_selected_threshold_signal_diagnostics.csv", index=False)
    summary = {
        "symbol": "2330", "selected_threshold_id": threshold_id,
        "min_reversal_pct": min_reversal, "atr_multiple": atr_multiple,
        "train_period": ["2010-01-01", train_end],
        "validation_period": [VALIDATION_START, validation_end],
        "test_period": [TEST_START, dates.max().date().isoformat()],
        "feature_count": len(feature_cols), "feature_columns": feature_cols,
        "horizons": list(HORIZONS), "results": rows,
        "test_period_used_for_threshold_selection": False,
        "test_period_is_pristine": False,
        "strategy_backtest": False,
        "warning": "Pivot threshold selected using perfect-foresight oracle metric on 2010-2016 only. Logistic model uses causal features. 2022+ was previously inspected and is not a pristine blind holdout; no trades or CAGR are simulated.",
    }
    (out / "tsmc_2330_selected_threshold_signal_diagnostics.json").write_text(
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
