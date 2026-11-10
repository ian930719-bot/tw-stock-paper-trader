"""Compare technical-only vs technical-plus-institutional features for TSMC 2330.

Uses the frozen training-selected pivot definition and a fixed logistic model.
2010-2016 is train, 2017-2021 validation, and 2022+ descriptive historical check.
Broker features are evaluated only when adequate broker history is present in
both train and validation; otherwise the broker model is explicitly skipped.
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
from tsmc_2330_selected_threshold_signal_diagnostics import period_events, build_target

EXCLUDE = {
    "obv",  # cumulative level is non-stationary; use normalized OBV flow ratios instead
    "symbol", "date", "open", "high", "low", "close", "volume",
    "turnover", "corporate_action_window", "atr14",
}
CHIP_PREFIXES = (
    "foreign_", "foreign_dealer_", "trust_", "dealer_",
    "institutional_", "broker_",
)
HORIZONS = (1, 3, 5)
TRAIN_START, TRAIN_END = "2010-01-01", "2016-12-31"
VALIDATION_START, VALIDATION_END = "2017-01-01", "2021-12-31"
TEST_START = "2022-01-01"


def feature_groups(features: pd.DataFrame) -> dict[str, list[str]]:
    """Build disjoint institutional-family ablations without double-counting aggregates."""
    numeric = [
        c for c in features.columns
        if c not in EXCLUDE and not c.startswith("raw_")
        and pd.api.types.is_numeric_dtype(features[c])
    ]
    technical = [c for c in numeric if not c.startswith(CHIP_PREFIXES)]
    institutional = [c for c in numeric if c.startswith(CHIP_PREFIXES) and not c.startswith("broker_")]
    broker = [c for c in numeric if c.startswith("broker_")]

    foreign = [c for c in numeric if c.startswith("foreign_") and not c.startswith("foreign_dealer_")]
    foreign_dealer = [c for c in numeric if c.startswith("foreign_dealer_")]
    trust = [c for c in numeric if c.startswith("trust_")]
    dealer = [c for c in numeric if c.startswith("dealer_")]
    institutional_total = [c for c in numeric if c.startswith("institutional_")]

    groups = {
        "technical_only": technical,
        "technical_plus_institutional": technical + institutional,
        "technical_plus_foreign_only": technical + foreign,
        "technical_plus_foreign_dealer_only": technical + foreign_dealer,
        "technical_plus_trust_only": technical + trust,
        "technical_plus_dealer_only": technical + dealer,
        "technical_plus_institutional_total_only": technical + institutional_total,
        "technical_plus_three_institution_families": technical + foreign + trust + dealer,
    }
    if broker:
        groups["technical_plus_all_chip"] = technical + institutional + broker
    return groups

def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=0.2, random_state=0),
    )


def evaluate(model, X: pd.DataFrame, mask: np.ndarray, y: np.ndarray) -> dict:
    if not mask.any() or len(np.unique(y[mask])) < 2:
        return {"error": "evaluation segment lacks rows or both target classes",
                "rows": int(mask.sum()), "event_days": int(y[mask].sum())}
    p = model.predict_proba(X.loc[mask])[:, 1]
    truth = y[mask]
    prevalence = float(truth.mean())
    n = max(1, int(np.ceil(len(p) * 0.10)))
    top = np.argsort(p)[-n:]
    precision = float(truth[top].mean())
    return {
        "rows": int(mask.sum()), "event_days": int(truth.sum()),
        "prevalence": prevalence, "roc_auc": float(roc_auc_score(truth, p)),
        "pr_auc": float(average_precision_score(truth, p)),
        "top_decile_precision": precision,
        "top_decile_lift": precision / prevalence if prevalence else None,
        "top_decile_recall": float(truth[top].sum() / truth.sum()) if truth.sum() else None,
    }


def run(features_path: str, oracle_path: str, output_dir: str,
        chip_manifest_path: str | None = None) -> dict:
    features = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    oracle = json.loads(Path(oracle_path).read_text(encoding="utf-8"))
    selected = oracle.get("threshold_selected_using_training_period_only", {})
    min_reversal, atr_multiple = selected.get("min_reversal_pct"), selected.get("atr_multiple")
    if min_reversal is None or atr_multiple is None:
        raise ValueError("Missing frozen training-selected pivot threshold")
    events = detect_confirmed_pivots(features, float(min_reversal), float(atr_multiple))
    if events.empty:
        raise ValueError("No pivot events for frozen threshold")
    dates = features["date"]
    idx = np.arange(len(features))
    train_events = period_events(events, dates, TRAIN_START, TRAIN_END, max(HORIZONS))
    validation_events = period_events(events, dates, VALIDATION_START, VALIDATION_END, max(HORIZONS))
    test_events = period_events(events, dates, TEST_START, dates.max().date().isoformat(), max(HORIZONS))
    train_mask_base = ((dates >= TRAIN_START) & (dates <= TRAIN_END)).to_numpy(copy=True)
    validation_mask = ((dates >= VALIDATION_START) & (dates <= VALIDATION_END)).to_numpy(copy=True)
    test_mask = (dates >= TEST_START).to_numpy(copy=True)
    if "corporate_action_window" in features:
        safe = ~features["corporate_action_window"].fillna(True).astype(bool).to_numpy()
        train_mask_base &= safe
        validation_mask &= safe
        test_mask &= safe
    groups = feature_groups(features)
    manifest = {}
    if chip_manifest_path:
        manifest = json.loads(Path(chip_manifest_path).read_text(encoding="utf-8"))
    broker_rows = int(manifest.get("broker_rows", manifest.get("broker_available_rows", manifest.get("broker_days_returned", 0))) or 0)
    broker_complete = bool(manifest.get("broker_backfill_complete", False))
    results = {}
    for group_name, cols in groups.items():
        is_broker = any(col.startswith("broker_") for col in cols)
        if is_broker and (not broker_complete or broker_rows < 500):
            results[group_name] = {
                "status": "skipped_insufficient_broker_history",
                "broker_rows": broker_rows, "broker_backfill_complete": broker_complete,
            }
            continue
        X = features[cols].replace([np.inf, -np.inf], np.nan)
        # A feature must have observations in training, otherwise it cannot be learned.
        train_cols = [col for col in cols if X.loc[train_mask_base, col].notna().any()]
        if not train_cols:
            results[group_name] = {"status": "skipped_no_trainable_features"}
            continue
        X = X[train_cols]
        rows = []
        for kind in ("LOW", "HIGH"):
            for horizon in HORIZONS:
                tr_events = period_events(events, dates, TRAIN_START, TRAIN_END, horizon)
                va_events = period_events(events, dates, VALIDATION_START, VALIDATION_END, horizon)
                te_events = period_events(events, dates, TEST_START, dates.max().date().isoformat(), horizon)
                y_train = build_target(dates, tr_events, horizon, kind)
                y_val = build_target(dates, va_events, horizon, kind)
                y_test = build_target(dates, te_events, horizon, kind)
                train_mask = train_mask_base & (idx >= np.flatnonzero((dates >= pd.Timestamp(TRAIN_START)).to_numpy())[0] + horizon)
                val_mask = validation_mask & (idx >= np.flatnonzero((dates >= pd.Timestamp(VALIDATION_START)).to_numpy())[0] + horizon)
                test_mask_h = test_mask & (idx >= np.flatnonzero((dates >= pd.Timestamp(TEST_START)).to_numpy())[0] + horizon)
                if len(np.unique(y_train[train_mask])) < 2 or len(np.unique(y_val[val_mask])) < 2:
                    rows.append({"pivot_type": kind, "horizon_sessions": horizon, "status": "skipped_one_class_segment"})
                    continue
                model = make_model()
                model.fit(X.loc[train_mask], y_train[train_mask])
                validation = evaluate(model, X, val_mask, y_val)
                # Refit through validation; the 2022+ result remains descriptive.
                train_val_mask = (train_mask_base | validation_mask) & (idx <= np.flatnonzero((dates >= pd.Timestamp(TEST_START)).to_numpy())[0] - horizon)
                y_train_val = np.maximum(y_train, y_val)
                final_model = make_model()
                if len(np.unique(y_train_val[train_val_mask])) == 2:
                    final_model.fit(X.loc[train_val_mask], y_train_val[train_val_mask])
                    test = evaluate(final_model, X, test_mask_h, y_test)
                else:
                    test = {"error": "train-plus-validation target has one class"}
                rows.append({
                    "pivot_type": kind,
                    "horizon_sessions": horizon,
                    "feature_count": len(train_cols),
                    "validation": validation,
                    "test_2022_plus_descriptive": test,
                })
        results[group_name] = {"status": "evaluated", "feature_count": len(train_cols), "features": train_cols, "horizons": rows}
    summary = {
        "symbol": "2330",
        "pivot_threshold": {"min_reversal_pct": float(min_reversal), "atr_multiple": float(atr_multiple)},
        "train_period": [TRAIN_START, TRAIN_END],
        "validation_period": [VALIDATION_START, VALIDATION_END],
        "test_period": [TEST_START, dates.max().date().isoformat()],
        "model": "fixed logistic regression; median imputation; standard scaling; balanced class weights; C=0.2",
        "target": "HIGH and LOW pivots within next 1/3/5 sessions; features available at signal-day close; assumed execution no earlier than next session",
        "results": results,
        "test_period_is_pristine": False,
        "test_period_used_for_selection": False,
        "note": "This is a feature-ablation diagnostic, not a trading-return backtest. Broker model is skipped unless historical broker coverage is complete and sufficiently large.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_chip_factor_ablation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    flat = []
    for group, result in results.items():
        for row in result.get("horizons", []):
            val, test = row.get("validation", {}), row.get("test_2022_plus_descriptive", {})
            flat.append({
                "feature_group": group, "pivot_type": row.get("pivot_type"), "horizon_sessions": row.get("horizon_sessions"),
                "feature_count": row.get("feature_count"),
                "validation_pr_auc": val.get("pr_auc"), "validation_top_decile_lift": val.get("top_decile_lift"),
                "test_pr_auc": test.get("pr_auc"), "test_top_decile_lift": test.get("top_decile_lift"),
                "test_status": test.get("error"),
            })
    pd.DataFrame(flat).to_csv(out / "tsmc_2330_chip_factor_ablation.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    parser.add_argument("--chip-manifest", default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.oracle, args.output, args.chip_manifest),
                     ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
