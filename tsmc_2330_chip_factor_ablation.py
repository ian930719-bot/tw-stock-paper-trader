"""Annual expanding walk-forward ablation of TSMC 2330 technical/chip features.

Each fold trains through the prior calendar year and validates on the next year.
Pivot labels are fixed at 5% reversal / 2 ATR and only confirmed training events
enter fitting. Broker features are skipped unless the backfill is sufficiently
complete; unavailable broker data is never treated as zero.
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


def annual_fold_years(dates: pd.Series) -> list[int]:
    """Expanding-window folds: train through Y-1, validate on calendar year Y."""
    years = sorted(pd.to_datetime(dates).dt.year.dropna().unique().tolist())
    return [int(year) for year in years[1:]]


def _events_for_fold(events: pd.DataFrame, dates: pd.Series, start_idx: int,
                     end_idx: int, horizon: int, *, training: bool,
                     available_end_idx: int) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    lookup = {pd.Timestamp(value): i for i, value in enumerate(dates)}
    pivot_idx = pd.to_datetime(events["pivot_date"]).map(
        lambda value: lookup.get(pd.Timestamp(value), -1)
    ).to_numpy()
    confirm_idx = pd.to_datetime(events["confirmation_date"]).map(
        lambda value: lookup.get(pd.Timestamp(value), -1)
    ).to_numpy()
    # A signal at t predicts a pivot in t+1 ... t+horizon.
    mask = (pivot_idx >= start_idx + horizon) & (pivot_idx <= end_idx + horizon)
    if training:
        # Only labels fully confirmed by the training cutoff may enter the fit.
        mask &= (pivot_idx <= end_idx) & (confirm_idx >= 0) & (confirm_idx <= end_idx)
    else:
        # Validation outcomes may mature after the calendar-year boundary, but
        # must be confirmed by the end of the available dataset.
        mask &= (confirm_idx >= 0) & (confirm_idx <= available_end_idx)
    return events.loc[mask].copy()


def _score_predictions(truth: np.ndarray, probability: np.ndarray) -> dict:
    if not len(truth) or len(np.unique(truth)) < 2:
        return {"error": "evaluation segment lacks rows or both target classes",
                "rows": int(len(truth)), "event_days": int(np.sum(truth))}
    prevalence = float(truth.mean())
    n = max(1, int(np.ceil(len(probability) * 0.10)))
    top = np.argsort(probability)[-n:]
    precision = float(truth[top].mean())
    return {
        "rows": int(len(truth)), "event_days": int(truth.sum()),
        "prevalence": prevalence, "roc_auc": float(roc_auc_score(truth, probability)),
        "pr_auc": float(average_precision_score(truth, probability)),
        "top_decile_precision": precision,
        "top_decile_lift": precision / prevalence if prevalence else None,
        "top_decile_recall": float(truth[top].sum() / truth.sum()) if truth.sum() else None,
    }


def run(features_path: str, oracle_path: str, output_dir: str,
        chip_manifest_path: str | None = None) -> dict:
    features = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    dates = features["date"]
    # Pivot labels are predeclared and fixed across all folds; no future-year
    # optimization of the target definition is allowed.
    min_reversal, atr_multiple = 0.05, 2.0
    events = detect_confirmed_pivots(features, min_reversal, atr_multiple)
    if events.empty:
        raise ValueError("No pivot events for the predeclared p0.050_atr2.0 target")
    idx = np.arange(len(features))
    available_end_idx = len(features) - 1
    safe = np.ones(len(features), dtype=bool)
    if "corporate_action_window" in features:
        safe &= ~features["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    years = annual_fold_years(dates)
    if not years:
        raise ValueError("At least two calendar years are required for annual walk-forward validation")
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
        X_all = features[cols].replace([np.inf, -np.inf], np.nan)
        group_rows = []
        for kind in ("LOW", "HIGH"):
            for horizon in HORIZONS:
                fold_rows = []
                pooled_truth, pooled_probability = [], []
                for year in years:
                    val_start_idx = int(np.flatnonzero((dates.dt.year == year).to_numpy())[0])
                    year_indices = np.flatnonzero((dates.dt.year == year).to_numpy())
                    val_end_calendar_idx = int(year_indices[-1])
                    # A partial latest year is evaluated only where the complete
                    # forward horizon is observable in the downloaded history.
                    val_end_idx = min(val_end_calendar_idx, available_end_idx - horizon)
                    train_end_idx = val_start_idx - 1
                    if val_end_idx < val_start_idx:
                        continue
                    train_events = _events_for_fold(
                        events, dates, 0, train_end_idx, horizon,
                        training=True, available_end_idx=available_end_idx,
                    )
                    val_events = _events_for_fold(
                        events, dates, val_start_idx, val_end_idx, horizon,
                        training=False, available_end_idx=available_end_idx,
                    )
                    y_train = build_target(dates, train_events, horizon, kind)
                    y_val = build_target(dates, val_events, horizon, kind)
                    train_mask = (idx <= train_end_idx - horizon) & safe
                    train_mask &= (dates.dt.year >= int(dates.dt.year.min())).to_numpy()
                    val_mask = (idx >= val_start_idx) & (idx <= val_end_idx) & safe
                    train_cols = [
                        col for col in cols
                        if X_all.loc[train_mask, col].notna().any()
                    ]
                    if not train_cols:
                        fold_rows.append({
                            "validation_year": year, "status": "skipped_no_trainable_features",
                            "training_end": dates.iloc[train_end_idx].date().isoformat(),
                        })
                        continue
                    if not train_mask.any() or not val_mask.any() or len(np.unique(y_train[train_mask])) < 2:
                        fold_rows.append({
                            "validation_year": year, "status": "skipped_insufficient_training_classes",
                            "training_rows": int(train_mask.sum()),
                            "training_events": int(y_train[train_mask].sum()),
                            "validation_rows": int(val_mask.sum()),
                            "validation_events": int(y_val[val_mask].sum()),
                        })
                        continue
                    model = make_model()
                    model.fit(X_all.loc[train_mask, train_cols], y_train[train_mask])
                    probability = model.predict_proba(X_all.loc[val_mask, train_cols])[:, 1]
                    truth = y_val[val_mask]
                    metrics = _score_predictions(truth, probability)
                    fold_rows.append({
                        "validation_year": year,
                        "training_start": dates.loc[train_mask].min().date().isoformat(),
                        "training_end": dates.loc[train_mask].max().date().isoformat(),
                        "validation_start": dates.loc[val_mask].min().date().isoformat(),
                        "validation_end": dates.loc[val_mask].max().date().isoformat(),
                        "partial_validation_year": year == int(dates.max().year),
                        "training_rows": int(train_mask.sum()),
                        "training_events": int(y_train[train_mask].sum()),
                        "validation": metrics,
                        "feature_count": len(train_cols),
                        "features": train_cols,
                    })
                    if "error" not in metrics:
                        pooled_truth.append(truth)
                        pooled_probability.append(probability)
                pooled = (
                    _score_predictions(np.concatenate(pooled_truth), np.concatenate(pooled_probability))
                    if pooled_truth else {"error": "no valid annual folds"}
                )
                group_rows.append({
                    "pivot_type": kind,
                    "horizon_sessions": horizon,
                    "annual_folds": fold_rows,
                    "pooled_walk_forward": pooled,
                })
        results[group_name] = {
            "status": "evaluated",
            "walk_forward_years": years,
            "horizons": group_rows,
        }

    summary = {
        "symbol": "2330",
        "pivot_threshold": {
            "min_reversal_pct": min_reversal, "atr_multiple": atr_multiple,
            "selection": "predeclared fixed target definition; not tuned on any validation year",
        },
        "validation_method": "expanding-window annual walk-forward: train on 2010..Y-1, validate on year Y, then expand training window by one year",
        "first_training_year": int(dates.dt.year.min()),
        "validation_years": years,
        "model": "fixed logistic regression; median imputation; standard scaling; balanced class weights; C=0.2; preprocessing fit separately inside each fold",
        "target": "HIGH and LOW pivots within next 1/3/5 sessions; signal features known at close; hypothetical execution no earlier than next session",
        "purging": "last horizon sessions removed from each training fold; validation labels may mature after year-end but must be confirmed by latest available data; latest partial year excludes rows without a complete forward horizon",
        "results": results,
        "test_period_is_pristine": False,
        "note": "Annual walk-forward feature-ablation diagnostic, not a trading-return backtest. Each year's model is trained only on prior years. Broker model is skipped unless broker history is complete and sufficiently large. 2026 is partial year-to-date.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_chip_factor_ablation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    flat = []
    for group, result in results.items():
        for row in result.get("horizons", []):
            pooled = row.get("pooled_walk_forward", {})
            for fold in row.get("annual_folds", []):
                val = fold.get("validation", {})
                flat.append({
                    "feature_group": group,
                    "pivot_type": row.get("pivot_type"),
                    "horizon_sessions": row.get("horizon_sessions"),
                    "validation_year": fold.get("validation_year"),
                    "training_end": fold.get("training_end"),
                    "validation_start": fold.get("validation_start"),
                    "validation_end": fold.get("validation_end"),
                    "partial_validation_year": fold.get("partial_validation_year"),
                    "validation_pr_auc": val.get("pr_auc"),
                    "validation_top_decile_lift": val.get("top_decile_lift"),
                    "validation_top_decile_precision": val.get("top_decile_precision"),
                    "fold_status": fold.get("status"),
                    "pooled_walk_forward_pr_auc": pooled.get("pr_auc"),
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
