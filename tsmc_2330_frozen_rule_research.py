"""Leakage-aware annual walk-forward diagnostics for predeclared TSMC 2330 swing rules.

All feature thresholds are learned only from confirmed training events in each
fold. Every predeclared conjunction is reported; no validation-year winner is
selected or presented as a frozen trading rule. These are event-ranking
diagnostics, not executable trading returns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_walk_forward import annual_fold_years, event_target, fold_events, fold_masks

THRESHOLD_ID = "p0.050_atr2.0"
FEATURES = ("return_3", "distance_sma_20", "rsi14", "bb_position20", "close_location")
COMBINATIONS = (
    ("return_3", "bb_position20"),
    ("bb_position20", "close_location"),
    ("return_3", "close_location"),
    ("return_3", "bb_position20", "close_location"),
    ("return_3", "distance_sma_20", "close_location"),
    ("return_3", "rsi14", "close_location"),
)


def metrics(signal: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    s = np.asarray(signal, dtype=bool) & mask
    truth = np.asarray(y, dtype=bool) & mask
    n, positives = int(s.sum()), int(truth.sum())
    tp = int((s & truth).sum())
    precision = tp / n if n else None
    recall = tp / positives if positives else None
    prevalence = positives / int(mask.sum()) if mask.sum() else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else 0.0)
    return {
        "rows": int(mask.sum()), "signal_days": n, "event_days": positives,
        "true_positive_days": tp, "prevalence": prevalence,
        "precision": precision, "recall": recall, "f1": f1,
        "lift": precision / prevalence if precision is not None and prevalence else None,
    }


def run(features_path: str, events_path: str, output_dir: str) -> dict:
    x = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    events = pd.read_csv(events_path, parse_dates=["pivot_date", "confirmation_date"])
    events = events.loc[events["threshold_id"] == THRESHOLD_ID].copy()
    if events.empty:
        raise ValueError(f"No events for fixed target {THRESHOLD_ID}")
    missing = sorted(set(FEATURES) - set(x.columns))
    if missing:
        raise ValueError(f"Missing features: {missing}")
    values = x[list(FEATURES)].replace([np.inf, -np.inf], np.nan)
    valid = values.notna().all(axis=1).to_numpy(copy=True)
    if "corporate_action_window" in x:
        valid &= ~x["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    dates = x["date"]
    rows = []
    pooled = {}
    years = annual_fold_years(dates)

    for kind in ("HIGH", "LOW"):
        direction = 1 if kind == "HIGH" else -1
        for horizon in (1, 3, 5):
            for combo in COMBINATIONS:
                pooled[(kind, horizon, combo)] = {"signal": [], "truth": []}
            for year in years:
                fold = fold_masks(dates, valid, year, horizon)
                if fold is None or not fold["validation_mask"].any():
                    continue
                train_events, val_events = fold_events(events, dates, fold, horizon)
                y_train = event_target(dates, train_events, horizon, kind).astype(bool)
                y_val = event_target(dates, val_events, horizon, kind).astype(bool)
                train_mask, val_mask = fold["train_mask"], fold["validation_mask"]
                positive_train = train_mask & y_train
                if int(positive_train.sum()) < 3:
                    rows.append({
                        "pivot_type": kind, "horizon_sessions": horizon,
                        "validation_year": year, "status": "skipped",
                        "reason": "fewer than 3 confirmed training event rows",
                    })
                    continue
                thresholds = {f: float(values.loc[positive_train, f].median()) for f in FEATURES}
                conditions = {
                    f: ((x[f] >= thresholds[f]) if direction > 0 else (x[f] <= thresholds[f]))
                    .fillna(False).to_numpy()
                    for f in FEATURES
                }
                for combo in COMBINATIONS:
                    signal = np.logical_and.reduce([conditions[f] for f in combo])
                    m = metrics(signal, y_val, val_mask)
                    pooled[(kind, horizon, combo)]["signal"].extend(signal[val_mask].astype(int).tolist())
                    pooled[(kind, horizon, combo)]["truth"].extend(y_val[val_mask].astype(int).tolist())
                    rows.append({
                        "pivot_type": kind, "horizon_sessions": horizon,
                        "validation_year": year, "status": "evaluated",
                        "training_end": fold["training_end"],
                        "validation_start": fold["validation_start"],
                        "validation_end": fold["validation_end"],
                        "partial_validation_year": fold["partial_validation_year"],
                        "features": "+".join(combo),
                        "training_event_thresholds": {f: thresholds[f] for f in combo},
                        **m,
                    })

    pooled_rows = []
    for (kind, horizon, combo), d in pooled.items():
        signal = np.asarray(d["signal"], dtype=bool)
        truth = np.asarray(d["truth"], dtype=bool)
        n, positives = int(signal.sum()), int(truth.sum())
        tp = int((signal & truth).sum())
        precision = tp / n if n else None
        recall = tp / positives if positives else None
        prevalence = positives / len(truth) if len(truth) else None
        pooled_rows.append({
            "pivot_type": kind, "horizon_sessions": horizon,
            "features": "+".join(combo), "validation_rows": len(truth),
            "signal_days": n, "event_days": positives, "true_positive_days": tp,
            "precision": precision, "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)
                  if precision is not None and recall is not None and precision + recall else 0.0),
            "prevalence": prevalence,
            "lift": precision / prevalence if precision is not None and prevalence else None,
        })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "tsmc_2330_frozen_rule_holdout.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(out / "tsmc_2330_frozen_rule_pooled.csv", index=False)
    summary = {
        "symbol": "2330", "threshold_id": THRESHOLD_ID,
        "validation_method": "expanding-window annual walk-forward: train 2010..Y-1, validate Y",
        "validation_years": years, "features": list(FEATURES),
        "predeclared_combinations": [list(c) for c in COMBINATIONS],
        "annual_results": rows, "pooled_out_of_fold_results": pooled_rows,
        "selection_performed": False, "strategy_backtest": False,
        "test_period_is_pristine": False,
        "warning": "Legacy filename retained for workflow compatibility. No fixed 2010-2016/2017-2021/2022+ split remains. Thresholds are fit inside each training fold; all combinations are reported without selecting on validation. Metrics are event-day classification diagnostics, not executable returns or CAGR.",
    }
    (out / "tsmc_2330_frozen_rule_holdout.json").write_text(
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
