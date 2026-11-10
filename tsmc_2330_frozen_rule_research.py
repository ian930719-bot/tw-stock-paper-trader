"""Nested chronological development of one TSMC 2330 swing rule per direction.

2010-2016: threshold estimation only.
2017-2021: choose one predeclared feature conjunction per direction by mean F1
across 1/3/5-session targets.
2022+: evaluate the frozen rule once as a descriptive historical holdout.
The 2022+ period was previously inspected, so it is NOT a pristine blind test.
No trades or CAGR are simulated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_START = "2010-01-01"
VALID_START = "2017-01-01"
TEST_START = "2022-01-01"
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


def target_for(dates: pd.Series, events: pd.DataFrame, horizon: int, kind: str) -> np.ndarray:
    chosen = events.loc[events["type"] == kind]
    lookup = {pd.Timestamp(v).normalize(): i for i, v in enumerate(dates)}
    y = np.zeros(len(dates), dtype=bool)
    for value in pd.to_datetime(chosen["pivot_date"], errors="coerce").dropna():
        idx = lookup.get(pd.Timestamp(value).normalize())
        if idx is not None:
            y[max(0, idx - horizon):idx] = True
    return y


def metrics(mask: np.ndarray, y: np.ndarray) -> dict:
    n = int(mask.sum())
    positives = int((mask & y).sum())
    total = int(y.sum())
    precision = positives / n if n else 0.0
    recall = positives / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    prevalence = float(y.mean()) if len(y) else 0.0
    return {
        "rows": int(len(y)), "signal_days": n, "true_positive_days": positives,
        "event_days": total, "prevalence": prevalence, "precision": precision,
        "recall": recall, "f1": f1,
        "lift": precision / prevalence if prevalence else None,
    }


def run(features_path: str, events_path: str, output_dir: str) -> dict:
    x = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    events = pd.read_csv(events_path, parse_dates=["pivot_date"])
    if not set(FEATURES).issubset(x.columns):
        raise ValueError(f"Missing features: {sorted(set(FEATURES) - set(x.columns))}")
    events = events.loc[events["threshold_id"] == THRESHOLD_ID].copy()
    if events.empty:
        raise ValueError(f"No pivot labels for {THRESHOLD_ID}")
    dates = x["date"]
    train_period = (dates >= TRAIN_START).to_numpy() & (dates < VALID_START).to_numpy()
    valid_period = (dates >= VALID_START).to_numpy() & (dates < TEST_START).to_numpy()
    test_period = (dates >= TEST_START).to_numpy()
    split_valid = np.flatnonzero(valid_period)
    split_test = np.flatnonzero(test_period)
    if not len(split_valid) or not len(split_test):
        raise ValueError("Chronological train/validation/test periods are incomplete")
    valid_start_idx, test_start_idx = int(split_valid[0]), int(split_test[0])
    valid_features = x[list(FEATURES)].replace([np.inf, -np.inf], np.nan)
    valid_rows = valid_features.notna().all(axis=1).to_numpy(copy=True)
    if "corporate_action_window" in x:
        valid_rows &= ~x["corporate_action_window"].fillna(True).astype(bool).to_numpy()

    all_rows = []
    selected_rules = {}
    for kind in ("HIGH", "LOW"):
        y_train_5 = target_for(dates, events, 5, kind)
        train_mask = train_period & valid_rows & (np.arange(len(x)) <= max(0, valid_start_idx - 6))
        positive_train = train_mask & y_train_5
        if int(positive_train.sum()) < 10:
            continue
        thresholds = {
            f: float(x.loc[positive_train, f].median()) for f in FEATURES
        }
        direction = 1 if kind == "HIGH" else -1
        conditions = {}
        for feature, cutoff in thresholds.items():
            if direction > 0:
                conditions[feature] = (x[feature] >= cutoff).fillna(False).to_numpy()
            else:
                conditions[feature] = (x[feature] <= cutoff).fillna(False).to_numpy()

        # Keep the last five validation sessions out of selection: their
        # labels could depend on pivot events occurring after TEST_START.
        validation_mask = valid_period & valid_rows & (np.arange(len(x)) <= max(0, test_start_idx - 6))
        test_mask = test_period & valid_rows
        candidates = []
        for combo in COMBINATIONS:
            signal = np.logical_and.reduce([conditions[f] for f in combo])
            validation_metrics = {}
            f1_values = []
            eligible = True
            for horizon in (1, 3, 5):
                y = target_for(dates, events, horizon, kind)
                m = metrics(signal & validation_mask, y[validation_mask])
                validation_metrics[str(horizon)] = m
                if m["signal_days"] < 8 or m["event_days"] < 5:
                    eligible = False
                f1_values.append(m["f1"])
            score = float(np.mean(f1_values)) if eligible else -1.0
            candidates.append({
                "features": list(combo),
                "validation_mean_f1": score,
                "validation_metrics": validation_metrics,
            })
        candidates.sort(key=lambda r: (r["validation_mean_f1"], r["features"]), reverse=True)
        chosen = candidates[0] if candidates and candidates[0]["validation_mean_f1"] >= 0 else None
        if chosen is None:
            selected_rules[kind] = {
                "selected": False, "reason": "No predeclared rule met minimum validation sample requirements",
                "training_thresholds": thresholds, "candidates": candidates,
            }
            continue
        frozen_combo = chosen["features"]
        frozen_signal = np.logical_and.reduce([conditions[f] for f in frozen_combo])
        test_metrics = {}
        for horizon in (1, 3, 5):
            y = target_for(dates, events, horizon, kind)
            test_metrics[str(horizon)] = metrics(frozen_signal & test_mask, y[test_mask])
        selected_rules[kind] = {
            "selected": True,
            "features": frozen_combo,
            "training_thresholds": thresholds,
            "selection_objective": "maximum mean F1 across 1/3/5-session validation targets; minimum 8 signal days and 5 event days for every horizon",
            "validation_mean_f1": chosen["validation_mean_f1"],
            "validation_metrics": chosen["validation_metrics"],
            "historical_holdout_metrics": test_metrics,
            "historical_holdout_is_pristine": False,
        }
        for horizon in (1, 3, 5):
            all_rows.append({
                "pivot_type": kind, "horizon_sessions": horizon,
                "selected_features": "+".join(frozen_combo),
                "validation_mean_f1": chosen["validation_mean_f1"],
                **{f"test_{k}": v for k, v in test_metrics[str(horizon)].items()},
                "test_start": dates.loc[test_mask].min().date().isoformat(),
                "test_end": dates.loc[test_mask].max().date().isoformat(),
                "exploratory_only": True,
            })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out / "tsmc_2330_frozen_rule_holdout.csv", index=False)
    summary = {
        "symbol": "2330", "threshold_id": THRESHOLD_ID,
        "training_period": ["2010-01-01", "2016-12-31"],
        "validation_period": ["2017-01-01", "2021-12-31"],
        "test_period": [dates.loc[test_period].min().date().isoformat(), dates.loc[test_period].max().date().isoformat()],
        "features": list(FEATURES),
        "candidate_conjunctions": [list(c) for c in COMBINATIONS],
        "purge_sessions": 5,
        "rules": selected_rules,
        "selection_used_test_period": False,
        "strategy_backtest": False,
        "warning": "The 2022+ period was inspected in prior diagnostics and is not a pristine blind holdout. These are event-ranking rules, not buy/sell orders. No execution, costs, slippage, dividends, capital allocation, or CAGR are simulated. Forward paper validation remains required.",
    }
    (out / "tsmc_2330_frozen_rule_holdout.json").write_text(
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
