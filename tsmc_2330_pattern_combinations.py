"""Predeclared TSMC 2330 swing-feature conjunction diagnostics.

Thresholds are learned only from the 2010-2021 training segment. The 2022+
segment is reported as an exploratory chronological holdout and must not be
used to select or retune a final signal because prior diagnostics have already
examined this period. This is not a trading strategy or return backtest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_DATE = "2022-01-01"
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


def build_target(dates: pd.Series, events: pd.DataFrame, horizon: int, kind: str) -> np.ndarray:
    selected = events.loc[events["type"] == kind]
    lookup = {pd.Timestamp(value).normalize(): i for i, value in enumerate(dates)}
    y = np.zeros(len(dates), dtype=bool)
    for value in pd.to_datetime(selected["pivot_date"], errors="coerce").dropna():
        idx = lookup.get(pd.Timestamp(value).normalize())
        if idx is not None:
            y[max(0, idx - horizon):idx] = True
    return y


def run(features_path: str, events_path: str, output_dir: str) -> dict:
    x = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    events = pd.read_csv(events_path, parse_dates=["pivot_date"])
    if not set(FEATURES).issubset(x.columns):
        raise ValueError(f"Missing features: {sorted(set(FEATURES) - set(x.columns))}")
    events = events.loc[events["threshold_id"] == THRESHOLD_ID].copy()
    if events.empty:
        raise ValueError(f"No events for predeclared threshold {THRESHOLD_ID}")
    train = (x["date"] < SPLIT_DATE).to_numpy()
    test = (x["date"] >= SPLIT_DATE).to_numpy()
    split_indices = np.flatnonzero(test)
    if len(split_indices) == 0:
        raise ValueError(f"No test rows at or after {SPLIT_DATE}")
    split_index = int(split_indices[0])
    valid = x[list(FEATURES)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).to_numpy(copy=True)
    if "corporate_action_window" in x:
        valid &= ~x["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    rows = []
    learned = {}
    for kind in ("HIGH", "LOW"):
        direction = 1 if kind == "HIGH" else -1
        for horizon in (1, 3, 5):
            y = build_target(x["date"], events, horizon, kind)
            # Purge pre-split rows whose forward pivot-label window overlaps test.
            train_end = max(0, split_index - horizon - 1)
            train_mask = train & valid & (np.arange(len(x)) <= train_end)
            test_mask = test & valid
            event_train = train_mask & y
            if int(event_train.sum()) < 10 or int((test_mask & y).sum()) < 5:
                continue
            thresholds = {}
            for feature in FEATURES:
                values = x.loc[event_train, feature].replace([np.inf, -np.inf], np.nan).dropna()
                thresholds[feature] = float(values.median())
            learned[f"{kind}_{horizon}"] = thresholds
            signals = {}
            for feature, threshold in thresholds.items():
                if direction > 0:
                    signals[feature] = (x[feature] >= threshold).fillna(False).to_numpy()
                else:
                    signals[feature] = (x[feature] <= threshold).fillna(False).to_numpy()
            prevalence = float(y[test_mask].mean())
            for combo in COMBINATIONS:
                signal = np.logical_and.reduce([signals[name] for name in combo]) & test_mask
                n_signal = int(signal.sum())
                if n_signal == 0:
                    continue
                true_positive = int((signal & y).sum())
                precision = true_positive / n_signal
                rows.append({
                    "threshold_id": THRESHOLD_ID,
                    "pivot_type": kind,
                    "horizon_sessions": horizon,
                    "features": "+".join(combo),
                    "training_event_thresholds": json.dumps(
                        {name: thresholds[name] for name in combo}, ensure_ascii=False, sort_keys=True
                    ),
                    "test_rows": int(test_mask.sum()),
                    "test_event_prevalence": prevalence,
                    "signal_days": n_signal,
                    "true_positive_days": true_positive,
                    "precision": precision,
                    "lift": precision / prevalence if prevalence else None,
                    "recall": true_positive / int((test_mask & y).sum()),
                    "exploratory_only": True,
                })
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / "tsmc_2330_pattern_combinations.csv", index=False)
    summary = {
        "symbol": "2330",
        "threshold_id": THRESHOLD_ID,
        "train_end": x.loc[train, "date"].max().date().isoformat(),
        "purge_sessions": [1, 3, 5],
        "test_start": SPLIT_DATE,
        "test_end": x.loc[test, "date"].max().date().isoformat(),
        "features": list(FEATURES),
        "predeclared_combinations": [list(c) for c in COMBINATIONS],
        "rows": int(len(result)),
        "learned_training_thresholds": learned,
        "selection_performed": False,
        "strategy_backtest": False,
        "warning": "The 2022+ period was already examined by prior diagnostics. These conjunction results are descriptive only and cannot be used to select a final strategy. No executions, transaction costs, slippage, dividends, or CAGR are simulated.",
    }
    (out / "tsmc_2330_pattern_combinations.json").write_text(
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
