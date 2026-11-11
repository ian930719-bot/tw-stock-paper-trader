"""Shared leakage-aware annual expanding walk-forward utilities for TSMC 2330."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def annual_fold_years(dates: pd.Series) -> list[int]:
    years = sorted(pd.to_datetime(dates).dt.year.dropna().astype(int).unique().tolist())
    return years[1:]


def event_target(dates: pd.Series, events: pd.DataFrame, horizon: int, kind: str) -> np.ndarray:
    lookup = {pd.Timestamp(v).normalize(): i for i, v in enumerate(dates)}
    y = np.zeros(len(dates), dtype=np.int8)
    if events.empty:
        return y
    chosen = events.loc[events["type"] == kind]
    for value in pd.to_datetime(chosen["pivot_date"], errors="coerce").dropna():
        idx = lookup.get(pd.Timestamp(value).normalize())
        if idx is not None and idx >= horizon:
            y[idx - horizon:idx] = 1
    return y


def fold_masks(dates: pd.Series, valid_rows: np.ndarray, year: int, horizon: int):
    dates = pd.to_datetime(dates).reset_index(drop=True)
    years = dates.dt.year.to_numpy()
    idx = np.arange(len(dates))
    train_end = int(np.flatnonzero(years < year)[-1])
    val_idx = np.flatnonzero(years == year)
    if not len(val_idx):
        return None
    val_start, val_end = int(val_idx[0]), int(val_idx[-1])
    # Train labels and their full forward horizon must be known by the cutoff.
    train_mask = (idx >= horizon) & (idx <= train_end - horizon) & valid_rows
    # Score only validation rows whose full forward horizon exists in the dataset.
    val_mask = (years == year) & (idx >= val_start + horizon) & (idx <= len(dates) - horizon - 1) & valid_rows
    return {
        "train_mask": train_mask,
        "validation_mask": val_mask,
        "train_end_idx": train_end,
        "validation_start_idx": val_start,
        "validation_end_idx": val_end,
        "training_end": dates.iloc[train_end].date().isoformat(),
        "validation_start": dates.iloc[val_start].date().isoformat(),
        "validation_end": dates.iloc[val_end].date().isoformat(),
        "partial_validation_year": val_end == len(dates) - 1 and dates.iloc[val_end].month < 12,
    }


def fold_events(events: pd.DataFrame, dates: pd.Series, fold: dict, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return events.copy(), events.copy()
    dates = pd.to_datetime(dates).reset_index(drop=True)
    lookup = {pd.Timestamp(v).normalize(): i for i, v in enumerate(dates)}
    pivot_idx = pd.to_datetime(events["pivot_date"], errors="coerce").map(
        lambda v: lookup.get(pd.Timestamp(v).normalize(), -1)
    )
    confirmation = pd.to_datetime(events["confirmation_date"], errors="coerce")
    train_cutoff = dates.iloc[fold["train_end_idx"]]
    val_start = dates.iloc[fold["validation_start_idx"]]
    val_end_idx = min(len(dates) - 1, fold["validation_end_idx"] + horizon)
    val_end = dates.iloc[val_end_idx]
    train = (pivot_idx >= horizon) & (pivot_idx <= fold["train_end_idx"]) & (confirmation <= train_cutoff)
    validation = (
        (pivot_idx >= fold["validation_start_idx"] + horizon)
        & (pivot_idx <= val_end_idx)
        & (confirmation <= dates.iloc[-1])
        & (confirmation >= val_start)
    )
    return events.loc[train].copy(), events.loc[validation].copy()


def score_predictions(y_true: np.ndarray, probability: np.ndarray) -> dict:
    truth = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probability, dtype=float)
    if not len(truth):
        return {"rows": 0, "event_days": 0, "prevalence": None, "roc_auc": None, "pr_auc": None}
    prevalence = float(truth.mean())
    result = {
        "rows": int(len(truth)),
        "event_days": int(truth.sum()),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(truth, probability)) if len(np.unique(truth)) == 2 else None,
        "pr_auc": float(average_precision_score(truth, probability)) if truth.sum() else 0.0,
    }
    for pct in (1, 5, 10, 20):
        n = max(1, int(np.ceil(len(truth) * pct / 100)))
        top = np.argsort(probability)[-n:]
        precision = float(truth[top].mean())
        result[f"top_{pct}_precision"] = precision
        result[f"top_{pct}_lift"] = precision / prevalence if prevalence else None
        result[f"top_{pct}_recall"] = float(truth[top].sum() / truth.sum()) if truth.sum() else None
    return result
