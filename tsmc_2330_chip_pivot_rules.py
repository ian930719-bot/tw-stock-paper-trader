"""Compare interpretable 1-session HIGH/LOW pivot rules with and without institutional flow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, _tree

from tsmc_swing_research import detect_confirmed_pivots
from tsmc_2330_selected_threshold_signal_diagnostics import period_events, build_target

EXCLUDE = {
    "obv",  # cumulative level is non-stationary; use normalized OBV flow ratios instead
    "symbol", "date", "open", "high", "low", "close", "volume", "turnover",
    "corporate_action_window", "atr14",
}
CHIP_PREFIXES = (
    "foreign_", "foreign_dealer_", "trust_", "dealer_", "institutional_", "broker_",
)
INSTITUTIONAL_PREFIXES = (
    "foreign_", "foreign_dealer_", "trust_", "dealer_", "institutional_",
)
DEPTHS = (2, 3, 4)
TRAIN_START, TRAIN_END = "2010-01-01", "2016-12-31"
VAL_START, VAL_END = "2017-01-01", "2021-12-31"
TEST_START = "2022-01-01"


def feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    numeric = [
        col for col in frame.columns
        if col not in EXCLUDE and not col.startswith("raw_")
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    technical = [col for col in numeric if not col.startswith(CHIP_PREFIXES)]
    institutional = [col for col in numeric if col.startswith(INSTITUTIONAL_PREFIXES)]
    return {
        "technical_only": technical,
        "technical_plus_institutional": technical + institutional,
    }


def make_tree(depth: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=30,
            class_weight="balanced", random_state=0,
        ),
    )


def metrics(model, X: pd.DataFrame, mask: np.ndarray, y: np.ndarray) -> dict:
    if not mask.any() or len(np.unique(y[mask])) < 2:
        return {"error": "segment lacks rows or both classes", "rows": int(mask.sum())}
    p = model.predict_proba(X.loc[mask])[:, 1]
    truth = y[mask]
    prevalence = float(truth.mean())
    n = max(1, int(np.ceil(len(p) * .10)))
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


def rules_for(model, cols: list[str]) -> dict[int, str]:
    tree = model.named_steps["decisiontreeclassifier"].tree_
    names = list(cols)
    result = {}
    def walk(node: int, conditions: list[str]):
        left, right = tree.children_left[node], tree.children_right[node]
        if left == _tree.TREE_LEAF:
            result[node] = " AND ".join(conditions) if conditions else "ALL"
            return
        name = names[tree.feature[node]]
        threshold = float(tree.threshold[node])
        walk(left, conditions + [f"{name} <= {threshold:.8g}"])
        walk(right, conditions + [f"{name} > {threshold:.8g}"])
    walk(0, [])
    return result


def leaf_metrics(model, X: pd.DataFrame, mask: np.ndarray, y: np.ndarray,
                 cols: list[str], group: str, kind: str, segment: str) -> list[dict]:
    if not mask.any():
        return []
    imputer = model.named_steps["simpleimputer"]
    tree = model.named_steps["decisiontreeclassifier"]
    leaves = tree.apply(imputer.transform(X.loc[mask]))
    truth = y[mask]
    baseline = float(truth.mean())
    rule_map = rules_for(model, cols)
    rows = []
    for leaf in np.unique(leaves):
        take = leaves == leaf
        count = int(take.sum())
        positives = int(truth[take].sum())
        precision = positives / count if count else 0.0
        rows.append({
            "feature_group": group, "pivot_type": kind, "segment": segment,
            "leaf_id": int(leaf), "rule": rule_map[int(leaf)],
            "rows": count, "event_days": positives,
            "precision": precision, "baseline_prevalence": baseline,
            "lift": precision / baseline if baseline else None,
            "recall": positives / int(truth.sum()) if truth.sum() else None,
        })
    return rows


def run(features_path: str, oracle_path: str, output_dir: str) -> dict:
    frame = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    oracle = json.loads(Path(oracle_path).read_text(encoding="utf-8"))
    chosen = oracle.get("threshold_selected_using_training_period_only", {})
    reversal, atr_multiple = chosen.get("min_reversal_pct"), chosen.get("atr_multiple")
    if reversal is None or atr_multiple is None:
        raise ValueError("Missing training-only selected pivot threshold")
    events = detect_confirmed_pivots(frame, float(reversal), float(atr_multiple))
    dates = frame["date"]
    idx = np.arange(len(frame))
    train_mask = ((dates >= TRAIN_START) & (dates <= TRAIN_END)).to_numpy(copy=True)
    val_mask = ((dates >= VAL_START) & (dates <= VAL_END)).to_numpy(copy=True)
    test_mask = (dates >= TEST_START).to_numpy(copy=True)
    if "corporate_action_window" in frame:
        safe = ~frame["corporate_action_window"].fillna(True).astype(bool).to_numpy()
        train_mask &= safe
        val_mask &= safe
        test_mask &= safe
    # One-session target: next-session pivot label, no same-day execution assumption.
    train_mask &= idx < int(np.flatnonzero((dates >= pd.Timestamp(VAL_START)).to_numpy())[0]) - 1
    val_mask &= idx < int(np.flatnonzero((dates >= pd.Timestamp(TEST_START)).to_numpy())[0]) - 1
    groups = feature_groups(frame)
    output, leaves = {}, []
    for group, candidates in groups.items():
        X_all = frame[candidates].replace([np.inf, -np.inf], np.nan)
        # Critical: remove columns with no training observations BEFORE fitting.
        # Otherwise sklearn's SimpleImputer drops them and tree feature indices
        # no longer map to the original feature names.
        cols = [col for col in candidates if X_all.loc[train_mask, col].notna().any()]
        X = X_all[cols]
        if not cols:
            output[group] = {"status": "skipped_no_training_features"}
            continue
        for kind in ("HIGH", "LOW"):
            tr_events = period_events(events, dates, TRAIN_START, TRAIN_END, 1)
            va_events = period_events(events, dates, VAL_START, VAL_END, 1)
            te_events = period_events(events, dates, TEST_START, dates.max().date().isoformat(), 1)
            y_train = build_target(dates, tr_events, 1, kind)
            y_val = build_target(dates, va_events, 1, kind)
            y_test = build_target(dates, te_events, 1, kind)
            tr = train_mask & (idx >= 1)
            va = val_mask & (idx >= int(np.flatnonzero((dates >= pd.Timestamp(VAL_START)).to_numpy())[0]) + 1)
            te = test_mask & (idx >= int(np.flatnonzero((dates >= pd.Timestamp(TEST_START)).to_numpy())[0]) + 1)
            if len(np.unique(y_train[tr])) < 2 or len(np.unique(y_val[va])) < 2:
                output[f"{group}_{kind}"] = {"status": "skipped_one_class"}
                continue
            trials = {}
            for depth in DEPTHS:
                model = make_tree(depth)
                model.fit(X.loc[tr], y_train[tr])
                trials[depth] = {"model": model, "validation": metrics(model, X, va, y_val)}
            best_depth = max(DEPTHS, key=lambda d: trials[d]["validation"].get("pr_auc", -1))
            frozen = trials[best_depth]["model"]
            validation_leaves = leaf_metrics(frozen, X, va, y_val, cols, group, kind, "validation")
            transfer_metrics = metrics(frozen, X, te, y_test)
            transfer_leaves = leaf_metrics(frozen, X, te, y_test, cols, group, kind, "frozen_tree_test_2022_plus")
            leaves.extend(validation_leaves + transfer_leaves)
            output[f"{group}_{kind}"] = {
                "status": "evaluated", "feature_count": len(cols), "features": cols,
                "selected_depth": best_depth,
                "validation_depths": {str(d): trials[d]["validation"] for d in DEPTHS},
                "validation_metrics": trials[best_depth]["validation"],
                "frozen_tree_test_2022_plus": transfer_metrics,
                "validation_leaf_rules": validation_leaves,
                "frozen_tree_test_leaf_rules": transfer_leaves,
            }
    result = {
        "symbol": "2330",
        "pivot_threshold_selected_on_train_only": {
            "min_reversal_pct": float(reversal), "atr_multiple": float(atr_multiple)
        },
        "train": [TRAIN_START, TRAIN_END], "validation": [VAL_START, VAL_END],
        "descriptive_test": [TEST_START, dates.max().date().isoformat()],
        "target": "HIGH/LOW confirmed pivot event within next 1 trading session",
        "feature_groups": output, "leaf_rules": leaves,
        "test_period_used_for_selection": False, "test_period_is_pristine": False,
        "note": "One-session pivot event classification only, not a trading-return backtest. Frozen validation trees are applied unchanged to 2022+; that period was previously inspected. All-empty training features are removed before fitting to preserve rule-name alignment.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_chip_pivot_rules.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    pd.DataFrame(leaves).to_csv(out / "tsmc_2330_chip_pivot_rules.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.oracle, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
