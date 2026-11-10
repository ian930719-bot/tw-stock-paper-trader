"""Interpretable, shallow decision-tree rules for TSMC 2330 swing pivots.

Pivot threshold is selected on 2010-2016 only. Tree depth is selected on
2017-2021 validation PR-AUC, then the chosen tree is refit through 2021.
2022+ metrics are descriptive because that period was previously inspected.
"""
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

HORIZONS = (1, 3, 5)
DEPTHS = (2, 3, 4)
VALIDATION_START = "2017-01-01"
VALIDATION_END = "2021-12-31"
TEST_START = "2022-01-01"
EXCLUDE = {
    "symbol", "date", "open", "high", "low", "close", "volume",
    "turnover", "corporate_action_window", "atr14",
}


def make_tree(depth: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=25,
            class_weight="balanced", random_state=0,
        ),
    )


def evaluate(model, X: pd.DataFrame, mask: np.ndarray, y: np.ndarray) -> dict:
    if not mask.any() or len(np.unique(y[mask])) < 2:
        return {"error": "evaluation segment lacks rows or both target classes",
                "rows": int(mask.sum()), "event_days": int(y[mask].sum())}
    p = model.predict_proba(X.loc[mask])[:, 1]
    truth = y[mask]
    prevalence = float(truth.mean())
    out = {
        "rows": int(mask.sum()), "event_days": int(truth.sum()),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(truth, p)),
        "pr_auc": float(average_precision_score(truth, p)),
    }
    for frac in (0.05, 0.10, 0.20):
        n = max(1, int(np.ceil(len(p) * frac)))
        ix = np.argsort(p)[-n:]
        precision = float(truth[ix].mean())
        key = str(int(frac * 100))
        out[f"top_{key}_precision"] = precision
        out[f"top_{key}_lift"] = precision / prevalence if prevalence else None
        out[f"top_{key}_recall"] = float(truth[ix].sum() / truth.sum()) if truth.sum() else None
    return out


def _rule_map(model, feature_names: list[str]) -> dict[int, str]:
    tree = model.named_steps["decisiontreeclassifier"].tree_
    rules: dict[int, str] = {}

    def walk(node: int, conditions: list[str]):
        left, right = tree.children_left[node], tree.children_right[node]
        if left == _tree.TREE_LEAF:
            rules[node] = " AND ".join(conditions) if conditions else "ALL"
            return
        feature = feature_names[tree.feature[node]]
        threshold = float(tree.threshold[node])
        walk(left, conditions + [f"{feature} <= {threshold:.8g}"])
        walk(right, conditions + [f"{feature} > {threshold:.8g}"])

    walk(0, [])
    return rules


def _leaf_rows(model, X: pd.DataFrame, mask: np.ndarray, y: np.ndarray,
               feature_names: list[str], segment: str, kind: str, horizon: int,
               depth: int) -> list[dict]:
    if not mask.any():
        return []
    imputer = model.named_steps["simpleimputer"]
    tree_model = model.named_steps["decisiontreeclassifier"]
    leaf_ids = tree_model.apply(imputer.transform(X.loc[mask]))
    truth = y[mask]
    base = float(truth.mean())
    rules = _rule_map(model, feature_names)
    rows = []
    for leaf in sorted(np.unique(leaf_ids)):
        selected = leaf_ids == leaf
        n = int(selected.sum())
        positives = int(truth[selected].sum())
        rows.append({
            "pivot_type": kind, "horizon_sessions": horizon,
            "tree_depth_selected": depth, "segment": segment,
            "leaf_id": int(leaf), "rule": rules[int(leaf)],
            "rows": n, "event_days": positives,
            "precision": positives / n if n else None,
            "baseline_prevalence": base,
            "lift": (positives / n / base) if n and base else None,
            "recall": positives / int(truth.sum()) if truth.sum() else None,
            "min_leaf_training_rows": int(tree_model.tree_.n_node_samples[leaf]),
            "model_leaf_event_probability": float(tree_model.tree_.value[leaf][0][1] /
                tree_model.tree_.value[leaf][0].sum()),
        })
    return rows


def run(features_path: str, oracle_path: str, output_dir: str) -> dict:
    x = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    oracle = json.loads(Path(oracle_path).read_text(encoding="utf-8"))
    selected = oracle.get("threshold_selected_using_training_period_only", {})
    min_reversal, atr_multiple = selected.get("min_reversal_pct"), selected.get("atr_multiple")
    if min_reversal is None or atr_multiple is None:
        raise ValueError("Missing training-selected pivot threshold")
    events = detect_confirmed_pivots(x, float(min_reversal), float(atr_multiple))
    if events.empty:
        raise ValueError("No pivots for training-selected threshold")
    cols = [c for c in x.columns if c not in EXCLUDE and not c.startswith("raw_")
            and pd.api.types.is_numeric_dtype(x[c])]
    X = x[cols].replace([np.inf, -np.inf], np.nan)
    valid = X.notna().any(axis=1).to_numpy(copy=True)
    if "corporate_action_window" in x:
        valid &= ~x["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    dates = x["date"]
    idx = np.arange(len(x))
    train_start = int(np.flatnonzero((dates >= pd.Timestamp("2010-01-01")).to_numpy())[0])
    val_start = int(np.flatnonzero((dates >= pd.Timestamp(VALIDATION_START)).to_numpy())[0])
    test_start = int(np.flatnonzero((dates >= pd.Timestamp(TEST_START)).to_numpy())[0])
    results, rule_rows = {}, []
    for horizon in HORIZONS:
        train_events = period_events(events, dates, "2010-01-01", "2016-12-31", horizon)
        val_events = period_events(events, dates, VALIDATION_START, VALIDATION_END, horizon)
        test_events = period_events(events, dates, TEST_START, dates.max().date().isoformat(), horizon)
        train_mask = (idx >= train_start + horizon) & (idx <= val_start - horizon - 1) & valid
        val_mask = ((dates >= VALIDATION_START) & (dates <= VALIDATION_END)).to_numpy()
        val_mask &= (idx >= val_start + horizon) & (idx <= test_start - horizon - 1) & valid
        test_mask = (idx >= test_start + horizon) & valid
        train_val_mask = (idx >= train_start + horizon) & (idx <= test_start - horizon - 1) & valid
        for kind in ("HIGH", "LOW"):
            y_train = build_target(dates, train_events, horizon, kind)
            y_val = build_target(dates, val_events, horizon, kind)
            y_test = build_target(dates, test_events, horizon, kind)
            y_train_val = build_target(dates, pd.concat([train_events, val_events], ignore_index=True), horizon, kind)
            key = f"{kind}_{horizon}"
            if len(np.unique(y_train[train_mask])) < 2:
                results[key] = {"status": "skipped", "reason": "training target has one class"}
                continue
            if not val_mask.any() or len(np.unique(y_val[val_mask])) < 2:
                results[key] = {"status": "skipped", "reason": "validation target has one class"}
                continue
            candidates = {}
            for depth in DEPTHS:
                model = make_tree(depth)
                model.fit(X.loc[train_mask], y_train[train_mask])
                candidates[str(depth)] = {
                    "validation": evaluate(model, X, val_mask, y_val),
                    "model": model,
                }
            chosen_depth = max(DEPTHS, key=lambda d: (
                candidates[str(d)]["validation"]["pr_auc"],
                candidates[str(d)]["validation"].get("top_10_precision", -1),
            ))
            validation_model = candidates[str(chosen_depth)]["model"]
            validation_rules = _leaf_rows(
                validation_model, X, val_mask, y_val, cols,
                "validation", kind, horizon, chosen_depth
            )
            final_model = make_tree(chosen_depth)
            final_model.fit(X.loc[train_val_mask], y_train_val[train_val_mask])
            test_metrics = evaluate(final_model, X, test_mask, y_test)
            test_rules = _leaf_rows(
                final_model, X, test_mask, y_test, cols,
                "test_2022_plus", kind, horizon, chosen_depth
            )
            rule_rows.extend(validation_rules + test_rules)
            results[key] = {
                "selected_depth": chosen_depth,
                "validation_candidates": {
                    d: {"validation": candidates[d]["validation"]} for d in candidates
                },
                "validation_metrics": candidates[str(chosen_depth)]["validation"],
                "test_metrics": test_metrics,
                "validation_leaf_rules": validation_rules,
                "refit_test_leaf_rules": test_rules,
            }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rule_rows).to_csv(out / "tsmc_2330_interpretable_pivot_rules.csv", index=False)
    summary = {
        "symbol": "2330",
        "pivot_threshold": {"min_reversal_pct": float(min_reversal), "atr_multiple": float(atr_multiple)},
        "feature_count": len(cols), "feature_columns": cols,
        "depths_compared": list(DEPTHS),
        "depth_selection_metric": "2017-2021 validation PR-AUC; tie-break top-decile precision",
        "train_period": ["2010-01-01", "2016-12-31"],
        "validation_period": [VALIDATION_START, VALIDATION_END],
        "test_period": [TEST_START, dates.max().date().isoformat()],
        "results": results,
        "test_period_used_for_selection": False,
        "test_period_is_pristine": False,
        "strategy_backtest": False,
        "warning": "These are interpretable event-classification rules, not yet executable buy/sell rules. Pivot labels are retrospective. 2022+ has previously been inspected and is descriptive only. Leaf precision is sample-dependent; inspect row counts and repeated stability before adopting any rule."
    }
    (out / "tsmc_2330_interpretable_pivot_rules.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.oracle, args.output), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
