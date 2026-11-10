"""Annual walk-forward interpretable HIGH/LOW pivot rules for TSMC 2330.

Compares technical-only against technical-plus-institutional features. Tree
depth is fixed before validation; each annual fold fits only on confirmed prior
events. Broker fields remain nullable and are not imputed as zero when absent.
This is event classification, not an executable trading-return backtest.
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
from tsmc_2330_walk_forward import (
    annual_fold_years, event_target, fold_events, fold_masks,
)

EXCLUDE = {
    "obv", "symbol", "date", "open", "high", "low", "close", "volume",
    "turnover", "corporate_action_window", "atr14",
}
INSTITUTIONAL_PREFIXES = (
    "foreign_", "foreign_dealer_", "trust_", "dealer_", "institutional_",
)
DEPTH = 3
HORIZON = 1


def feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    numeric = [
        col for col in frame.columns
        if col not in EXCLUDE and not col.startswith("raw_")
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    institutional = [c for c in numeric if c.startswith(INSTITUTIONAL_PREFIXES)]
    technical = [c for c in numeric if c not in institutional]
    return {
        "technical_only": technical,
        "technical_plus_institutional": technical + institutional,
    }


def make_tree(depth: int = DEPTH):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(
            max_depth=int(depth), min_samples_leaf=30,
            class_weight="balanced", random_state=0,
        ),
    )


def metrics(truth: np.ndarray, probability: np.ndarray) -> dict:
    truth = np.asarray(truth, dtype=np.int8)
    probability = np.asarray(probability, dtype=float)
    if not len(truth):
        return {"rows": 0, "event_days": 0, "prevalence": None,
                "roc_auc": None, "pr_auc": None}
    prevalence = float(truth.mean())
    n = max(1, int(np.ceil(len(truth) * 0.10)))
    top = np.argsort(probability)[-n:]
    precision = float(truth[top].mean())
    return {
        "rows": int(len(truth)), "event_days": int(truth.sum()),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(truth, probability))
        if len(np.unique(truth)) == 2 else None,
        "pr_auc": float(average_precision_score(truth, probability))
        if truth.sum() else 0.0,
        "top_decile_precision": precision,
        "top_decile_lift": precision / prevalence if prevalence else None,
        "top_decile_recall": float(truth[top].sum() / truth.sum())
        if truth.sum() else None,
    }


def rules_for(model, cols: list[str]) -> dict[int, str]:
    tree = model.named_steps["decisiontreeclassifier"].tree_
    result = {}

    def walk(node: int, conditions: list[str]):
        left, right = tree.children_left[node], tree.children_right[node]
        if left == _tree.TREE_LEAF:
            result[node] = " AND ".join(conditions) if conditions else "ALL"
            return
        name = cols[tree.feature[node]]
        threshold = float(tree.threshold[node])
        walk(left, conditions + [f"{name} <= {threshold:.8g}"])
        walk(right, conditions + [f"{name} > {threshold:.8g}"])

    walk(0, [])
    return result


def run(features_path: str, oracle_path: str, output_dir: str) -> dict:
    frame = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    oracle = json.loads(Path(oracle_path).read_text(encoding="utf-8"))
    chosen = oracle.get("threshold_selected_using_training_period_only", {})
    reversal, atr_multiple = chosen.get("min_reversal_pct"), chosen.get("atr_multiple")
    if reversal is None or atr_multiple is None:
        raise ValueError("Missing training-only selected pivot threshold")
    events = detect_confirmed_pivots(frame, float(reversal), float(atr_multiple))
    if events.empty:
        raise ValueError("No confirmed pivot events")
    dates = frame["date"]
    groups = feature_groups(frame)
    years = annual_fold_years(dates)
    annual_rows, pooled_rows, leaf_rows = [], [], []

    for group, candidates in groups.items():
        X_all = frame[candidates].replace([np.inf, -np.inf], np.nan)
        valid_rows = X_all.notna().any(axis=1).to_numpy(copy=True)
        if "corporate_action_window" in frame:
            valid_rows &= ~frame["corporate_action_window"].fillna(True).astype(bool).to_numpy()
        for kind in ("HIGH", "LOW"):
            pooled = {"truth": [], "probability": []}
            for year in years:
                fold = fold_masks(dates, valid_rows, year, HORIZON)
                if fold is None or not fold["validation_mask"].any():
                    continue
                train_events, val_events = fold_events(events, dates, fold, HORIZON)
                train_mask, val_mask = fold["train_mask"], fold["validation_mask"]
                y_train = event_target(dates, train_events, HORIZON, kind)
                y_val = event_target(dates, val_events, HORIZON, kind)
                if "corporate_action_window" in frame:
                    safe = ~frame["corporate_action_window"].fillna(True).astype(bool).to_numpy()
                    train_mask &= safe
                    val_mask &= safe
                cols = [c for c in candidates if X_all.loc[train_mask, c].notna().any()]
                if not cols or not train_mask.any() or len(np.unique(y_train[train_mask])) < 2:
                    annual_rows.append({
                        "feature_group": group, "pivot_type": kind,
                        "validation_year": year, "status": "skipped_insufficient_training_data",
                        "training_rows": int(train_mask.sum()),
                        "training_events": int(y_train[train_mask].sum()),
                    })
                    continue
                X = X_all[cols]
                model = make_tree()
                model.fit(X.loc[train_mask], y_train[train_mask])
                probability = model.predict_proba(X.loc[val_mask])[:, 1]
                truth = y_val[val_mask]
                m = metrics(truth, probability)
                annual_rows.append({
                    "feature_group": group, "pivot_type": kind,
                    "validation_year": year, "status": "evaluated",
                    "training_start": dates.loc[train_mask].min().date().isoformat(),
                    "training_end": dates.loc[train_mask].max().date().isoformat(),
                    "validation_start": dates.loc[val_mask].min().date().isoformat(),
                    "validation_end": dates.loc[val_mask].max().date().isoformat(),
                    "partial_validation_year": year == int(dates.max().year),
                    "training_rows": int(train_mask.sum()),
                    "training_events": int(y_train[train_mask].sum()),
                    "feature_count": len(cols), "features": cols,
                    "fixed_tree_depth": DEPTH, **m,
                })
                pooled["truth"].extend(truth.tolist())
                pooled["probability"].extend(probability.tolist())

                imputer = model.named_steps["simpleimputer"]
                tree = model.named_steps["decisiontreeclassifier"]
                leaves = tree.apply(imputer.transform(X.loc[val_mask]))
                rule_map = rules_for(model, cols)
                baseline = float(truth.mean()) if len(truth) else 0.0
                for leaf in np.unique(leaves):
                    take = leaves == leaf
                    count = int(take.sum())
                    positives = int(truth[take].sum())
                    precision = positives / count if count else 0.0
                    leaf_rows.append({
                        "feature_group": group, "pivot_type": kind,
                        "validation_year": year, "leaf_id": int(leaf),
                        "rule": rule_map[int(leaf)], "rows": count,
                        "event_days": positives, "precision": precision,
                        "baseline_prevalence": baseline,
                        "lift": precision / baseline if baseline else None,
                        "recall": positives / int(truth.sum()) if truth.sum() else None,
                    })
            pooled_rows.append({
                "feature_group": group, "pivot_type": kind,
                "fixed_tree_depth": DEPTH,
                **metrics(np.asarray(pooled["truth"], dtype=np.int8),
                          np.asarray(pooled["probability"], dtype=float)),
            })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(annual_rows).to_csv(out / "tsmc_2330_chip_pivot_rules_annual.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(out / "tsmc_2330_chip_pivot_rules_pooled.csv", index=False)
    pd.DataFrame(leaf_rows).to_csv(out / "tsmc_2330_chip_pivot_rules.csv", index=False)
    summary = {
        "symbol": "2330",
        "pivot_threshold_selected_on_training_only": {
            "min_reversal_pct": float(reversal), "atr_multiple": float(atr_multiple)
        },
        "validation_method": "annual expanding walk-forward; train through Y-1, validate Y",
        "validation_years": years, "target": "HIGH/LOW confirmed pivot event within next session",
        "feature_groups": list(groups), "fixed_tree_depth": DEPTH,
        "annual_results": annual_rows, "pooled_out_of_fold_results": pooled_rows,
        "leaf_rules": leaf_rows, "validation_year_selection": False,
        "strategy_backtest": False, "pristine_holdout": False,
        "note": "Technical-only and technical-plus-institutional models are evaluated on the same annual folds. Tree depth is fixed before validation; no validation-year model selection. Missing broker fields are left null, never converted to zero. Latest calendar year is partial. Classification metrics are not trading returns or CAGR.",
    }
    (out / "tsmc_2330_chip_pivot_rules.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.oracle, args.output),
                     ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
