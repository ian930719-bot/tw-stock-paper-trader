"""Expanding OOF, single-feature rule discovery for causal ATR-2 low events.

This is an event-quality research diagnostic, not an executable strategy.
Each annual validation fold selects one causal feature threshold using only
matured labels from earlier years. No pivot-date retrospective fields are used.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = [
    "confirm_return_1", "confirm_return_5", "confirm_return_20",
    "confirm_atr_pct14", "confirm_rsi14", "confirm_macd_hist_change_3",
    "confirm_distance_sma_20", "confirm_distance_ema_20",
    "confirm_distance_sma_50", "confirm_distance_sma_200",
    "confirm_relative_volume20", "confirm_close_location",
    "confirm_intraday_range_pct", "confirm_obv_flow_ratio_5",
    "confirm_plus_di14", "confirm_minus_di14", "confirm_mfi14",
    "confirm_stoch_k14", "confirmation_lag_sessions",
    "reversal_at_confirmation_pct",
    "pre_pivot_5d_return", "pre_pivot_10d_return",
    "pre_pivot_20d_return", "pre_pivot_40d_return",
    "pre_pivot_60d_return", "pre_pivot_5d_range_pct",
    "pre_pivot_10d_range_pct", "pre_pivot_20d_range_pct",
    "pre_pivot_40d_range_pct", "pre_pivot_60d_range_pct",
]

def rule_mask(frame, feature, direction, threshold):
    x = pd.to_numeric(frame[feature], errors="coerce")
    return (x <= threshold) if direction == "low" else (x >= threshold)

def select_rule(train):
    candidates = []
    for feature in FEATURES:
        x = pd.to_numeric(train[feature], errors="coerce")
        valid = x.notna()
        if int(valid.sum()) < 25 or x[valid].nunique() < 5:
            continue
        for q in (0.20, 0.30, 0.40):
            threshold = float(x[valid].quantile(q))
            for direction in ("low", "high"):
                mask = valid & rule_mask(train, feature, direction, threshold)
                n = int(mask.sum())
                coverage = n / len(train)
                if n < 8 or coverage < 0.15:
                    continue
                precision = float(train.loc[mask, "quality_label"].mean())
                # Penalize tiny coverage while rewarding precision above baseline.
                score = (precision - float(train.quality_label.mean())) * np.sqrt(coverage)
                candidates.append({
                    "feature": feature, "direction": direction,
                    "quantile": q, "threshold": threshold, "train_n": n,
                    "train_coverage": coverage, "train_precision": precision,
                    "train_baseline": float(train.quality_label.mean()),
                    "selection_score": float(score),
                })
    if not candidates:
        return None
    # Deterministic tie-breaks: score, precision, coverage, feature name.
    return sorted(candidates, key=lambda r: (
        r["selection_score"], r["train_precision"], r["train_coverage"], r["feature"]
    ), reverse=True)[0]

def summarize_rows(rows):
    if not rows:
        return {"n": 0, "positive_rate": None, "precision": None, "recall": None, "coverage": None}
    y = np.asarray([r["actual_quality_label"] for r in rows], dtype=int)
    p = np.asarray([r["selected_by_rule"] for r in rows], dtype=bool)
    tp = int(np.sum((y == 1) & p))
    return {
        "n": int(len(y)), "positive_rate": float(y.mean()),
        "selected_n": int(p.sum()), "selected_positive_n": tp,
        "precision": float(tp / p.sum()) if p.sum() else None,
        "recall": float(tp / np.sum(y)) if np.sum(y) else None,
        "coverage": float(p.mean()),
        "unfiltered_baseline_precision": float(y.mean()),
    }

def run(events_path, output):
    d = pd.read_csv(events_path, parse_dates=["confirmation_date"])
    d = d[np.isclose(pd.to_numeric(d.atr_multiplier, errors="coerce"), 2.0)].copy()
    d["corporate_action_window_at_confirmation"] = (
        d.corporate_action_window_at_confirmation.astype(str).str.lower().isin(["true", "1", "yes"])
    )
    d = d[~d.corporate_action_window_at_confirmation].copy()
    # Only fully observed 60-session labels are admissible.
    d = d[pd.to_numeric(d.observed_sessions_60d, errors="coerce").eq(60)].copy()
    d = d.dropna(subset=["close_return_60d", "mfe_60d"])
    d["quality_label"] = (
        (pd.to_numeric(d.close_return_60d) > 0) &
        (pd.to_numeric(d.mfe_60d) >= 0.08)
    ).astype(int)
    d["validation_year"] = d.confirmation_date.dt.year
    predictions, folds = [], []
    for year in sorted(d.validation_year.unique()):
        # Labels must have matured before validation year: cutoff is Dec 31 two years earlier.
        cutoff = pd.Timestamp(year=int(year)-2, month=12, day=31)
        train = d[d.confirmation_date <= cutoff].copy()
        test = d[d.validation_year == year].copy()
        if len(train) < 30 or train.quality_label.nunique() < 2 or test.empty:
            folds.append({"validation_year": int(year), "train_n": int(len(train)),
                          "test_n": int(len(test)), "status": "insufficient_matured_training"})
            continue
        rule = select_rule(train)
        if rule is None:
            folds.append({"validation_year": int(year), "train_n": int(len(train)),
                          "test_n": int(len(test)), "status": "no_supported_rule"})
            continue
        selected = rule_mask(test, rule["feature"], rule["direction"], rule["threshold"])
        selected = selected.fillna(False).to_numpy(bool)
        for (_, row), keep in zip(test.iterrows(), selected):
            predictions.append({
                "confirmation_date": row.confirmation_date.date().isoformat(),
                "validation_year": int(year),
                "actual_quality_label": int(row.quality_label),
                "selected_by_rule": bool(keep),
                "feature": rule["feature"], "direction": rule["direction"],
                "threshold": rule["threshold"],
                "close_return_60d": float(row.close_return_60d),
                "mfe_60d": float(row.mfe_60d),
            })
        folds.append({
            "validation_year": int(year), "training_cutoff": cutoff.date().isoformat(),
            "train_n": int(len(train)), "test_n": int(len(test)), "status": "scored",
            "selected_rule": rule,
            "validation": summarize_rows([
                {"actual_quality_label": int(row.quality_label), "selected_by_rule": bool(keep)}
                for (_, row), keep in zip(test.iterrows(), selected)
            ]),
        })
    pred = pd.DataFrame(predictions)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "tsmc_2330_atr2_low_rule_oof_predictions.csv", index=False)
    result = {
        "symbol": "2330",
        "classification": "EXPANDING_ANNUAL_OOF_EVENT_RULE_DIAGNOSTIC_NOT_TRADING_STRATEGY",
        "source_atr_multiplier": 2.0,
        "event_count_after_complete_label_filter": int(len(d)),
        "quality_label": "60-session close return >0 and MFE >=8%; complete 60-session window only",
        "features": FEATURES,
        "feature_timing": "Confirmation-close features and pre-pivot history known by confirmation; no retrospective pivot-row feature columns.",
        "training_protocol": "Annual expanding OOF; train labels confirmed no later than Dec 31 two calendar years before validation year; min 30 matured training events; one feature threshold selected from training only.",
        "rule_selection": "Among predeclared feature lower/upper 20%, 30%, 40% quantile rules with at least 8 selected training events and 15% training coverage, maximize (precision - training base rate) * sqrt(coverage).",
        "overall_oof": summarize_rows(pred.to_dict("records")) if len(pred) else {"n": 0},
        "annual_folds": folds,
        "predictions": int(len(pred)),
        "limitations": [
            "This is event-quality filtering, not a trade simulator and cannot establish CAGR, completed-trade win rate, or drawdown.",
            "Single-feature selection still incurs multiple-comparison risk; annual OOF mitigates but does not erase research-family selection bias.",
            "ATR-2 events are temporally clustered and are not independent.",
            "Historical 2010-2026 observations have been examined in prior research; this is not pristine blind OOS."
        ],
    }
    (out / "tsmc_2330_atr2_low_rule_oof.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True)
    p.add_argument("--output", default="artifacts/tsmc_2330_atr2_low_rule_oof")
    a = p.parse_args()
    print(json.dumps(run(a.events, a.output), ensure_ascii=False, indent=2))
