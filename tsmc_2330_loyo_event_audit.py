"""Leave-one-year-out event/control discrimination audit for TSMC 2330.
This is an exploratory generalization diagnostic, NOT a trading backtest.
Retrospective pivot labels and candidate preselection prevent an independent
out-of-sample performance claim.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONFIG = {
    "HIGH": {
        "technical": ["distance_sma_200", "sma50_slope_10", "overnight_gap",
                      "bearish_volume_divergence20", "return_1"],
        "technical_plus_institutional": [
            "distance_sma_200", "sma50_slope_10", "overnight_gap",
            "bearish_volume_divergence20", "return_1",
            "foreign_net_to_volume", "trust_net_to_volume",
            "trust_net_negative_streak"],
    },
    "LOW": {
        "technical": ["distance_sma_200", "minus_di14", "sma50_slope_10",
                      "overnight_gap"],
        "technical_plus_institutional": [
            "distance_sma_200", "minus_di14", "sma50_slope_10",
            "overnight_gap", "foreign_net_to_volume",
            "trust_net_to_volume", "trust_net_negative_streak"],
    },
}
INSTITUTIONAL_START = pd.Timestamp("2012-05-02")


def run_audit(input_csv: str | Path, output_dir: str | Path) -> dict:
    d = pd.read_csv(input_csv, parse_dates=["pivot_date", "feature_date", "control_date"])
    required = {"pivot_type", "lead_sessions", "pivot_date", "feature_date",
                "control_date", "event_no"}
    if not required.issubset(d.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(d.columns))}")
    d["pivot_year"] = d.pivot_date.dt.year
    rows = []
    for kind, model_sets in CONFIG.items():
        for lead in sorted(d.lead_sessions.dropna().unique()):
            g = d[(d.pivot_type == kind) & (d.lead_sessions == lead)].copy()
            if g.empty:
                continue
            for model_name, features in model_sets.items():
                # Keep the full historical span for technical-only tests. Restrict
                # only models containing institutional flow to the verified coverage
                # window, for both event and matched-control feature dates.
                model_data = g
                if model_name == "technical_plus_institutional":
                    model_data = g[(g.feature_date >= INSTITUTIONAL_START) &
                                   (g.control_date >= INSTITUTIONAL_START)]
                if model_data.empty:
                    continue
                missing = [f for f in features if f not in model_data or f"control_{f}" not in model_data]
                if missing:
                    raise ValueError(f"Missing paired features for {kind}: {missing}")
                event_x = model_data[features].apply(pd.to_numeric, errors="coerce")
                control_x = model_data[[f"control_{f}" for f in features]].apply(
                    pd.to_numeric, errors="coerce")
                control_x.columns = features
                x = pd.concat([event_x, control_x], ignore_index=True)
                y = np.r_[np.ones(len(event_x), dtype=int),
                          np.zeros(len(control_x), dtype=int)]
                years = np.r_[model_data.pivot_year.to_numpy(), model_data.pivot_year.to_numpy()]
                for heldout in sorted(model_data.pivot_year.unique()):
                    train, test = years != heldout, years == heldout
                    if len(np.unique(y[test])) != 2 or len(np.unique(y[train])) != 2:
                        continue
                    model = make_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(),
                        LogisticRegression(C=0.1, max_iter=3000,
                                           class_weight="balanced", random_state=2330))
                    model.fit(x.loc[train], y[train])
                    probability = model.predict_proba(x.loc[test])[:, 1]
                    rows.append({
                        "pivot_type": kind, "lead_sessions": int(lead),
                        "model": model_name, "heldout_year": int(heldout),
                        "test_events": int((model_data.pivot_year == heldout).sum()),
                        "train_events": int((model_data.pivot_year != heldout).sum()),
                        "auc": float(roc_auc_score(y[test], probability)),
                        "balanced_accuracy": float(balanced_accuracy_score(
                            y[test], probability >= 0.5)),
                        "feature_count": len(features),
                    })
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("No valid leave-one-year-out folds were produced")
    summary = (detail.groupby(["pivot_type", "lead_sessions", "model"])
        .agg(year_folds=("heldout_year", "nunique"),
             mean_auc=("auc", "mean"), median_auc=("auc", "median"),
             share_year_auc_over_055=("auc", lambda s: float((s > 0.55).mean())),
             mean_balanced_accuracy=("balanced_accuracy", "mean"))
        .reset_index())
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "tsmc_2330_loyo_fold_metrics.csv", index=False)
    summary.to_csv(out / "tsmc_2330_loyo_summary.csv", index=False)
    result = {
        "input_rows": int(len(d)),
        "institutional_coverage_start": "2012-05-02; applied only to models containing foreign/trust features",
        "technical_only_uses_full_history": True,
        "validation": "leave-one-pivot-year-out; all observations from held-out year excluded from training",
        "models": CONFIG,
        "dealer_features": "excluded; only foreign-investor and investment-trust features used",
        "summary_rows": int(len(summary)),
        "warning": (
            "Exploratory event-versus-matched-control discrimination only. "
            "This is not a prospective signal test, trade simulation, or annualized return. "
            "Candidate features were selected after inspecting the same 500-pivot study, "
            "so this diagnostic is post-selection and not an untouched final holdout. "
            "Retrospective pivot labels, overlapping events, repeated regimes, and "
            "control matching remain limitations. Next validation must freeze rules "
            "before a genuinely untouched chronological period and include fees, tax, "
            "slippage, execution timing, and false-signal costs."
        ),
        "outputs": ["tsmc_2330_loyo_fold_metrics.csv", "tsmc_2330_loyo_summary.csv"],
    }
    (out / "tsmc_2330_loyo_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.input, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
