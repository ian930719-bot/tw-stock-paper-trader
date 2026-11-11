"""Paired event/control audit of fixed Taiwan swing-trading heuristics.

Pre-specified exploratory rules only; not a tradable strategy or P&L backtest.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

LEADS = (1, 3, 5, 10, 20)
BOOTSTRAPS = 5000
SEED = 23301111

def rules():
    return {
        "LOW_volume_dryup_near_low": ("LOW", lambda x: (x["relative_volume20"] <= 0.8) & (x["distance_prior_low20"] <= 0.05)),
        "LOW_bullish_divergence": ("LOW", lambda x: (x["bullish_rsi_divergence20"] > 0) | (x["bullish_macd_divergence20"] > 0) | (x["bullish_volume_divergence20"] > 0)),
        "LOW_oversold_RSI_stoch": ("LOW", lambda x: (x["rsi14"] <= 35) & (x["stoch_k14"] <= 25)),
        "LOW_foreign_and_trust_5d_buy": ("LOW", lambda x: (x["foreign_net_5d_lots"] > 0) & (x["trust_net_5d_lots"] > 0)),
        "LOW_OBV_positive_MFI_low": ("LOW", lambda x: (x["obv_flow_ratio_5"] > 0) & (x["mfi14"] < 40)),
        # Proxy only: source idea uses the 60-day line and fundamentals; snapshot has SMA50 and no monthly revenue.
        "LOW_sma50_pullback_positive_close_proxy": ("LOW", lambda x: (x["distance_sma_50"].abs() <= 0.04) & (x["return_1"] > 0)),
        "HIGH_bearish_divergence": ("HIGH", lambda x: (x["bearish_rsi_divergence20"] > 0) | (x["bearish_macd_divergence20"] > 0) | (x["bearish_volume_divergence20"] > 0)),
        "HIGH_blowoff_reversal": ("HIGH", lambda x: (x["relative_volume20"] >= 1.5) & (x["return_1"] < 0) & (x["close_location"] <= 0.35)),
        "HIGH_overbought_RSI_stoch": ("HIGH", lambda x: (x["rsi14"] >= 70) & (x["stoch_k14"] >= 80)),
        "HIGH_positive_bias_15pct": ("HIGH", lambda x: x["distance_sma_20"] >= 0.15),
        "HIGH_foreign_and_trust_5d_sell": ("HIGH", lambda x: (x["foreign_net_5d_lots"] < 0) & (x["trust_net_5d_lots"] < 0)),
    }

def paired_metrics(frame, fn, rng, n_boot):
    event = fn(frame).fillna(False).to_numpy(dtype=bool)
    cf = frame.copy()
    for col in [c for c in frame.columns if c.startswith("control_")]:
        cf[col[8:]] = frame[col]
    control = fn(cf).fillna(False).to_numpy(dtype=bool)
    n = len(frame)
    if not n:
        raise ValueError("empty event/control slice")
    draws = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, n, size=n)
        draws[b] = event[ix].mean() - control[ix].mean()
    e_only = int((event & ~control).sum())
    c_only = int((~event & control).sum())
    p = float(binomtest(e_only, e_only + c_only, 0.5).pvalue) if e_only + c_only else 1.0
    return {
        "n_events": n, "event_hit_rate": float(event.mean()),
        "matched_control_hit_rate": float(control.mean()),
        "paired_risk_difference": float(event.mean() - control.mean()),
        "bootstrap_ci_low": float(np.quantile(draws, .025)),
        "bootstrap_ci_high": float(np.quantile(draws, .975)),
        "event_only_pairs": e_only, "control_only_pairs": c_only,
        "mcnemar_exact_p_unadjusted": p,
    }

def run(input_csv, output_dir, n_boot=BOOTSTRAPS):
    d = pd.read_csv(input_csv, parse_dates=["pivot_date", "feature_date", "control_date"])
    d = d.loc[d.threshold_id.astype(str) == "p0.020_atr1.0"].copy()
    if d.empty:
        raise ValueError("No rows for frozen threshold p0.020_atr1.0")
    if d.duplicated(["event_no", "lead_sessions"]).any():
        raise ValueError("Duplicate event/lead rows")
    if set(d.pivot_type.unique()) != {"HIGH", "LOW"}:
        raise ValueError("Expected HIGH and LOW pivots")
    rng = np.random.default_rng(SEED)
    rows, year_rows = [], []
    for name, (kind, fn) in rules().items():
        for lead in LEADS:
            part = d.loc[(d.pivot_type.str.upper() == kind) & (d.lead_sessions == lead)].copy()
            m = paired_metrics(part, fn, rng, n_boot)
            rows.append({"rule": name, "pivot_type": kind, "lead_sessions": lead, **m})
            cf = part.copy()
            for col in [c for c in part.columns if c.startswith("control_")]:
                cf[col[8:]] = part[col]
            ev = fn(part).fillna(False).astype(int).to_numpy()
            ct = fn(cf).fillna(False).astype(int).to_numpy()
            years = part.pivot_date.dt.year.to_numpy()
            for year in sorted(np.unique(years)):
                mask = years == year
                year_rows.append({
                    "rule": name, "pivot_type": kind, "lead_sessions": lead,
                    "year": int(year), "n_events": int(mask.sum()),
                    "event_hit_rate": float(ev[mask].mean()),
                    "matched_control_hit_rate": float(ct[mask].mean()),
                    "paired_risk_difference": float(ev[mask].mean() - ct[mask].mean()),
                })
    summary = pd.DataFrame(rows)
    p = summary.mcnemar_exact_p_unadjusted.to_numpy(float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(ranked)
    q[order] = np.minimum(ranked, 1.0)
    summary["q_bh_all_rule_lead_tests"] = q
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "tsmc_2330_trader_rule_event_control_summary.csv", index=False)
    pd.DataFrame(year_rows).to_csv(out / "tsmc_2330_trader_rule_year_stability.csv", index=False)
    return summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshots", required=True)
    p.add_argument("--output", default="artifacts/trader_rule_audit")
    p.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    a = p.parse_args()
    run(a.snapshots, a.output, a.bootstraps)

if __name__ == "__main__":
    main()
