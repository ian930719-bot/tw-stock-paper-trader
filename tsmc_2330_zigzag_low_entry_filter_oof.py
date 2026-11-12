"""Annual OOF comparison of causal ZigZag-low entry filters for TSMC 2330.

Candidates: unchanged baseline, pivot-low-only entries, base-low OR pivot-low,
and base-low entries confirmed by a causal pivot low within the prior 10 sessions.
All pivot signals occur on confirmation close and execute at next session open.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_4019_replication import (
    INITIAL_CAPITAL, SLIPPAGE_BPS, annual_metrics,
    build_candidate_signals, inputs, simulate,
)
from tsmc_2330_causal_zigzag_wave_oof import ATR_MULTIPLIERS, causal_atr_zigzag

WIN_FLOOR = 0.8695652173913043
MIN_TRAIN_TRADES = 5
VALIDATION_START_YEAR = 2015
CONFLUENCE_LOOKBACK = 10


def eligible(m: dict) -> bool:
    win, cagr = m.get("win_rate_completed_only"), m.get("cagr_mark_to_market")
    return (
        int(m.get("completed_trades", 0)) >= MIN_TRAIN_TRADES
        and win is not None and np.isfinite(win) and win >= WIN_FLOOR
        and cagr is not None and np.isfinite(cagr)
    )


def choose_policy(metrics: dict[str, dict]) -> str:
    pool = [(name, m) for name, m in metrics.items() if eligible(m)]
    if not pool:
        return "baseline"
    return max(pool, key=lambda x: (
        x[1]["cagr_mark_to_market"], x[1]["win_rate_completed_only"],
        -abs(x[1].get("max_drawdown") or -1),
    ))[0]


def causal_recent_base_low(base_low: np.ndarray, pivot_low: np.ndarray,
                           lookback: int) -> np.ndarray:
    """Pivot confirmation today, with a base-low event today or in prior N bars."""
    out = np.zeros(len(base_low), dtype=bool)
    for i in range(len(base_low)):
        if pivot_low[i]:
            start = max(0, i - lookback)
            out[i] = bool(base_low[start:i + 1].any())
    return out


def run(features_path: str, limits_path: str, dividends_path: str, output_dir: str) -> dict:
    d, limits, dividends = inputs(features_path, limits_path, dividends_path)
    d = d.sort_values("date").reset_index(drop=True)
    base_low, base_high, risk = build_candidate_signals(d)
    action = d["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    strong_hold = (
        (d["distance_sma_200"] >= 0.25)
        & (d["distance_ema_20"] > 0)
        & d["rsi14"].between(50, 65)
        & (d["plus_di14"] > d["minus_di14"])
        & ~d["corporate_action_window"].fillna(True).astype(bool)
    ).fillna(False).to_numpy(bool)
    zero = np.zeros(len(d), dtype=bool)

    policies: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "baseline": (base_low, zero),
        "high_hold_only": (base_low, strong_hold),
    }
    candidate_counts = {}
    for mult in ATR_MULTIPLIERS:
        pivot_low, _, events = causal_atr_zigzag(d, mult)
        pivot_low &= ~action
        key = f"atr_{mult:.1f}"
        variants = {
            f"{key}_pivot_only": pivot_low,
            f"{key}_base_or_pivot": base_low | pivot_low,
            f"{key}_base_pivot_confluence_{CONFLUENCE_LOOKBACK}d":
                causal_recent_base_low(base_low, pivot_low, CONFLUENCE_LOOKBACK),
        }
        candidate_counts[key] = {
            "confirmed_pivot_low_signals": int(pivot_low.sum()),
            "base_pivot_confluence_signals": int(
                variants[f"{key}_base_pivot_confluence_{CONFLUENCE_LOOKBACK}d"].sum()
            ),
        }
        for name, low_signal in variants.items():
            policies[name] = (low_signal, zero)
            policies[f"{name}_high_hold"] = (low_signal, strong_hold)

    years = sorted(int(y) for y in d["date"].dt.year.unique())
    if VALIDATION_START_YEAR not in years:
        raise RuntimeError("2015 validation start year is missing")
    selections, folds = {}, []
    for year in [y for y in years if y >= VALIDATION_START_YEAR]:
        prior = np.flatnonzero((d["date"].dt.year < year).to_numpy())
        if not len(prior):
            selections[year] = "baseline"
            continue
        train_end = int(prior[-1])
        training = {}
        for name, (low_sig, hold_sig) in policies.items():
            sim = simulate(
                d, low_sig, base_high, risk, limits, dividends,
                10, 0, train_end, high_hold_signal=hold_sig,
            )
            training[name] = {
                k: v for k, v in sim.items()
                if k not in {"trades", "equity_curve"}
            }
        chosen = choose_policy(training)
        selections[year] = chosen
        folds.append({
            "validation_year": year,
            "training_start": d["date"].iloc[0].date().isoformat(),
            "training_end": d["date"].iloc[train_end].date().isoformat(),
            "selected_policy": chosen,
            "frozen_before_validation_year": True,
            "training_metrics": training,
        })

    oof_low, oof_hold = zero.copy(), zero.copy()
    for i, row in d.iterrows():
        low_sig, hold_sig = policies[selections.get(int(row["date"].year), "baseline")]
        oof_low[i], oof_hold[i] = low_sig[i], hold_sig[i]

    start_idx = int(np.flatnonzero((d["date"].dt.year >= VALIDATION_START_YEAR).to_numpy())[0])
    end_idx = len(d) - 1
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_by_cost, annual_by_cost = {}, {}
    for slip in SLIPPAGE_BPS:
        sim = simulate(
            d, oof_low, base_high, risk, limits, dividends,
            slip, start_idx, end_idx, high_hold_signal=oof_hold,
        )
        metrics_by_cost[f"{slip}bps"] = {
            k: v for k, v in sim.items()
            if k not in {"trades", "equity_curve"}
        }
        annual_by_cost[f"{slip}bps"] = annual_metrics(
            sim["equity_curve"], sim["trades"], INITIAL_CAPITAL
        )
        pd.DataFrame(sim["trades"]).to_csv(out / f"trades_{slip}bps.csv", index=False)
        pd.DataFrame(sim["equity_curve"]).to_csv(out / f"equity_{slip}bps.csv", index=False)
        pd.DataFrame(annual_by_cost[f"{slip}bps"]).to_csv(out / f"annual_{slip}bps.csv", index=False)

    result = {
        "symbol": "2330",
        "classification": "EXPLORATORY_CAUSAL_ZIGZAG_LOW_ENTRY_FILTER_ANNUAL_OOF",
        "data_period": [d.date.min().date().isoformat(), d.date.max().date().isoformat()],
        "validation_period": [d.date.iloc[start_idx].date().isoformat(), d.date.iloc[end_idx].date().isoformat()],
        "signal_timing": "pivot extreme is diagnostic only; confirmed pivot low at confirmation close; next available open execution",
        "candidate_definitions": {
            "pivot_only": "Replace original base low entry with causally confirmed ATR ZigZag low",
            "base_or_pivot": "Accept either original base low or causally confirmed ZigZag low",
            "base_pivot_confluence": f"Accept a ZigZag low confirmation only if a base-low signal occurred today or within the prior {CONFLUENCE_LOOKBACK} sessions",
        },
        "atr_multipliers": list(ATR_MULTIPLIERS),
        "candidate_signal_counts": candidate_counts,
        "annual_selection": "Each validation year selects among baseline, high-hold, and twelve ZigZag low entry variants (with/without high-hold) using only data through prior year; maximize 10bps training CAGR subject to >=5 completed trades and >=86.9565% win rate; fallback baseline.",
        "minimum_training_trades": MIN_TRAIN_TRADES,
        "win_rate_floor": WIN_FLOOR,
        "selected_policy_by_year": {str(k): v for k, v in selections.items()},
        "annual_frozen_selections": folds,
        "continuous_oof_metrics_by_cost": metrics_by_cost,
        "annual_oof_metrics_by_cost": annual_by_cost,
        "target_status": "UNMET unless CAGR>=50% and win rate>=86.9565%; MDD<=25% tracked separately.",
        "limitations": [
            "2010-2026 history has been inspected in prior research; not pristine blind OOS.",
            "Annual OOF controls temporal fitting but not research-family selection bias.",
            "Daily OHLC cannot model queue priority or exact intraday fills.",
            "2026 is a partial year.",
        ],
    }
    (out / "tsmc_2330_zigzag_low_entry_filter_oof.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    pd.DataFrame([{"year": y, "selected_policy": p} for y, p in selections.items()]).to_csv(
        out / "selected_policy_by_year.csv", index=False
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--limits", required=True)
    p.add_argument("--dividends", required=True)
    p.add_argument("--output", default="artifacts/tsmc_2330_zigzag_low_entry_filter_oof")
    a = p.parse_args()
    result = run(a.features, a.limits, a.dividends, a.output)
    print(json.dumps({k: v for k, v in result.items() if k != "annual_frozen_selections"},
                     ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
