"""Annual expanding OOF for post-exit pullback-recovery re-entry.

This is distinct from EMA20 cross-reclaim and Donchian breakout: it requires a
recent EMA20 pullback followed by a close above the prior session's high, or a
higher-low / short-range breakout in an established trend. It only re-enters
after an existing high-confirm exit. Signals at close t fill next session open.
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

WIN_FLOOR = 0.8695652173913043
MIN_TRAIN_TRADES = 5
VALIDATION_START_YEAR = 2015


def build_pullback_reentry_signals(d: pd.DataFrame) -> dict[str, np.ndarray]:
    required = {
        "raw_open", "raw_high", "raw_low", "raw_close",
        "corporate_action_window", "distance_ema_20", "distance_sma_50",
        "ema20_slope_5", "sma50_slope_10", "rsi14", "adx14",
        "plus_di14", "minus_di14", "relative_volume20",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Missing pullback-reentry features: {missing}")

    action = d["corporate_action_window"].fillna(True).astype(bool)
    trend = (
        (d["distance_sma_50"] > 0)
        & (d["ema20_slope_5"] > 0)
        & (d["sma50_slope_10"] > 0)
        & (d["adx14"] >= 18)
        & (d["plus_di14"] > d["minus_di14"])
        & d["rsi14"].between(45, 68)
        & (d["relative_volume20"] >= 0.9)
        & ~action
    )
    recent_pullback = (
        d["distance_ema_20"].shift(1).rolling(5, min_periods=5).min() <= 0
    )
    pullback_recovery = (
        trend & recent_pullback
        & (d["distance_ema_20"] > 0)
        & (d["raw_close"] > d["raw_high"].shift(1))
    )

    prior_low5 = d["raw_low"].shift(1).rolling(5, min_periods=5).min()
    prior_high3 = d["raw_high"].shift(1).rolling(3, min_periods=3).max()
    higher_low_break = (
        trend
        & (d["distance_ema_20"] > 0)
        & (d["raw_low"] > prior_low5)
        & (d["raw_close"] > prior_high3)
    )
    return {
        "pullback_recovery": pullback_recovery.fillna(False).to_numpy(bool),
        "higher_low_break": higher_low_break.fillna(False).to_numpy(bool),
    }


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


def run(features_path: str, limits_path: str, dividends_path: str, output_dir: str) -> dict:
    d, limits, dividends = inputs(features_path, limits_path, dividends_path)
    d = d.sort_values("date").reset_index(drop=True)
    low, high, risk = build_candidate_signals(d)
    signals = build_pullback_reentry_signals(d)
    zero = np.zeros(len(d), dtype=bool)
    strong_hold = (
        (d["distance_sma_200"] >= 0.25)
        & (d["distance_ema_20"] > 0)
        & d["rsi14"].between(50, 65)
        & (d["plus_di14"] > d["minus_di14"])
        & ~d["corporate_action_window"].fillna(True).astype(bool)
    ).fillna(False).to_numpy(bool)

    policies = {
        "baseline": (zero, zero),
        "high_hold_only": (strong_hold, zero),
        "pullback_recovery": (zero, signals["pullback_recovery"]),
        "higher_low_break": (zero, signals["higher_low_break"]),
        "high_hold_plus_pullback_recovery": (strong_hold, signals["pullback_recovery"]),
        "high_hold_plus_higher_low_break": (strong_hold, signals["higher_low_break"]),
    }
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
        for name, (hold_mask, reentry_mask) in policies.items():
            sim = simulate(
                d, low, high, risk, limits, dividends, 10, 0, train_end,
                high_hold_signal=hold_mask,
                reentry_signal=reentry_mask,
                reentry_cooldown=5,
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

    oof_hold, oof_reentry = zero.copy(), zero.copy()
    for i, row in d.iterrows():
        hold_mask, reentry_mask = policies[selections.get(int(row.date.year), "baseline")]
        oof_hold[i], oof_reentry[i] = hold_mask[i], reentry_mask[i]

    start_idx = int(np.flatnonzero((d["date"].dt.year >= VALIDATION_START_YEAR).to_numpy())[0])
    end_idx = len(d) - 1
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_by_cost, annual_by_cost = {}, {}
    for slip in SLIPPAGE_BPS:
        sim = simulate(
            d, low, high, risk, limits, dividends, slip, start_idx, end_idx,
            high_hold_signal=oof_hold,
            reentry_signal=oof_reentry,
            reentry_cooldown=5,
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
        "classification": "EXPLORATORY_EXPANDING_ANNUAL_OOF_PULLBACK_RECOVERY_REENTRY",
        "data_period": [d.date.min().date().isoformat(), d.date.max().date().isoformat()],
        "validation_period": [d.date.iloc[start_idx].date().isoformat(), d.date.iloc[end_idx].date().isoformat()],
        "signal_timing": "close t signal; next available session open execution",
        "entry_rule": "existing low-signal entry unchanged; additional entries only after high-confirm exit",
        "candidate_family": {
            "pullback_recovery": "within prior five sessions price closed at/below EMA20; current close reclaims EMA20 and exceeds prior high; positive EMA20/SMA50 slopes, ADX>=18, +DI>-DI, RSI 45-68, relative volume>=0.9",
            "higher_low_break": "current low above prior five-session low, close above prior three-session high and EMA20; same trend gate",
        },
        "annual_selection": "Select among six frozen policies using only data through prior year; maximize 10bps training CAGR subject to >=5 completed trades and >=86.9565% win rate; fallback baseline.",
        "win_rate_floor": WIN_FLOOR,
        "minimum_training_trades": MIN_TRAIN_TRADES,
        "reentry_cooldown_sessions": 5,
        "selected_policy_by_year": {str(k): v for k, v in selections.items()},
        "annual_frozen_selections": folds,
        "continuous_oof_metrics_by_cost": metrics_by_cost,
        "annual_oof_metrics_by_cost": annual_by_cost,
        "target_status": "UNMET unless CAGR>=50% and win rate>=86.9565%; MDD<=25% tracked separately.",
        "limitations": [
            "2010-2026 history has been inspected in prior research; this is not pristine blind OOS.",
            "New family remains exploratory and subject to research-family selection bias.",
            "Daily OHLC cannot model queue priority or exact intraday fills.",
            "2026 is a partial year.",
        ],
    }
    (out / "tsmc_2330_pullback_reentry_annual_oof.json").write_text(
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
    p.add_argument("--output", default="artifacts/tsmc_2330_pullback_reentry_annual_oof")
    a = p.parse_args()
    result = run(a.features, a.limits, a.dividends, a.output)
    print(json.dumps({k: v for k, v in result.items() if k != "annual_frozen_selections"},
                     ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
