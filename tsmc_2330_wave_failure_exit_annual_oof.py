"""Causal wave-structure failure exits with annual expanding OOF selection.

New family: identify a failed swing structure / medium-term trend deterioration
while holding TSMC 2330, and exit at the next session open. It does not alter
the entry rule. Policies are selected using only the prefix before each year.
All earlier research artifacts and the base simulator remain untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_4019_replication import (
    INITIAL_CAPITAL,
    SLIPPAGE_BPS,
    annual_metrics,
    build_candidate_signals,
    inputs,
    simulate,
)

WIN_FLOOR = 0.8695652173913043
MIN_TRAIN_TRADES = 5
VALIDATION_START_YEAR = 2015


def build_wave_failure_signals(d: pd.DataFrame) -> dict[str, np.ndarray]:
    required = {
        "raw_close", "raw_low", "corporate_action_window",
        "distance_prior_low20", "distance_sma_50", "distance_ema_50",
        "ema50_slope_10", "ema20_slope_5", "rsi14",
        "atr_pct14", "close_location",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Missing causal wave-failure features: {missing}")

    action = d["corporate_action_window"].fillna(True).astype(bool)
    # Structural break: close falls below the prior 20-session swing low
    # while the 50-session trend is already deteriorating.
    structure_break = (
        (d["distance_prior_low20"] < 0)
        & (d["distance_sma_50"] < 0)
        & (d["ema50_slope_10"] < 0)
        & (d["rsi14"] < 50)
    )
    # Medium-term trend failure: close below EMA50, declining EMA50,
    # and momentum below the neutral zone.
    ema50_failure = (
        (d["distance_ema_50"] < 0)
        & (d["ema50_slope_10"] < 0)
        & (d["rsi14"] < 45)
    )
    # Two-close EMA20 breakdown to avoid reacting to a single intraday wick.
    below_ema20 = d["distance_ema_20"] < 0
    two_close_ema20_failure = (
        below_ema20
        & below_ema20.shift(1).fillna(False)
        & (d["ema20_slope_5"] < 0)
        & (d["rsi14"] < 45)
    )
    # Capitulation bar in a weak intermediate trend; ATR-scaled rather than
    # a fixed price/percentage stop.
    atr_shock = (
        (d["raw_close"].pct_change() <= -2.0 * d["atr_pct14"])
        & (d["close_location"] <= 0.25)
        & (d["distance_sma_50"] < 0)
    )
    masks = {
        "structure_break": structure_break,
        "ema50_failure": ema50_failure,
        "two_close_ema20_failure": two_close_ema20_failure,
        "atr_shock_weak_trend": atr_shock,
    }
    return {
        name: (signal.fillna(False).to_numpy(bool) & ~action.to_numpy(bool))
        for name, signal in masks.items()
    }


def is_eligible(m: dict) -> bool:
    win = m.get("win_rate_completed_only")
    cagr = m.get("cagr_mark_to_market")
    return (
        int(m.get("completed_trades", 0)) >= MIN_TRAIN_TRADES
        and win is not None and np.isfinite(win) and win >= WIN_FLOOR
        and cagr is not None and np.isfinite(cagr)
    )


def choose_policy(metrics: dict[str, dict]) -> str:
    eligible = [(name, m) for name, m in metrics.items() if is_eligible(m)]
    if not eligible:
        return "baseline"
    return max(
        eligible,
        key=lambda item: (
            item[1]["cagr_mark_to_market"],
            item[1]["win_rate_completed_only"],
            -abs(item[1].get("max_drawdown") or -1),
        ),
    )[0]


def run(features_path: str, limits_path: str, dividends_path: str, output_dir: str) -> dict:
    d, limits, dividends = inputs(features_path, limits_path, dividends_path)
    d = d.sort_values("date").reset_index(drop=True)
    low, high, base_risk = build_candidate_signals(d)
    failure = build_wave_failure_signals(d)

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
        "baseline": (zero, base_risk),
        "high_hold_only": (strong_hold, base_risk),
    }
    for name, mask in failure.items():
        policies[name] = (zero, base_risk | mask)
        policies[f"high_hold_plus_{name}"] = (strong_hold, base_risk | mask)

    years = sorted(int(y) for y in d["date"].dt.year.unique())
    if VALIDATION_START_YEAR not in years:
        raise RuntimeError("2015 validation start year is missing")

    selections: dict[int, str] = {}
    folds = []
    for year in [y for y in years if y >= VALIDATION_START_YEAR]:
        prior = np.flatnonzero((d["date"].dt.year < year).to_numpy())
        if not len(prior):
            selections[year] = "baseline"
            continue
        train_end = int(prior[-1])
        train_metrics = {}
        for name, (hold_mask, risk_mask) in policies.items():
            sim = simulate(
                d, low, high, risk_mask, limits, dividends,
                10, 0, train_end, high_hold_signal=hold_mask,
            )
            train_metrics[name] = {
                k: v for k, v in sim.items()
                if k not in {"trades", "equity_curve"}
            }
        chosen = choose_policy(train_metrics)
        selections[year] = chosen
        folds.append({
            "validation_year": year,
            "training_start": d["date"].iloc[0].date().isoformat(),
            "training_end": d["date"].iloc[train_end].date().isoformat(),
            "selected_policy": chosen,
            "frozen_before_validation_year": True,
            "training_metrics": train_metrics,
        })

    oof_hold = zero.copy()
    oof_risk = base_risk.copy()
    for i, row in d.iterrows():
        year = int(row["date"].year)
        selected = selections.get(year, "baseline")
        hold_mask, risk_mask = policies[selected]
        oof_hold[i] = hold_mask[i]
        oof_risk[i] = risk_mask[i]

    start_idx = int(np.flatnonzero((d["date"].dt.year >= VALIDATION_START_YEAR).to_numpy())[0])
    end_idx = len(d) - 1
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_by_cost, annual_by_cost = {}, {}
    for slip in SLIPPAGE_BPS:
        sim = simulate(
            d, low, high, oof_risk, limits, dividends,
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
        "classification": "EXPLORATORY_EXPANDING_ANNUAL_OOF_WAVE_STRUCTURE_FAILURE_EXIT",
        "data_period": [
            d["date"].min().date().isoformat(),
            d["date"].max().date().isoformat(),
        ],
        "validation_period": [
            d["date"].iloc[start_idx].date().isoformat(),
            d["date"].iloc[end_idx].date().isoformat(),
        ],
        "signal_timing": "causal signal at close t; exit at next available session open",
        "entry_rule": "unchanged existing low-signal entry; long-only one position all-in/all-out",
        "new_exit_family": {
            "structure_break": "close below prior 20-session low, below SMA50, falling EMA50, RSI<50",
            "ema50_failure": "close below EMA50, falling EMA50, RSI<45",
            "two_close_ema20_failure": "two consecutive closes below EMA20, falling EMA20, RSI<45",
            "atr_shock_weak_trend": "one-day return <= -2*ATR%, close location<=0.25, below SMA50",
        },
        "annual_selection": (
            "Each validation year selects among baseline, high-hold, and the "
            "four new failure-exit variants (with/without high-hold) using only "
            "data through prior year. Maximize 10bps training CAGR subject to "
            "at least five completed trades and win rate >=86.9565%; fallback baseline."
        ),
        "minimum_training_trades": MIN_TRAIN_TRADES,
        "win_rate_floor": WIN_FLOOR,
        "risk_exit_cooldown_sessions": 70,
        "selected_policy_by_year": {str(k): v for k, v in selections.items()},
        "annual_frozen_selections": folds,
        "continuous_oof_metrics_by_cost": metrics_by_cost,
        "annual_oof_metrics_by_cost": annual_by_cost,
        "target_status": "UNMET unless CAGR>=50%, win rate>=86.9565%; MDD<=25% tracked separately.",
        "limitations": [
            "2010-2026 historical data has been inspected in previous research; not pristine blind OOS.",
            "This is a new exploratory family; expanding OOF controls temporal leakage but not research-family selection bias.",
            "Daily OHLC does not model queue priority or exact intraday fills.",
            "2026 is a partial year.",
        ],
    }
    (out / "tsmc_2330_wave_failure_exit_annual_oof.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    pd.DataFrame([
        {"year": year, "selected_policy": policy}
        for year, policy in selections.items()
    ]).to_csv(out / "selected_policy_by_year.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--limits", required=True)
    parser.add_argument("--dividends", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_wave_failure_exit_annual_oof")
    args = parser.parse_args()
    result = run(args.features, args.limits, args.dividends, args.output)
    print(json.dumps({
        k: v for k, v in result.items()
        if k != "annual_frozen_selections"
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
