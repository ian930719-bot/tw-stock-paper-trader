"""Annual expanding OOF for causal Donchian wave-continuation re-entry.

The experiment adds a distinct breakout continuation family after a confirmed
high-signal exit. Signals are computed at close t and filled next session open.
The 2010-2026 history has been inspected in prior work; this is exploratory
expanding OOF, not pristine blind OOS. Existing research artifacts are untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_4019_replication import (
    INITIAL_CAPITAL, SLIPPAGE_BPS, annual_metrics, build_candidate_signals,
    inputs, simulate,
)

WIN_FLOOR = 0.8695652173913043
MIN_TRAIN_TRADES = 5


def build_breakout_signals(d: pd.DataFrame) -> dict[str, np.ndarray]:
    required = {
        "date", "raw_high", "raw_close", "corporate_action_window",
        "distance_prior_high20", "trend_stack", "adx14",
        "plus_di14", "minus_di14", "distance_ema_20",
        "relative_volume20",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Missing breakout features: {missing}")

    action = d["corporate_action_window"].fillna(True).astype(bool)
    directional = (
        d["trend_stack"].fillna(False).astype(bool)
        & (d["adx14"] >= 18)
        & (d["plus_di14"] > d["minus_di14"])
        & (d["distance_ema_20"] > 0)
        & (d["relative_volume20"] >= 1.0)
        & ~action
    )
    prior_high55 = d["raw_high"].shift(1).rolling(55, min_periods=55).max()
    breakout20 = (
        directional & (d["distance_prior_high20"] > 0)
    ).fillna(False).to_numpy(bool)
    breakout55 = (
        directional & (d["raw_close"] > prior_high55)
    ).fillna(False).to_numpy(bool)
    return {
        "baseline": np.zeros(len(d), dtype=bool),
        "donchian20_reentry": breakout20,
        "donchian55_reentry": breakout55,
    }


def eligible(m: dict) -> bool:
    win = m.get("win_rate_completed_only")
    return (
        int(m.get("completed_trades", 0)) >= MIN_TRAIN_TRADES
        and win is not None
        and np.isfinite(win)
        and win >= WIN_FLOOR
        and m.get("cagr_mark_to_market") is not None
        and np.isfinite(m["cagr_mark_to_market"])
    )


def choose_policy(metrics: dict[str, dict]) -> str:
    pool = [(name, m) for name, m in metrics.items() if eligible(m)]
    if not pool:
        return "baseline"
    return max(
        pool,
        key=lambda item: (
            item[1]["cagr_mark_to_market"],
            item[1]["win_rate_completed_only"],
            -abs(item[1].get("max_drawdown") or -1),
        ),
    )[0]


def run(features_path: str, limits_path: str, dividends_path: str, output_dir: str) -> dict:
    d, limits, dividends = inputs(features_path, limits_path, dividends_path)
    d = d.sort_values("date").reset_index(drop=True)
    low, high, risk = build_candidate_signals(d)
    breakout = build_breakout_signals(d)
    strong_hold = (
        (d["distance_sma_200"] >= 0.25)
        & (d["distance_ema_20"] > 0)
        & d["rsi14"].between(50, 65)
        & (d["plus_di14"] > d["minus_di14"])
        & ~d["corporate_action_window"].fillna(True).astype(bool)
    ).fillna(False).to_numpy(bool)

    policies = {
        "baseline": (np.zeros(len(d), dtype=bool), np.zeros(len(d), dtype=bool)),
        "high_hold_only": (strong_hold, np.zeros(len(d), dtype=bool)),
        "donchian20_reentry": (np.zeros(len(d), dtype=bool), breakout["donchian20_reentry"]),
        "donchian55_reentry": (np.zeros(len(d), dtype=bool), breakout["donchian55_reentry"]),
        "high_hold_plus_donchian20": (strong_hold, breakout["donchian20_reentry"]),
        "high_hold_plus_donchian55": (strong_hold, breakout["donchian55_reentry"]),
    }
    years = sorted(int(y) for y in d["date"].dt.year.unique())
    if 2015 not in years:
        raise RuntimeError("2015 validation start is missing")
    selections: dict[int, str] = {}
    folds = []
    for year in [y for y in years if y >= 2015]:
        prior = np.flatnonzero((d["date"].dt.year < year).to_numpy())
        if not len(prior):
            selections[year] = "baseline"
            continue
        end_i = int(prior[-1])
        train_metrics = {}
        for name, (hold_mask, reentry_mask) in policies.items():
            sim = simulate(
                d, low, high, risk, limits, dividends, 10, 0, end_i,
                high_hold_signal=hold_mask,
                reentry_signal=reentry_mask,
                reentry_cooldown=5,
            )
            train_metrics[name] = {
                k: v for k, v in sim.items()
                if k not in {"trades", "equity_curve"}
            }
        selected = choose_policy(train_metrics)
        selections[year] = selected
        folds.append({
            "validation_year": year,
            "training_start": d["date"].iloc[0].date().isoformat(),
            "training_end": d["date"].iloc[end_i].date().isoformat(),
            "selected_policy": selected,
            "frozen_before_validation_year": True,
            "training_metrics": train_metrics,
        })

    oof_hold = np.zeros(len(d), dtype=bool)
    oof_reentry = np.zeros(len(d), dtype=bool)
    for i, row in d.iterrows():
        year = int(row["date"].year)
        selected = selections.get(year, "baseline")
        hold_mask, reentry_mask = policies[selected]
        oof_hold[i] = bool(hold_mask[i])
        oof_reentry[i] = bool(reentry_mask[i])

    start_i = int(np.flatnonzero((d["date"].dt.year >= 2015).to_numpy())[0])
    end_i = len(d) - 1
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_by_cost, annual_by_cost = {}, {}
    for slip in SLIPPAGE_BPS:
        sim = simulate(
            d, low, high, risk, limits, dividends, slip, start_i, end_i,
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
        "classification": "EXPLORATORY_EXPANDING_ANNUAL_OOF_DONCHIAN_REENTRY",
        "data_period": [
            d["date"].min().date().isoformat(),
            d["date"].max().date().isoformat(),
        ],
        "validation_period": [
            d["date"].iloc[start_i].date().isoformat(),
            d["date"].iloc[end_i].date().isoformat(),
        ],
        "signal_timing": "close t signal; next available session open execution",
        "candidate_family": (
            "After a high-confirm exit, allow one-position all-in re-entry after "
            "a 20/55-session Donchian breakout, only with trend_stack, ADX>=18, "
            "+DI>-DI, close above EMA20, relative_volume20>=1, and no corporate-action window."
        ),
        "candidate_policies": list(policies),
        "annual_selection": (
            "Each validation year selects from the six frozen policies using "
            "only data through the previous year; maximize training 10bps CAGR "
            "subject to at least five completed trades and >=86.9565% win rate; "
            "fallback baseline."
        ),
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
            "The Donchian family is a new exploratory family and remains subject to research-family selection bias.",
            "Daily OHLC cannot reproduce queue priority or true intraday fills.",
            "2026 is a partial year.",
        ],
    }
    (out / "tsmc_2330_donchian_reentry_annual_oof.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    pd.DataFrame([
        {"year": year, "selected_policy": policy}
        for year, policy in selections.items()
    ]).to_csv(out / "selected_policy_by_year.csv", index=False)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--limits", required=True)
    p.add_argument("--dividends", required=True)
    p.add_argument("--output", default="artifacts/tsmc_2330_donchian_reentry_annual_oof")
    a = p.parse_args()
    result = run(a.features, a.limits, a.dividends, a.output)
    print(json.dumps({
        k: v for k, v in result.items()
        if k not in {"annual_frozen_selections"}
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
