"""Causal ATR ZigZag wave-turn research and expanding annual OOF.

A pivot is timestamped on its actual extreme day, but the signal is emitted
only on the later close that confirms an ATR-sized reversal. Orders therefore
execute no earlier than the next available session open. No pivot is backdated
into the trading signal stream.
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
ATR_MULTIPLIERS = (1.5, 2.0, 2.5, 3.0)
MIN_PIVOT_SEPARATION = 3


def causal_atr_zigzag(d: pd.DataFrame, multiplier: float,
                      min_separation: int = MIN_PIVOT_SEPARATION) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    required = {"date", "raw_high", "raw_low", "raw_close", "atr_pct14"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Missing causal ZigZag columns: {missing}")
    n = len(d)
    low_confirm = np.zeros(n, dtype=bool)
    high_confirm = np.zeros(n, dtype=bool)
    events: list[dict] = []
    close = d["raw_close"].to_numpy(float)
    high = d["raw_high"].to_numpy(float)
    low = d["raw_low"].to_numpy(float)
    atr_pct = d["atr_pct14"].to_numpy(float)
    dates = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d").to_numpy()

    # 1 = seeking a high after a confirmed low; -1 = seeking a low after a high.
    direction = 0
    candidate_high = candidate_low = np.nan
    high_idx = low_idx = -1
    last_pivot_idx = -10**9

    for i in range(n):
        if not (np.isfinite(close[i]) and np.isfinite(high[i]) and
                np.isfinite(low[i]) and np.isfinite(atr_pct[i]) and atr_pct[i] > 0):
            continue
        atr_abs = close[i] * atr_pct[i]

        if direction == 0:
            candidate_high, high_idx = high[i], i
            candidate_low, low_idx = low[i], i
            direction = 1
            continue

        if direction == 1:
            if high[i] > candidate_high:
                candidate_high, high_idx = high[i], i
            pivot_atr = close[high_idx] * atr_pct[high_idx]
            reversal = candidate_high - close[i]
            if (i - high_idx >= min_separation and
                    np.isfinite(pivot_atr) and pivot_atr > 0 and
                    reversal >= multiplier * pivot_atr):
                high_confirm[i] = True
                events.append({
                    "pivot_type": "HIGH",
                    "pivot_date": dates[high_idx],
                    "confirmation_date": dates[i],
                    "pivot_index": int(high_idx),
                    "confirmation_index": int(i),
                    "confirmation_lag_sessions": int(i - high_idx),
                    "pivot_price": float(candidate_high),
                    "pivot_atr": float(pivot_atr),
                    "reversal_at_confirmation": float(reversal / candidate_high),
                    "atr_multiplier": float(multiplier),
                })
                last_pivot_idx = high_idx
                candidate_low, low_idx = low[i], i
                direction = -1
        else:
            if low[i] < candidate_low:
                candidate_low, low_idx = low[i], i
            pivot_atr = close[low_idx] * atr_pct[low_idx]
            reversal = close[i] - candidate_low
            if (i - low_idx >= min_separation and
                    np.isfinite(pivot_atr) and pivot_atr > 0 and
                    reversal >= multiplier * pivot_atr):
                low_confirm[i] = True
                events.append({
                    "pivot_type": "LOW",
                    "pivot_date": dates[low_idx],
                    "confirmation_date": dates[i],
                    "pivot_index": int(low_idx),
                    "confirmation_index": int(i),
                    "confirmation_lag_sessions": int(i - low_idx),
                    "pivot_price": float(candidate_low),
                    "pivot_atr": float(pivot_atr),
                    "reversal_at_confirmation": float(reversal / candidate_low),
                    "atr_multiplier": float(multiplier),
                })
                last_pivot_idx = low_idx
                candidate_high, high_idx = high[i], i
                direction = 1

    return low_confirm, high_confirm, events


def add_forward_labels(d: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    rows = []
    close = d["raw_close"].to_numpy(float)
    op = d["raw_open"].to_numpy(float) if "raw_open" in d.columns else close
    for event in events:
        i = int(event["confirmation_index"])
        row = dict(event)
        for horizon in (20, 40, 60):
            j = i + horizon
            if j < len(d) and np.isfinite(op[i + 1]) and np.isfinite(close[j]):
                row[f"forward_{horizon}d_return_from_next_open"] = float(close[j] / op[i + 1] - 1)
            else:
                row[f"forward_{horizon}d_return_from_next_open"] = None
        rows.append(row)
    return pd.DataFrame(rows)


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
    base_low, base_high, risk = build_candidate_signals(d)

    pivots: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_events = []
    for mult in ATR_MULTIPLIERS:
        low_sig, high_sig, events = causal_atr_zigzag(d, mult)
        key = f"atr_{mult:.1f}"
        pivots[key] = (low_sig, high_sig)
        for e in events:
            all_events.append(e)
    event_frame = add_forward_labels(d, all_events)

    action = d["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    strong_hold = (
        (d["distance_sma_200"] >= 0.25)
        & (d["distance_ema_20"] > 0)
        & d["rsi14"].between(50, 65)
        & (d["plus_di14"] > d["minus_di14"])
        & ~d["corporate_action_window"].fillna(True).astype(bool)
    ).fillna(False).to_numpy(bool)
    zero = np.zeros(len(d), dtype=bool)

    policies: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "baseline": (base_low, base_high, zero),
        "high_hold_only": (base_low, base_high, strong_hold),
    }
    for key, (low_sig, high_sig) in pivots.items():
        valid_low, valid_high = low_sig & ~action, high_sig & ~action
        policies[key] = (valid_low, valid_high, zero)
        policies[f"{key}_high_hold"] = (valid_low, valid_high, strong_hold)

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
        for name, (low_sig, high_sig, hold_sig) in policies.items():
            sim = simulate(
                d, low_sig, high_sig, risk, limits, dividends,
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

    oof_low, oof_high, oof_hold = zero.copy(), zero.copy(), zero.copy()
    for i, row in d.iterrows():
        name = selections.get(int(row["date"].year), "baseline")
        low_sig, high_sig, hold_sig = policies[name]
        oof_low[i], oof_high[i], oof_hold[i] = low_sig[i], high_sig[i], hold_sig[i]

    start_idx = int(np.flatnonzero((d["date"].dt.year >= VALIDATION_START_YEAR).to_numpy())[0])
    end_idx = len(d) - 1
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    event_frame.to_csv(out / "causal_zigzag_pivot_events.csv", index=False)
    metrics_by_cost, annual_by_cost = {}, {}
    for slip in SLIPPAGE_BPS:
        sim = simulate(
            d, oof_low, oof_high, risk, limits, dividends,
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

    pivot_summary = {}
    if not event_frame.empty:
        for key, group in event_frame.groupby(["atr_multiplier", "pivot_type"]):
            k = f"{key[0]:.1f}_{key[1].lower()}"
            pivot_summary[k] = {
                "events": int(len(group)),
                "median_confirmation_lag_sessions": float(group.confirmation_lag_sessions.median()),
                "median_reversal_pct": float(group.reversal_at_confirmation.median()),
                "forward_return_stats": {
                    str(h): {
                        "n": int(group[f"forward_{h}d_return_from_next_open"].notna().sum()),
                        "positive_fraction": float((group[f"forward_{h}d_return_from_next_open"].dropna() > 0).mean()) if group[f"forward_{h}d_return_from_next_open"].notna().any() else None,
                        "median": float(group[f"forward_{h}d_return_from_next_open"].median()) if group[f"forward_{h}d_return_from_next_open"].notna().any() else None,
                    } for h in (20, 40, 60)
                },
            }

    result = {
        "symbol": "2330",
        "classification": "EXPLORATORY_CAUSAL_ATR_ZIGZAG_EXPANDING_ANNUAL_OOF",
        "data_period": [d.date.min().date().isoformat(), d.date.max().date().isoformat()],
        "validation_period": [d.date.iloc[start_idx].date().isoformat(), d.date.iloc[end_idx].date().isoformat()],
        "pivot_definition": "alternating ATR-sized reversals; candidate extreme recorded at its date but signal only emitted on confirmation close after >=3 sessions; execution next available open",
        "atr_multipliers": list(ATR_MULTIPLIERS),
        "minimum_pivot_separation_sessions": MIN_PIVOT_SEPARATION,
        "annual_selection": "Each validation year selects among baseline, high-hold, and causal ZigZag thresholds (with/without high-hold) using only data through prior year; maximize 10bps training CAGR subject to >=5 completed trades and >=86.9565% win rate; fallback baseline.",
        "minimum_training_trades": MIN_TRAIN_TRADES,
        "win_rate_floor": WIN_FLOOR,
        "selected_policy_by_year": {str(k): v for k, v in selections.items()},
        "annual_frozen_selections": folds,
        "pivot_event_summary_descriptive_only": pivot_summary,
        "continuous_oof_metrics_by_cost": metrics_by_cost,
        "annual_oof_metrics_by_cost": annual_by_cost,
        "target_status": "UNMET unless CAGR>=50% and win rate>=86.9565%; MDD<=25% tracked separately.",
        "limitations": [
            "2010-2026 history has been inspected in prior research; this is not pristine blind OOS.",
            "Pivot-event forward returns are descriptive and are not strategy P&L.",
            "Annual OOF controls temporal fitting but not research-family selection bias.",
            "Daily OHLC cannot model queue priority or exact intraday fills.",
            "2026 is a partial year.",
        ],
    }
    (out / "tsmc_2330_causal_zigzag_wave_oof.json").write_text(
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
    p.add_argument("--output", default="artifacts/tsmc_2330_causal_zigzag_wave_oof")
    a = p.parse_args()
    result = run(a.features, a.limits, a.dividends, a.output)
    print(json.dumps({k: v for k, v in result.items() if k != "annual_frozen_selections"},
                     ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
