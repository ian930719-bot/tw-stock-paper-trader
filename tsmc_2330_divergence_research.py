"""Causal RSI/MACD/volume divergence diagnostics for TSMC 2330 swing pivots.

Divergence rules are fixed in advance. One rule per direction is selected using
2017-2021 only; the 2022+ period is a descriptive historical check and is never
used to select or tune. All features use past/current close data only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_execution_backtest import build_signals

THRESHOLD_ID = "p0.050_atr2.0"
VALID_START = "2017-01-01"
TEST_START = "2022-01-01"
WICK_MIN = 0.005


def target_for(dates: pd.Series, events: pd.DataFrame, horizon: int, kind: str) -> np.ndarray:
    chosen = events.loc[events["type"] == kind]
    lookup = {pd.Timestamp(v).normalize(): i for i, v in enumerate(dates)}
    y = np.zeros(len(dates), dtype=bool)
    for value in pd.to_datetime(chosen["pivot_date"], errors="coerce").dropna():
        idx = lookup.get(pd.Timestamp(value).normalize())
        if idx is not None:
            y[max(0, idx - horizon):idx] = True
    return y


def calc_metrics(signal: np.ndarray, y: np.ndarray) -> dict:
    n = int(signal.sum())
    tp = int((signal & y).sum())
    event_days = int(y.sum())
    precision = tp / n if n else 0.0
    recall = tp / event_days if event_days else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    prevalence = float(y.mean()) if len(y) else 0.0
    return {
        "rows": int(len(y)), "signal_days": n, "true_positive_days": tp,
        "event_days": event_days, "prevalence": prevalence,
        "precision": precision, "recall": recall, "f1": f1,
        "lift": precision / prevalence if prevalence else None,
    }


def candidate_signals(features: pd.DataFrame, base_low: np.ndarray,
                      base_high: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    n = len(features)
    b_rsi = features["bearish_rsi_divergence20"].fillna(0).astype(bool).to_numpy()
    u_rsi = features["bullish_rsi_divergence20"].fillna(0).astype(bool).to_numpy()
    b_macd = features["bearish_macd_divergence20"].fillna(0).astype(bool).to_numpy()
    u_macd = features["bullish_macd_divergence20"].fillna(0).astype(bool).to_numpy()
    b_vol = features["bearish_volume_divergence20"].fillna(0).astype(bool).to_numpy()
    u_vol = features["bullish_volume_divergence20"].fillna(0).astype(bool).to_numpy()
    upper = pd.to_numeric(features["upper_wick_pct"], errors="coerce").fillna(0).to_numpy() >= WICK_MIN
    lower = pd.to_numeric(features["lower_wick_pct"], errors="coerce").fillna(0).to_numpy() >= WICK_MIN
    loc = pd.to_numeric(features["close_location"], errors="coerce")
    weak_close = loc.le(0.5).fillna(False).to_numpy()
    strong_close = loc.ge(0.5).fillna(False).to_numpy()
    return {
        "HIGH": {
            "bearish_rsi": b_rsi,
            "bearish_macd": b_macd,
            "bearish_volume": b_vol,
            "bearish_rsi_and_macd": b_rsi & b_macd,
            "bearish_rsi_upper_wick": b_rsi & upper,
            "bearish_macd_weak_close": b_macd & weak_close,
            "base_high_and_bearish_rsi": base_high & b_rsi,
            "base_high_and_bearish_macd": base_high & b_macd,
        },
        "LOW": {
            "bullish_rsi": u_rsi,
            "bullish_macd": u_macd,
            "bullish_volume": u_vol,
            "bullish_rsi_and_macd": u_rsi & u_macd,
            "bullish_rsi_lower_wick": u_rsi & lower,
            "bullish_macd_strong_close": u_macd & strong_close,
            "base_low_and_bullish_rsi": base_low & u_rsi,
            "base_low_and_bullish_macd": base_low & u_macd,
        },
    }


def run(features_path: str, events_path: str, rules_path: str, output_dir: str) -> dict:
    x = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    events = pd.read_csv(events_path, parse_dates=["pivot_date"])
    required = {
        "bearish_rsi_divergence20", "bullish_rsi_divergence20",
        "bearish_macd_divergence20", "bullish_macd_divergence20",
        "bearish_volume_divergence20", "bullish_volume_divergence20",
        "upper_wick_pct", "lower_wick_pct", "close_location",
    }
    if not required.issubset(x.columns):
        raise ValueError(f"Missing divergence features: {sorted(required - set(x.columns))}")
    events = events.loc[events["threshold_id"] == THRESHOLD_ID].copy()
    if events.empty:
        raise ValueError(f"No event labels for {THRESHOLD_ID}")
    rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    base_low, base_high = build_signals(x, rules)
    candidates = candidate_signals(x, base_low, base_high)

    dates = x["date"]
    valid = x[list(required)].notna().all(axis=1).to_numpy(copy=True)
    if "corporate_action_window" in x:
        valid &= ~x["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    validation = ((dates >= VALID_START) & (dates < TEST_START)).to_numpy() & valid
    test = (dates >= TEST_START).to_numpy() & valid
    valid_idx = np.flatnonzero(validation)
    test_idx = np.flatnonzero(test)
    if not len(valid_idx) or not len(test_idx):
        raise ValueError("Chronological validation/test split incomplete")
    valid_start_idx, test_start_idx = int(valid_idx[0]), int(test_idx[0])
    validation_mask = validation & (np.arange(len(x)) <= max(0, test_start_idx - 6))
    test_mask = test

    selected = {}
    result_rows = []
    for kind in ("HIGH", "LOW"):
        ranked = []
        for name, signal in candidates[kind].items():
            per_horizon = {}
            f1s = []
            eligible = True
            for horizon in (1, 3, 5):
                y = target_for(dates, events, horizon, kind)
                m = calc_metrics(signal[validation_mask], y[validation_mask])
                per_horizon[str(horizon)] = m
                if m["signal_days"] < 5 or m["event_days"] < 5:
                    eligible = False
                f1s.append(m["f1"])
            score = float(np.mean(f1s)) if eligible else -1.0
            ranked.append({
                "name": name, "validation_mean_f1": score,
                "validation_metrics": per_horizon,
            })
        ranked.sort(key=lambda row: (row["validation_mean_f1"], row["name"]), reverse=True)
        best = ranked[0] if ranked and ranked[0]["validation_mean_f1"] >= 0 else None
        if best is None:
            selected[kind] = {"selected": False, "reason": "No candidate met minimum validation sample sizes", "candidates": ranked}
            continue
        name = best["name"]
        signal = candidates[kind][name]
        holdout_metrics = {}
        for horizon in (1, 3, 5):
            y = target_for(dates, events, horizon, kind)
            holdout_metrics[str(horizon)] = calc_metrics(signal[test_mask], y[test_mask])
            result_rows.append({
                "pivot_type": kind, "selected_rule": name,
                "horizon_sessions": horizon,
                "validation_mean_f1": best["validation_mean_f1"],
                **{f"test_{k}": v for k, v in holdout_metrics[str(horizon)].items()},
            })
        selected[kind] = {
            "selected": True, "rule": name,
            "validation_mean_f1": best["validation_mean_f1"],
            "validation_metrics": best["validation_metrics"],
            "historical_holdout_metrics": holdout_metrics,
            "candidates": ranked,
        }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_rows).to_csv(out / "tsmc_2330_divergence_holdout.csv", index=False)
    summary = {
        "symbol": "2330", "threshold_id": THRESHOLD_ID,
        "train_period": ["2010-01-01", "2016-12-31"],
        "validation_period": ["2017-01-01", "2021-12-31"],
        "historical_holdout_period": [dates.loc[test_mask].min().date().isoformat(),
                                      dates.loc[test_mask].max().date().isoformat()],
        "predeclared_wick_min_pct": WICK_MIN,
        "purge_sessions": 5,
        "selected_rules": selected,
        "test_period_used_for_selection": False,
        "holdout_is_pristine": False,
        "strategy_backtest": False,
        "warning": "2022+ was inspected in prior diagnostics; divergence holdout results are descriptive only. No trades or CAGR are simulated.",
    }
    (out / "tsmc_2330_divergence_research.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.events, args.rules, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
