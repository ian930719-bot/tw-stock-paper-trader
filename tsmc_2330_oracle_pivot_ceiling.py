"""Perfect-foresight pivot-to-pivot upper-bound study for TSMC 2330.

This is deliberately an oracle benchmark, NOT a tradable strategy: it assumes
buying exactly at each retrospective swing low and selling exactly at the next
retrospective swing high. It quantifies the maximum return available from the
selected pivot definitions before causal signal delay and fill uncertainty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_swing_research import detect_confirmed_pivots

MIN_REVERSALS = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.08)
ATR_MULTIPLES = (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
SLIPPAGE_BPS = (0, 5, 10, 20, 30)
TRAIN_PERIOD = ("2010-01-01", "2016-12-31")
VALIDATION_PERIOD = ("2017-01-01", "2021-12-31")
HOLDOUT_PERIOD = ("2022-01-01", "2026-12-31")
MIN_TRAIN_SWINGS = 10
COMMISSION_RATE = 0.001425
SELL_TAX_RATE = 0.003


def pair_pivots(events: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    action = features["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    pairs = []
    armed_low = None
    for event in events.sort_values("pivot_index").itertuples(index=False):
        pivot_i = int(event.pivot_index)
        confirm_i = int(event.confirmation_index)
        if action[pivot_i] or action[confirm_i]:
            continue
        if event.type == "LOW":
            armed_low = event
        elif event.type == "HIGH" and armed_low is not None:
            low_price = float(armed_low.pivot_price)
            high_price = float(event.pivot_price)
            if low_price > 0 and high_price > low_price:
                pairs.append({
                    "low_date": armed_low.pivot_date,
                    "high_date": event.pivot_date,
                    "low_price_adjusted": low_price,
                    "high_price_adjusted": high_price,
                    "gross_return": high_price / low_price - 1,
                    "holding_calendar_days": (
                        pd.Timestamp(event.pivot_date) - pd.Timestamp(armed_low.pivot_date)
                    ).days,
                    "low_confirmation_lag": int(armed_low.confirmation_lag),
                    "high_confirmation_lag": int(event.confirmation_lag),
                })
            armed_low = None
    return pd.DataFrame(pairs)


def net_factor(gross_return: float, slippage_bps: int) -> float:
    slip = slippage_bps / 10_000.0
    buy_cost = 1 + slip + COMMISSION_RATE
    sell_proceeds = (1 + gross_return) * (1 - slip) * (1 - COMMISSION_RATE - SELL_TAX_RATE)
    return sell_proceeds / buy_cost


def summarize(pairs: pd.DataFrame, features: pd.DataFrame) -> dict:
    if pairs.empty:
        return {"swings": 0}
    start = pd.Timestamp(pairs["low_date"].iloc[0])
    end = pd.Timestamp(pairs["high_date"].iloc[-1])
    years = (end - start).days / 365.25
    gross_factors = 1 + pairs["gross_return"].to_numpy()
    gross_multiple = float(np.prod(gross_factors))
    gross_cagr = gross_multiple ** (1 / years) - 1 if years > 0 else None
    out = {
        "swings": int(len(pairs)),
        "first_low": start.date().isoformat(),
        "last_high": end.date().isoformat(),
        "calendar_years": years,
        "gross_compounded_multiple": gross_multiple,
        "gross_oracle_cagr": gross_cagr,
        "gross_mean_swing_return": float(pairs["gross_return"].mean()),
        "gross_median_swing_return": float(pairs["gross_return"].median()),
        "gross_p25_swing_return": float(pairs["gross_return"].quantile(0.25)),
        "gross_p75_swing_return": float(pairs["gross_return"].quantile(0.75)),
        "mean_holding_calendar_days": float(pairs["holding_calendar_days"].mean()),
        "median_low_confirmation_lag": float(pairs["low_confirmation_lag"].median()),
        "median_high_confirmation_lag": float(pairs["high_confirmation_lag"].median()),
    }
    for slip in SLIPPAGE_BPS:
        factors = np.array([net_factor(float(r), slip) for r in pairs["gross_return"]])
        multiple = float(np.prod(factors))
        out[f"net_oracle_cagr_{slip}bps"] = multiple ** (1 / years) - 1 if years > 0 else None
        out[f"net_oracle_multiple_{slip}bps"] = multiple
    first_close = float(features.loc[features["date"] >= start, "close"].iloc[0])
    last_close = float(features.loc[features["date"] <= end, "close"].iloc[-1])
    out["buy_hold_total_return_same_window"] = last_close / first_close - 1
    out["buy_hold_cagr_same_window"] = (
        (last_close / first_close) ** (1 / years) - 1 if years > 0 else None
    )
    return out


def run(features_path: str, output_dir: str) -> dict:
    features = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    required = {"date", "high", "low", "close", "atr_pct14", "corporate_action_window"}
    if not required.issubset(features.columns):
        raise ValueError(f"Features missing required columns: {sorted(required - set(features.columns))}")
    summaries = []
    pair_frames = []
    for min_reversal in MIN_REVERSALS:
        for atr_multiple in ATR_MULTIPLES:
            events = detect_confirmed_pivots(features, min_reversal, atr_multiple)
            pairs = pair_pivots(events, features)
            summary = summarize(pairs, features)
            summary.update({
                "min_reversal_pct": min_reversal,
                "atr_multiple": atr_multiple,
                "events": int(len(events)),
                "high_events": int((events["type"] == "HIGH").sum()) if not events.empty else 0,
                "low_events": int((events["type"] == "LOW").sum()) if not events.empty else 0,
            })
            for period_name, period in (
                ("train", TRAIN_PERIOD),
                ("validation", VALIDATION_PERIOD),
                ("holdout", HOLDOUT_PERIOD),
            ):
                period_events = events.copy()
                if not period_events.empty:
                    pdate = pd.to_datetime(period_events["pivot_date"])
                    cdate = pd.to_datetime(period_events["confirmation_date"])
                    period_events = period_events.loc[
                        (pdate >= pd.Timestamp(period[0]))
                        & (pdate <= pd.Timestamp(period[1]))
                        & (cdate >= pd.Timestamp(period[0]))
                        & (cdate <= pd.Timestamp(period[1]))
                    ].copy()
                period_pairs = pair_pivots(period_events, features)
                period_summary = summarize(period_pairs, features)
                for key in ("swings", "gross_oracle_cagr", "net_oracle_cagr_10bps",
                            "net_oracle_cagr_30bps", "gross_compounded_multiple"):
                    summary[f"{period_name}_{key}"] = period_summary.get(key)
            summaries.append(summary)
            if not pairs.empty:
                pairs = pairs.copy()
                pairs["min_reversal_pct"] = min_reversal
                pairs["atr_multiple"] = atr_multiple
                pair_frames.append(pairs)
    table = pd.DataFrame(summaries)
    table = table.sort_values("net_oracle_cagr_10bps", ascending=False, na_position="last")
    train_candidates = table.loc[
        (pd.to_numeric(table["train_swings"], errors="coerce") >= MIN_TRAIN_SWINGS)
        & pd.to_numeric(table["train_net_oracle_cagr_10bps"], errors="coerce").notna()
    ].sort_values("train_net_oracle_cagr_10bps", ascending=False)
    best_train = train_candidates.iloc[0].to_dict() if not train_candidates.empty else {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "tsmc_2330_oracle_pivot_ceiling.csv", index=False)
    all_pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    all_pairs.to_csv(out / "tsmc_2330_oracle_pivot_pairs.csv", index=False)
    best = table.iloc[0].to_dict() if not table.empty else {}
    summary = {
        "symbol": "2330",
        "data_period": [features["date"].min().date().isoformat(),
                        features["date"].max().date().isoformat()],
        "min_reversal_grid": list(MIN_REVERSALS),
        "atr_multiple_grid": list(ATR_MULTIPLES),
        "combinations": int(len(table)),
        "best_by_full_period_net_oracle_cagr_10bps": best,
        "training_period": list(TRAIN_PERIOD),
        "validation_period": list(VALIDATION_PERIOD),
        "holdout_period": list(HOLDOUT_PERIOD),
        "minimum_training_swings_for_selection": MIN_TRAIN_SWINGS,
        "threshold_selected_using_training_period_only": {
            "min_reversal_pct": best_train.get("min_reversal_pct"),
            "atr_multiple": best_train.get("atr_multiple"),
            "train_swings": best_train.get("train_swings"),
            "train_net_oracle_cagr_10bps": best_train.get("train_net_oracle_cagr_10bps"),
            "validation_swings": best_train.get("validation_swings"),
            "validation_net_oracle_cagr_10bps": best_train.get("validation_net_oracle_cagr_10bps"),
            "holdout_swings": best_train.get("holdout_swings"),
            "holdout_net_oracle_cagr_10bps": best_train.get("holdout_net_oracle_cagr_10bps"),
        },
        "warning": "Perfect-foresight oracle only. Uses retrospective pivot dates/prices, assumes exact fills at every low and next high, ignores causal confirmation delay, queue/limit fills, capital/odd-lot constraints and many real execution frictions. The full-period best is exploratory only; the train-selected threshold is reported separately and its validation/holdout results are not a tradable strategy.",
    }
    (out / "tsmc_2330_oracle_pivot_ceiling.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.output), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
