"""Trend-regime and ATR-risk variants for the frozen TSMC reversal strategy.

Uses only predeclared entry filters and ATR trailing stops. Selects one variant
on 2017-2021 using CAGR - 0.5*|max drawdown|, then evaluates the frozen choice
on 2022+ with slippage sensitivity. No 2022+ selection or tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_execution_backtest import (
    SLIPPAGE_BPS, INITIAL_CAPITAL, load_inputs, _simulate_period,
)
from tsmc_2330_reversal_confirmation_backtest import build_confirmed_signals

VALIDATION = ("2017-01-01", "2021-12-31")
HOLDOUT_START = "2022-01-01"
ENTRY_FILTERS = ("none", "above_sma200", "positive_sma50_slope", "trend_stack")
ATR_STOPS = (None, 2.5, 4.0)
CONFIRMATION = "break_prior_range"
SELECTION_SLIPPAGE_BPS = 10


def apply_entry_filter(features: pd.DataFrame, low_signal: np.ndarray,
                       filter_name: str) -> np.ndarray:
    if filter_name == "none":
        return low_signal.copy()
    if filter_name == "above_sma200":
        condition = pd.to_numeric(features["distance_sma_200"], errors="coerce").gt(0)
    elif filter_name == "positive_sma50_slope":
        condition = pd.to_numeric(features["sma50_slope_10"], errors="coerce").gt(0)
    elif filter_name == "trend_stack":
        condition = pd.to_numeric(features["trend_stack"], errors="coerce").eq(1)
    else:
        raise ValueError(f"Unknown entry filter: {filter_name}")
    return low_signal & condition.fillna(False).to_numpy()


def compact(result: dict) -> dict:
    return {k: result[k] for k in (
        "period_start", "period_end", "initial_capital", "final_equity",
        "total_return", "cagr", "max_drawdown", "trade_count", "win_rate",
        "average_trade_return", "exposure_ratio", "total_cash_dividends",
        "total_commission", "total_sell_tax", "estimated_slippage_cost",
        "blocked_limit_orders", "missing_price_limit_orders",
    )}


def run(features_path: str, rules_path: str, limits_path: str, dividends_path: str,
        output_dir: str, initial_capital: float = INITIAL_CAPITAL) -> dict:
    bars, rules, limits, dividends = load_inputs(
        features_path, rules_path, limits_path, dividends_path
    )
    base_low, base_high = build_confirmed_signals(bars, rules, CONFIRMATION)
    limit_lookup = {
        row.date.strftime("%Y-%m-%d"): (float(row.limit_up), float(row.limit_down))
        for row in limits.itertuples(index=False)
        if np.isfinite(row.limit_up) and np.isfinite(row.limit_down)
    }
    candidates = {}
    for entry_filter in ENTRY_FILTERS:
        low = apply_entry_filter(bars, base_low, entry_filter)
        for stop in ATR_STOPS:
            result = _simulate_period(
                bars, low, base_high, limit_lookup, dividends,
                *VALIDATION, SELECTION_SLIPPAGE_BPS, initial_capital, True,
                atr_stop_multiple=stop,
            )
            score = float(result["cagr"] - 0.5 * abs(result["max_drawdown"])) if result["cagr"] is not None else -1.0
            if result["trade_count"] < 5:
                score = -1.0
            key = f"{entry_filter}_atr_{'none' if stop is None else str(stop).replace('.', 'p')}"
            candidates[key] = {
                "entry_filter": entry_filter, "atr_stop_multiple": stop,
                "selection_score": score, "validation": compact(result),
                "low_signal_days_after_filter": int(low.sum()),
                "high_signal_days": int(base_high.sum()),
            }
    selected = max(candidates, key=lambda k: (
        candidates[k]["selection_score"],
        candidates[k]["validation"]["cagr"] or -1.0,
    ))
    chosen = candidates[selected]
    low = apply_entry_filter(bars, base_low, chosen["entry_filter"])
    holdout = {}
    for slip in SLIPPAGE_BPS:
        result = _simulate_period(
            bars, low, base_high, limit_lookup, dividends,
            HOLDOUT_START, bars["date"].max().date().isoformat(),
            slip, initial_capital, True,
            atr_stop_multiple=chosen["atr_stop_multiple"],
        )
        holdout[f"strategy_{slip}bps"] = result
    benchmark = _simulate_period(
        bars, low, base_high, limit_lookup, dividends,
        HOLDOUT_START, bars["date"].max().date().isoformat(),
        SELECTION_SLIPPAGE_BPS, initial_capital, False,
    )
    summary = {
        "symbol": "2330",
        "base_confirmation": CONFIRMATION,
        "entry_filters": list(ENTRY_FILTERS),
        "atr_stop_candidates": list(ATR_STOPS),
        "selection_period": list(VALIDATION),
        "selection_slippage_bps_per_side": SELECTION_SLIPPAGE_BPS,
        "selection_objective": "validation CAGR - 0.5*abs(max drawdown), minimum five trades",
        "candidate_validation_results": candidates,
        "selected_variant": selected,
        "selected_parameters": {
            "entry_filter": chosen["entry_filter"],
            "atr_stop_multiple": chosen["atr_stop_multiple"],
        },
        "historical_holdout_period": [HOLDOUT_START, bars["date"].max().date().isoformat()],
        "historical_holdout_results": {
            key: compact(value) for key, value in holdout.items()
        },
        "historical_holdout_buy_hold_10bps": compact(benchmark),
        "test_period_used_for_selection": False,
        "holdout_is_pristine": False,
        "warning": "2022+ was examined in previous diagnostics; this is not a pristine blind test. Candidate selection uses only 2017-2021. Forward paper validation remains required.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_trend_atr_risk_research.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([
        {"variant": name, "entry_filter": value["entry_filter"],
         "atr_stop_multiple": value["atr_stop_multiple"],
         "selection_score": value["selection_score"], **value["validation"]}
        for name, value in candidates.items()
    ]).to_csv(out / "tsmc_2330_trend_atr_validation.csv", index=False)
    for key, value in holdout.items():
        pd.DataFrame(value["trades"]).to_csv(out / f"risk_{selected}_{key}_trades.csv", index=False)
        pd.DataFrame(value["equity_curve"]).to_csv(out / f"risk_{selected}_{key}_equity.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--limits", required=True)
    parser.add_argument("--dividends", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.rules, args.limits, args.dividends,
                         args.output, args.initial_capital), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
