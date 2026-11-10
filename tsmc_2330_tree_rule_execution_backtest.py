"""Execution-aware test of the TSMC 2330 one-day tree-derived low setup.

The low setup thresholds are frozen from the 2010-2016-trained shallow tree,
with tree depth selected on 2017-2021 validation. The exit is the previously
frozen base-high + bearish MACD divergence rule. Compare direct next-session
entry against waiting up to five sessions for a prior-range break confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_execution_backtest import (
    SLIPPAGE_BPS, INITIAL_CAPITAL, load_inputs, _simulate_period, build_signals,
)
from tsmc_2330_divergence_research import candidate_signals
from tsmc_2330_reversal_confirmation_backtest import build_confirmed_signals

VALIDATION = ("2017-01-01", "2021-12-31")
HOLDOUT_START = "2022-01-01"
LOW_RETURN_MAX = -0.0025693492
LOW_CLOSE_LOCATION_MAX = 0.25657895
MAX_WAIT = 5


def tree_low_setup(features: pd.DataFrame) -> np.ndarray:
    ret = pd.to_numeric(features["return_1"], errors="coerce")
    loc = pd.to_numeric(features["close_location"], errors="coerce")
    signal = (ret.le(LOW_RETURN_MAX) & loc.le(LOW_CLOSE_LOCATION_MAX)).fillna(False).to_numpy(copy=True)
    if "corporate_action_window" in features:
        signal &= ~features["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    return signal


def confirm_low_setup(features: pd.DataFrame, setup: np.ndarray,
                      max_wait: int = MAX_WAIT) -> np.ndarray:
    close = pd.to_numeric(features["close"], errors="coerce")
    prior_high = pd.to_numeric(features["high"], errors="coerce").shift(1)
    loc = pd.to_numeric(features["close_location"], errors="coerce")
    confirm = (close.gt(prior_high) & loc.ge(0.55)).fillna(False).to_numpy()
    out = np.zeros(len(features), dtype=bool)
    armed_until = -1
    for i in range(len(features)):
        if setup[i] and armed_until < i:
            armed_until = i + max_wait
        if armed_until >= i and i > armed_until - max_wait and confirm[i]:
            out[i] = True
            armed_until = -1
        if armed_until < i:
            armed_until = -1
    if "corporate_action_window" in features:
        out &= ~features["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    return out


def compact(result: dict) -> dict:
    keys = (
        "period_start", "period_end", "initial_capital", "final_equity",
        "total_return", "cagr", "max_drawdown", "trade_count", "win_rate",
        "average_trade_return", "exposure_ratio", "total_cash_dividends",
        "total_commission", "total_sell_tax", "estimated_slippage_cost",
        "blocked_limit_orders", "missing_price_limit_orders",
    )
    return {k: result[k] for k in keys}


def run(features_path: str, rules_path: str, limits_path: str,
        dividends_path: str, output_dir: str,
        initial_capital: float = INITIAL_CAPITAL) -> dict:
    bars, rules, limits, dividends = load_inputs(
        features_path, rules_path, limits_path, dividends_path
    )
    setup = tree_low_setup(bars)
    confirmed = confirm_low_setup(bars, setup)
    baseline_low, _ = build_confirmed_signals(bars, rules, "break_prior_range")
    base_low_setup, base_high_setup = build_signals(bars, rules)
    candidates = candidate_signals(bars, base_low_setup, base_high_setup)
    high_signal = candidates["HIGH"]["base_high_and_bearish_macd"]
    limit_lookup = {
        row.date.strftime("%Y-%m-%d"): (float(row.limit_up), float(row.limit_down))
        for row in limits.itertuples(index=False)
        if np.isfinite(row.limit_up) and np.isfinite(row.limit_down)
    }
    strategies = {
        "tree_low_direct": setup,
        "tree_low_confirmed": confirmed,
        "prior_range_baseline": baseline_low,
    }
    periods = {
        "validation_2017_2021": VALIDATION,
        "historical_check_2022_plus": (
            HOLDOUT_START, bars["date"].max().date().isoformat()
        ),
    }
    results = {}
    for period_name, (start, end) in periods.items():
        for strategy_name, low_signal in strategies.items():
            for slip in SLIPPAGE_BPS:
                result = _simulate_period(
                    bars, low_signal, high_signal, limit_lookup, dividends,
                    start, end, slip, initial_capital, True,
                )
                results[f"{period_name}_{strategy_name}_{slip}bps"] = result
    buy_hold = _simulate_period(
        bars, setup, high_signal, limit_lookup, dividends,
        HOLDOUT_START, bars["date"].max().date().isoformat(),
        10, initial_capital, False,
    )
    summary = {
        "symbol": "2330",
        "entry_setup": {
            "rule": "return_1 <= -0.0025693492 AND close_location <= 0.25657895",
            "threshold_origin": "2010-2016 training tree; depth selected on 2017-2021 validation",
            "interpretation": "one-day negative close return with close in bottom 25.7% of intraday range",
        },
        "confirmation": "within 5 following sessions, close > prior session high AND close_location >= 0.55",
        "exit_rule": "base high setup + bearish MACD histogram divergence20, selected previously on 2017-2021",
        "validation_period": list(VALIDATION),
        "historical_check_period": [HOLDOUT_START, bars["date"].max().date().isoformat()],
        "results": {k: compact(v) for k, v in results.items()},
        "historical_check_buy_hold_10bps": compact(buy_hold),
        "test_period_used_for_rule_selection": False,
        "historical_check_is_pristine": False,
        "warning": "2022+ has previously been inspected and is descriptive only. This is a fixed rule test, not evidence of a pristine blind test or a 200% CAGR strategy."
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_tree_rule_execution_backtest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for key, value in results.items():
        pd.DataFrame(value["trades"]).to_csv(out / f"{key}_trades.csv", index=False)
        pd.DataFrame(value["equity_curve"]).to_csv(out / f"{key}_equity.csv", index=False)
    return summary


def main():
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
