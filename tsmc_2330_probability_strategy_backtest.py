"""Annual walk-forward, costed TSMC 2330 probability strategy.

Each calendar-year fold fits on confirmed labels from prior years only. A fixed
logistic model and predeclared causal 85th-percentile signal policy avoid
choosing model families or probability cutoffs on the validation year. Signals
are formed at close t and orders execute no earlier than the next session.
Each annual simulation is liquidated at year end and carries resulting cash
into the next fold; no position is carried across fold boundaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tsmc_swing_research import detect_confirmed_pivots
from tsmc_2330_walk_forward import (
    annual_fold_years, event_target, fold_events, fold_masks,
)
from tsmc_2330_execution_backtest import (
    SLIPPAGE_BPS, INITIAL_CAPITAL, load_inputs, _simulate_period,
)

HORIZON = 1
SIGNAL_QUANTILE = 0.85
ROLLING_WINDOW = 252
ROLLING_MIN_PERIODS = 60
EXCLUDE = {
    "obv", "symbol", "date", "open", "high", "low", "close", "volume",
    "turnover", "corporate_action_window", "atr14",
}


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=2000, class_weight="balanced", C=0.2, random_state=0
        ),
    )


def rolling_probability_threshold(
    probability: np.ndarray, quantile: float = SIGNAL_QUANTILE,
    window: int = ROLLING_WINDOW, min_periods: int = ROLLING_MIN_PERIODS,
) -> np.ndarray:
    """Causal rolling cutoff: only probabilities from earlier sessions count."""
    if not 0 < float(quantile) < 1:
        raise ValueError("quantile must be between 0 and 1")
    values = pd.Series(np.asarray(probability, dtype=float))
    return values.shift(1).rolling(
        window=window, min_periods=min_periods
    ).quantile(float(quantile)).to_numpy(dtype=float)


def rolling_probability_signal(
    probability: np.ndarray, quantile: float = SIGNAL_QUANTILE,
    window: int = ROLLING_WINDOW, min_periods: int = ROLLING_MIN_PERIODS,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probability, dtype=float)
    threshold = rolling_probability_threshold(p, quantile, window, min_periods)
    return (np.isfinite(p) & np.isfinite(threshold) & (p >= threshold)), threshold


def probability_signal(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Compatibility helper for evaluating a fixed, externally predeclared cutoff."""
    p = np.asarray(probability, dtype=float)
    return np.isfinite(p) & (p >= float(threshold))


def score(result: dict) -> float:
    """Legacy score helper retained for downstream imports and tests."""
    return float(result["cagr"]) - 0.5 * abs(float(result["max_drawdown"]))


def compact(result: dict) -> dict:
    keys = (
        "period_start", "period_end", "initial_capital", "final_equity",
        "total_return", "cagr", "max_drawdown", "trade_count", "win_rate",
        "average_trade_return", "exposure_ratio", "total_cash_dividends",
        "total_commission", "total_sell_tax", "estimated_slippage_cost",
        "blocked_limit_orders", "missing_price_limit_orders",
    )
    return {k: result.get(k) for k in keys}


def run(features_path: str, oracle_path: str, limits_path: str,
        dividends_path: str, output_dir: str,
        initial_capital: float = INITIAL_CAPITAL) -> dict:
    bars, _, limits, dividends = load_inputs(
        features_path, oracle_path, limits_path, dividends_path
    )
    oracle = json.loads(Path(oracle_path).read_text(encoding="utf-8"))
    selected = oracle.get("threshold_selected_using_training_period_only", {})
    min_reversal = selected.get("min_reversal_pct")
    atr_multiple = selected.get("atr_multiple")
    if min_reversal is None or atr_multiple is None:
        raise ValueError("Missing training-only pivot threshold")
    events = detect_confirmed_pivots(bars, float(min_reversal), float(atr_multiple))
    if events.empty:
        raise ValueError("No pivot events")

    feature_cols = [
        c for c in bars.columns
        if c not in EXCLUDE and not c.startswith("raw_")
        and pd.api.types.is_numeric_dtype(bars[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric model features")
    X = bars[feature_cols].replace([np.inf, -np.inf], np.nan)
    dates = pd.to_datetime(bars["date"]).reset_index(drop=True)
    valid_rows = X.notna().any(axis=1).to_numpy(copy=True)
    if "corporate_action_window" in bars:
        valid_rows &= ~bars["corporate_action_window"].fillna(True).astype(bool).to_numpy()

    years = annual_fold_years(dates)
    if not years:
        raise ValueError("At least two calendar years are required")
    probabilities = {kind: np.full(len(bars), np.nan, dtype=float) for kind in ("LOW", "HIGH")}
    fold_metadata = {kind: [] for kind in ("LOW", "HIGH")}
    skipped = []

    for year in years:
        fold = fold_masks(dates, valid_rows, year, HORIZON)
        if fold is None or not fold["validation_mask"].any():
            skipped.append({"year": year, "reason": "no eligible validation rows"})
            continue
        train_events, val_events = fold_events(events, dates, fold, HORIZON)
        train_mask = fold["train_mask"]
        val_mask = fold["validation_mask"]
        if not train_mask.any() or not val_mask.any():
            skipped.append({"year": year, "reason": "empty train or validation mask"})
            continue
        year_ok = True
        for kind in ("LOW", "HIGH"):
            y_train = event_target(dates, train_events, HORIZON, kind)
            y_val = event_target(dates, val_events, HORIZON, kind)
            if len(np.unique(y_train[train_mask])) < 2:
                skipped.append({
                    "year": year, "pivot_type": kind,
                    "reason": "training fold has one target class",
                    "training_rows": int(train_mask.sum()),
                    "training_events": int(y_train[train_mask].sum()),
                })
                year_ok = False
                continue
            # Feature availability is determined from this fold's training
            # history only; a field first populated in a future year cannot
            # silently enter an earlier model.
            train_cols = [
                col for col in feature_cols
                if X.loc[train_mask, col].notna().any()
            ]
            if not train_cols:
                skipped.append({
                    "year": year, "pivot_type": kind,
                    "reason": "no features populated in training history",
                })
                year_ok = False
                continue
            model = make_model()
            model.fit(X.loc[train_mask, train_cols], y_train[train_mask])
            probabilities[kind][val_mask] = model.predict_proba(
                X.loc[val_mask, train_cols]
            )[:, 1]
            p = probabilities[kind][val_mask]
            truth = y_val[val_mask]
            fold_metadata[kind].append({
                "validation_year": year,
                "training_start": dates.loc[train_mask].min().date().isoformat(),
                "training_end": dates.loc[train_mask].max().date().isoformat(),
                "validation_start": dates.loc[val_mask].min().date().isoformat(),
                "validation_end": dates.loc[val_mask].max().date().isoformat(),
                "partial_validation_year": year == int(dates.max().year),
                "training_rows": int(train_mask.sum()),
                "training_events": int(y_train[train_mask].sum()),
                "validation_rows": int(val_mask.sum()),
                "validation_events": int(truth.sum()),
                "feature_count": len(train_cols),
                "features": train_cols,
                "validation_prevalence": float(truth.mean()),
                "validation_pr_auc": (
                    float(average_precision_score(truth, p))
                    if truth.sum() else 0.0
                ),
            })
        if not year_ok:
            # Do not execute a calendar year unless both directions have OOF scores.
            for kind in ("LOW", "HIGH"):
                probabilities[kind][val_mask] = np.nan

    low_signal, low_threshold = rolling_probability_signal(probabilities["LOW"])
    high_signal, high_threshold = rolling_probability_signal(probabilities["HIGH"])
    limit_lookup = {
        r.date.strftime("%Y-%m-%d"): (float(r.limit_up), float(r.limit_down))
        for r in limits.itertuples(index=False)
        if np.isfinite(r.limit_up) and np.isfinite(r.limit_down)
    }

    # Annual folds are separately liquidated and the cash balance is compounded.
    yearly_results = {}
    compounded = {}
    trade_ledger_rows = []
    equity_curve_rows = []
    for slip in SLIPPAGE_BPS:
        cash = float(initial_capital)
        rows = []
        for year in years:
            year_mask = (dates.dt.year == year).to_numpy()
            if not year_mask.any():
                continue
            year_start = dates.loc[year_mask].min().date().isoformat()
            year_end = dates.loc[year_mask].max().date().isoformat()
            eligible = year_mask & np.isfinite(probabilities["LOW"]) & np.isfinite(probabilities["HIGH"])
            if not eligible.any():
                rows.append({"validation_year": year, "status": "skipped_no_oof_probabilities"})
                continue
            result = _simulate_period(
                bars, low_signal, high_signal, limit_lookup, dividends,
                year_start, year_end, slip, cash, True,
            )
            item = compact(result)
            item.update({"validation_year": year, "status": "evaluated"})
            rows.append(item)
            for trade in result.get("trades", []):
                trade_ledger_rows.append({
                    **trade, "validation_year": year, "slippage_bps_per_side": slip,
                })
            for point in result.get("equity_curve", []):
                equity_curve_rows.append({
                    **point, "validation_year": year, "slippage_bps_per_side": slip,
                })
            cash = float(result["final_equity"])
        evaluated = [r for r in rows if r.get("status") == "evaluated"]
        first = next((r["period_start"] for r in evaluated), None)
        last = next((r["period_end"] for r in reversed(evaluated)), None)
        elapsed = ((pd.Timestamp(last) - pd.Timestamp(first)).days / 365.25
                   if first and last else 0)
        final_equity = cash
        total_return = final_equity / initial_capital - 1
        cagr = ((final_equity / initial_capital) ** (1 / elapsed) - 1
                if elapsed > 0 and final_equity > 0 else None)
        yearly_results[f"{slip}bps"] = rows
        compounded[f"{slip}bps"] = {
            "initial_capital": initial_capital, "final_equity": final_equity,
            "total_return": total_return, "cagr": cagr,
            "max_annual_fold_drawdown": (
                min((r["max_drawdown"] for r in evaluated
                     if r.get("max_drawdown") is not None), default=None)
            ),
            "evaluated_years": [r["validation_year"] for r in evaluated],
            "skipped_years": [r["validation_year"] for r in rows
                              if r.get("status") != "evaluated"],
            "annual_fold_results": rows,
        }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trade_ledger_rows).to_csv(
        out / "tsmc_2330_probability_strategy_trades.csv", index=False
    )
    pd.DataFrame(equity_curve_rows).to_csv(
        out / "tsmc_2330_probability_strategy_equity.csv", index=False
    )
    pd.DataFrame({
        "date": dates.dt.strftime("%Y-%m-%d"),
        "low_probability_oof": probabilities["LOW"],
        "high_probability_oof": probabilities["HIGH"],
        "low_signal": low_signal,
        "high_signal": high_signal,
        "low_dynamic_threshold": low_threshold,
        "high_dynamic_threshold": high_threshold,
    }).to_csv(out / "tsmc_2330_probability_strategy_signals.csv", index=False)

    summary = {
        "symbol": "2330", "horizon_sessions": HORIZON,
        "pivot_threshold": {
            "min_reversal_pct": min_reversal, "atr_multiple": atr_multiple,
            "source": "oracle threshold selected using training period only",
        },
        "validation_method": "expanding annual walk-forward; each validation year trained only on confirmed events from prior years; last horizon sessions purged",
        "validation_years": years, "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "model": "fixed logistic regression; median imputation and scaling fitted inside each annual fold; balanced classes; C=0.2",
        "model_selection": "none; model family fixed before annual validation",
        "probability_policy": {
            "quantile": SIGNAL_QUANTILE,
            "window_sessions": ROLLING_WINDOW,
            "minimum_prior_probabilities": ROLLING_MIN_PERIODS,
            "threshold": "rolling quantile of prior out-of-fold probabilities only; shifted one session",
            "selection_on_validation_year": False,
        },
        "fold_probabilities": fold_metadata,
        "skipped_folds": skipped,
        "annual_costed_backtests": yearly_results,
        "compounded_oos_results": compounded,
        "trade_ledger_rows": len(trade_ledger_rows),
        "equity_curve_rows": len(equity_curve_rows),
        "audit_files": [
            "tsmc_2330_probability_strategy_signals.csv",
            "tsmc_2330_probability_strategy_trades.csv",
            "tsmc_2330_probability_strategy_equity.csv",
        ],
        "execution": {
            "signal_timing": "probability at close t; order submitted for next session open",
            "annual_boundary": "liquidate at each fold's final close; carry cash into next fold; no cross-year position",
            "commission_rate_per_side": 0.001425, "minimum_commission_ntd": 20,
            "sell_transaction_tax_rate": 0.003,
            "slippage_bps_per_side": list(SLIPPAGE_BPS),
            "price_limits": "historical TWSE price limits and tick sizes; locked orders handled conservatively",
            "dividends": "cash dividends credited to shares held at prior session close",
            "initial_capital": initial_capital, "leverage": 1.0, "short_selling": False,
        },
        "pristine_holdout": False,
        "warning": "This annual OOF backtest supersedes the previously inspected fixed 2010-2016/2017-2021/2022+ selection procedure. The 2026 fold is partial. Results are research diagnostics, not a guarantee or a 200% CAGR claim.",
    }
    (out / "tsmc_2330_probability_strategy_backtest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--limits", required=True)
    parser.add_argument("--dividends", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.oracle, args.limits, args.dividends,
                         args.output, args.initial_capital),
                     ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
