"""Continuous-account annual OOF TSMC 2330 probability strategy.

Annual models are fit only on confirmed events from preceding years. Their
out-of-fold probabilities are concatenated into one continuous portfolio.
Cash, shares, pending orders and peak-close state carry through year boundaries.
The sample-end close is an explicit terminal accounting settlement only.
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
from tsmc_2330_walk_forward import annual_fold_years, event_target, fold_events, fold_masks
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
        LogisticRegression(max_iter=2000, class_weight="balanced", C=0.2, random_state=0),
    )


def rolling_probability_threshold(probability, quantile=SIGNAL_QUANTILE,
                                   window=ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS):
    if not 0 < float(quantile) < 1:
        raise ValueError("quantile must be between 0 and 1")
    if window < 1 or min_periods < 1 or min_periods > window:
        raise ValueError("Require 1 <= min_periods <= window")
    return pd.Series(np.asarray(probability, dtype=float)).shift(1).rolling(
        window=window, min_periods=min_periods
    ).quantile(float(quantile)).to_numpy(dtype=float)


def rolling_probability_signal(probability, quantile=SIGNAL_QUANTILE,
                               window=ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS):
    p = np.asarray(probability, dtype=float)
    cutoff = rolling_probability_threshold(p, quantile, window, min_periods)
    return np.isfinite(p) & np.isfinite(cutoff) & (p >= cutoff), cutoff


def probability_signal(probability, threshold):
    p = np.asarray(probability, dtype=float)
    return np.isfinite(p) & (p >= float(threshold))


def score(result):
    return float(result["cagr"]) - 0.5 * abs(float(result["max_drawdown"]))


def compact(result):
    keys = (
        "period_start", "period_end", "initial_capital", "final_equity",
        "total_return", "cagr", "max_drawdown", "trade_count", "win_rate",
        "average_trade_return", "exposure_ratio", "total_cash_dividends",
        "total_commission", "total_sell_tax", "estimated_slippage_cost",
        "blocked_limit_orders", "missing_price_limit_orders",
    )
    return {k: result.get(k) for k in keys}


def summarize_years_continuous(equity_curve, trades, initial_capital):
    """Annual returns and exit-year trade win rates from one continuous path."""
    curve = pd.DataFrame(equity_curve).copy()
    if curve.empty:
        return []
    curve["date"] = pd.to_datetime(curve["date"])
    curve = curve.sort_values("date").reset_index(drop=True)
    curve["global_peak"] = curve["equity"].cummax()
    curve["global_drawdown"] = curve["equity"] / curve["global_peak"] - 1.0
    ledger = pd.DataFrame(trades).copy()
    if ledger.empty:
        ledger = pd.DataFrame(columns=["exit_date", "net_pnl", "net_return", "exit_year"])
    else:
        ledger["exit_date"] = pd.to_datetime(ledger["exit_date"])
        ledger["exit_year"] = ledger["exit_date"].dt.year
    rows = []
    prior_equity = float(initial_capital)
    for year, group in curve.groupby(curve["date"].dt.year, sort=True):
        end_equity = float(group["equity"].iloc[-1])
        yr_trades = ledger.loc[ledger["exit_year"] == year]
        pnl = pd.to_numeric(yr_trades.get("net_pnl", pd.Series(dtype=float)), errors="coerce")
        trade_returns = pd.to_numeric(yr_trades.get("net_return", pd.Series(dtype=float)), errors="coerce")
        rows.append({
            "year": int(year),
            "period_start": group["date"].iloc[0].date().isoformat(),
            "period_end": group["date"].iloc[-1].date().isoformat(),
            "start_equity": prior_equity,
            "end_equity": end_equity,
            "annual_return": end_equity / prior_equity - 1.0 if prior_equity > 0 else None,
            "global_high_water_mark_drawdown": float(group["global_drawdown"].min()),
            "trade_count_exited": int(len(yr_trades)),
            "win_rate_exited": float((pnl > 0).mean()) if len(pnl) else None,
            "average_trade_return_exited": float(trade_returns.mean()) if len(trade_returns) else None,
        })
        prior_equity = end_equity
    return rows


def run(features_path, oracle_path, limits_path, dividends_path, output_dir,
        initial_capital=INITIAL_CAPITAL):
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
    probabilities = {kind: np.full(len(bars), np.nan) for kind in ("LOW", "HIGH")}
    fold_metadata = {kind: [] for kind in ("LOW", "HIGH")}
    skipped = []

    for year in years:
        fold = fold_masks(dates, valid_rows, year, HORIZON)
        if fold is None or not fold["validation_mask"].any():
            skipped.append({"year": year, "reason": "no eligible validation rows"})
            continue
        train_events, val_events = fold_events(events, dates, fold, HORIZON)
        train_mask, val_mask = fold["train_mask"], fold["validation_mask"]
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
            train_cols = [c for c in feature_cols if X.loc[train_mask, c].notna().any()]
            if not train_cols:
                skipped.append({"year": year, "pivot_type": kind,
                                "reason": "no features populated in training history"})
                year_ok = False
                continue
            model = make_model()
            model.fit(X.loc[train_mask, train_cols], y_train[train_mask])
            probabilities[kind][val_mask] = model.predict_proba(X.loc[val_mask, train_cols])[:, 1]
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
                "validation_pr_auc": float(average_precision_score(truth, p)) if truth.sum() else 0.0,
            })
        if not year_ok:
            probabilities["LOW"][val_mask] = np.nan
            probabilities["HIGH"][val_mask] = np.nan

    low_signal, low_threshold = rolling_probability_signal(probabilities["LOW"])
    high_signal, high_threshold = rolling_probability_signal(probabilities["HIGH"])
    eligible = np.isfinite(probabilities["LOW"]) & np.isfinite(probabilities["HIGH"])
    indices = np.flatnonzero(eligible)
    if not len(indices):
        raise RuntimeError("No rows have both causal OOF probabilities; refusing backtest")
    first_date = dates.iloc[int(indices[0])].date().isoformat()
    last_date = dates.iloc[int(indices[-1])].date().isoformat()

    continuous_results, annual_results = {}, {}
    trade_rows, equity_rows = [], []
    for slip in SLIPPAGE_BPS:
        # Single continuous account; do not reset or liquidate at annual boundaries.
        full = _simulate_period(
            bars, low_signal, high_signal, {
                r.date.strftime("%Y-%m-%d"): (float(r.limit_up), float(r.limit_down))
                for r in limits.itertuples(index=False)
                if np.isfinite(r.limit_up) and np.isfinite(r.limit_down)
            }, dividends, first_date, last_date, slip, initial_capital, True,
        )
        continuous_results[f"{slip}bps"] = compact(full)
        annual_results[f"{slip}bps"] = summarize_years_continuous(
            full["equity_curve"], full["trades"], initial_capital
        )
        for trade in full["trades"]:
            trade_rows.append({**trade, "slippage_bps_per_side": slip})
        peak = 0.0
        for point in full["equity_curve"]:
            peak = max(peak, float(point["equity"]))
            equity_rows.append({
                **point,
                "global_high_water_mark": peak,
                "global_drawdown": float(point["equity"]) / peak - 1.0,
                "slippage_bps_per_side": slip,
            })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trade_rows).to_csv(out / "tsmc_2330_probability_strategy_trades.csv", index=False)
    pd.DataFrame(equity_rows).to_csv(out / "tsmc_2330_probability_strategy_equity.csv", index=False)
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
        "symbol": "2330",
        "horizon_sessions": HORIZON,
        "pivot_threshold": {
            "min_reversal_pct": min_reversal, "atr_multiple": atr_multiple,
            "source": "oracle threshold selected using training period only",
        },
        "validation_method": "expanding annual walk-forward; each fold trained only on confirmed events from preceding years; signal at close t and execution no earlier than next session",
        "validation_years": years,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "model": "fixed logistic regression; median imputation and scaling fitted inside each annual fold; balanced classes; C=0.2",
        "model_selection": "none; model family fixed before annual validation",
        "probability_policy": {
            "quantile": SIGNAL_QUANTILE,
            "window_sessions": ROLLING_WINDOW,
            "minimum_prior_probabilities": ROLLING_MIN_PERIODS,
            "threshold": "rolling quantile of prior OOF probabilities only; shifted one session",
            "selection_on_validation_year": False,
        },
        "fold_probabilities": fold_metadata,
        "skipped_folds": skipped,
        "annual_continuous_oos_results": annual_results,
        "continuous_oos_results": continuous_results,
        "trade_ledger_rows": len(trade_rows),
        "equity_curve_rows": len(equity_rows),
        "audit_files": [
            "tsmc_2330_probability_strategy_signals.csv",
            "tsmc_2330_probability_strategy_trades.csv",
            "tsmc_2330_probability_strategy_equity.csv",
        ],
        "execution": {
            "signal_timing": "probability at close t; order for next session open",
            "annual_boundary": "continuous account; cash, shares, peak close and pending orders carry over year boundaries; no year-end liquidation",
            "terminal_boundary": "remaining position is settled at final sample close with adverse slippage, commission and sell tax; terminal accounting only, not a predictive exit signal",
            "commission_rate_per_side": 0.001425,
            "minimum_commission_ntd": 20,
            "sell_transaction_tax_rate": 0.003,
            "slippage_bps_per_side": list(SLIPPAGE_BPS),
            "price_limits": "historical TWSE price limits and tick sizes; locked orders handled conservatively",
            "dividends": "cash dividends credited to shares held at prior session close",
            "initial_capital": initial_capital,
            "leverage": 1.0,
            "short_selling": False,
        },
        "pristine_holdout": False,
        "warning": "Continuous accounting removes artificial year-end liquidation, but 2011-2026 is not pristine holdout; 2026 is partial and the underlying pivot definition was selected in prior research. Historical results are not a guarantee.",
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
                         args.output, args.initial_capital), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
