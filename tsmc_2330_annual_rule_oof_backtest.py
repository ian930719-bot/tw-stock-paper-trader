"""Executable annual expanding walk-forward for TSMC 2330, 2010-2026.

2010-2014 form the initial research history. For each validation year Y from
2015 onward, candidates are selected only on data through Y-1, then frozen for
Y. Signals from all validation years are stitched into one continuous account;
positions are not liquidated at annual boundaries. The latest calendar year is
flagged as partial when the data ends before year-end.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_40pct_rule_search import (
    WIN_FLOOR, inputs, make_signals, compact,
)
from tsmc_2330_execution_backtest import (
    INITIAL_CAPITAL, SLIPPAGE_BPS, _simulate_period,
)

GRID = tuple(itertools.product(
    (-0.006, -0.008, -0.010),
    (1.1, 1.2),
    (0.70, 0.75, 0.80),
    (0.08, 0.10, 0.12),
))
SEED_RESEARCH_END = 2014
FIRST_VALIDATION_YEAR = 2015
MIN_TRAIN_TRADES = 5


def _params(values):
    low_ret, low_rv, high_loc, high_dist = values
    return {
        "low_return": low_ret,
        "low_relative_volume": low_rv,
        "high_close_location": high_loc,
        "high_distance_sma200": high_dist,
    }


def select_on_history(d, limits, dividends, end_date: str) -> dict:
    """Select a parameter set using only the historical prefix ending end_date."""
    candidates = []
    for values in GRID:
        p = _params(values)
        low, high = make_signals(
            d, p["low_return"], p["low_relative_volume"],
            p["high_close_location"], p["high_distance_sma200"],
        )
        result = _simulate_period(
            d, low, high, limits, dividends,
            d.date.min().date().isoformat(), end_date,
            10, INITIAL_CAPITAL, True,
        )
        win = result["win_rate"]
        eligible = (
            result["trade_count"] >= MIN_TRAIN_TRADES
            and win is not None and win >= WIN_FLOOR
            and result["cagr"] is not None
        )
        candidates.append({
            "params": p, "metrics": compact(result),
            "eligible": bool(eligible),
        })
    eligible = [c for c in candidates if c["eligible"]]
    if not eligible:
        return {
            "selected": None, "eligible_count": 0,
            "candidate_count": len(candidates),
            "best_unqualified": max(
                candidates,
                key=lambda c: (
                    c["metrics"]["cagr"] if c["metrics"]["cagr"] is not None else -1,
                    c["metrics"]["win_rate"] if c["metrics"]["win_rate"] is not None else -1,
                ),
            ),
        }
    chosen = max(
        eligible,
        key=lambda c: (
            c["metrics"]["cagr"],
            c["metrics"]["win_rate"],
            -abs(c["metrics"]["max_drawdown"] or -1),
        ),
    )
    return {
        "selected": chosen, "eligible_count": len(eligible),
        "candidate_count": len(candidates),
        "best_unqualified": None,
    }


def annual_metrics(curve: pd.DataFrame, trades: list[dict], initial_capital: float) -> list[dict]:
    if curve.empty:
        return []
    q = curve.copy()
    q["date"] = pd.to_datetime(q["date"])
    q["year"] = q["date"].dt.year
    q["global_peak"] = q["equity"].cummax()
    q["global_drawdown"] = q["equity"] / q["global_peak"] - 1
    trade_frame = pd.DataFrame(trades)
    if not trade_frame.empty:
        trade_frame["exit_year"] = pd.to_datetime(trade_frame["exit_date"]).dt.year
    rows = []
    prior_equity = float(initial_capital)
    last_data_year = int(q["year"].max())
    for year, group in q.groupby("year", sort=True):
        end_equity = float(group["equity"].iloc[-1])
        year_trades = (
            trade_frame.loc[trade_frame["exit_year"] == year]
            if not trade_frame.empty else pd.DataFrame()
        )
        rows.append({
            "year": int(year),
            "start_equity": prior_equity,
            "end_equity": end_equity,
            "annual_return": end_equity / prior_equity - 1 if prior_equity else None,
            "max_drawdown_from_global_peak": float(group["global_drawdown"].min()),
            "trade_count": int(len(year_trades)),
            "win_rate": float((year_trades["net_pnl"] > 0).mean()) if len(year_trades) else None,
            "partial_year": int(year) == last_data_year and group["date"].max().month < 12,
        })
        prior_equity = end_equity
    return rows


def run(features_path: str, limits_path: str, dividends_path: str, output_dir: str) -> dict:
    d, limits, dividends = inputs(features_path, limits_path, dividends_path)
    years = sorted(d["date"].dt.year.unique().astype(int).tolist())
    if not years or years[0] > 2010 or years[-1] < 2015:
        raise RuntimeError(f"Expected TSMC history covering 2010-2015+, got {years[:1]}..{years[-1:]}")
    validation_years = [y for y in years if FIRST_VALIDATION_YEAR <= y <= 2026]
    if not validation_years:
        raise RuntimeError("No annual validation years from 2015 through 2026")

    low_oof = np.zeros(len(d), dtype=bool)
    high_oof = np.zeros(len(d), dtype=bool)
    fold_rows = []
    prior_params = None

    for year in validation_years:
        year_end = d.loc[d["date"].dt.year < year, "date"].max()
        if pd.isna(year_end):
            fold_rows.append({"validation_year": year, "status": "skipped_no_prior_history"})
            continue
        selection = select_on_history(d, limits, dividends, year_end.date().isoformat())
        chosen = selection["selected"]
        year_mask = (d["date"].dt.year == year).to_numpy()
        if chosen is not None:
            prior_params = chosen["params"]
            low, high = make_signals(
                d, prior_params["low_return"], prior_params["low_relative_volume"],
                prior_params["high_close_location"], prior_params["high_distance_sma200"],
            )
            low_oof[year_mask] = low[year_mask]
            high_oof[year_mask] = high[year_mask]
            status = "evaluated"
        elif prior_params is not None:
            # Do not open new positions when the strict win-rate gate fails.
            # Retain the last eligible exit rule to manage an existing position.
            _, high = make_signals(
                d, prior_params["low_return"], prior_params["low_relative_volume"],
                prior_params["high_close_location"], prior_params["high_distance_sma200"],
            )
            high_oof[year_mask] = high[year_mask]
            status = "no_eligible_entry_rule_carry_exit_only"
        else:
            status = "no_eligible_rule_no_trade"
        best = selection["best_unqualified"]
        fold_rows.append({
            "validation_year": year,
            "training_start": d["date"].min().date().isoformat(),
            "training_end": year_end.date().isoformat(),
            "status": status,
            "candidate_count": selection["candidate_count"],
            "eligible_candidate_count": selection["eligible_count"],
            "selected_parameters": prior_params if chosen is not None else None,
            "selected_training_metrics": chosen["metrics"] if chosen is not None else None,
            "best_unqualified_parameters": best["params"] if best else None,
            "best_unqualified_training_metrics": best["metrics"] if best else None,
            "win_rate_floor": WIN_FLOOR,
            "training_minimum_trades": MIN_TRAIN_TRADES,
        })

    start = f"{FIRST_VALIDATION_YEAR}-01-01"
    end = d["date"].max().date().isoformat()
    results = {}
    trade_rows = []
    equity_rows = []
    annual_rows = []
    for slip in SLIPPAGE_BPS:
        sim = _simulate_period(
            d, low_oof, high_oof, limits, dividends,
            start, end, slip, INITIAL_CAPITAL, True,
        )
        results[f"{slip}bps"] = compact(sim)
        for number, trade in enumerate(sim["trades"], start=1):
            trade_rows.append({
                "slippage_bps_per_side": slip,
                "trade_number": number,
                **trade,
            })
        curve = pd.DataFrame(sim["equity_curve"])
        for point in sim["equity_curve"]:
            equity_rows.append({"slippage_bps_per_side": slip, **point})
        if slip == 10:
            annual_rows = annual_metrics(curve, sim["trades"], INITIAL_CAPITAL)

    summary = {
        "symbol": "2330",
        "data_period": [d["date"].min().date().isoformat(), end],
        "research_validation_period": ["2010-01-01", "2026-12-31"],
        "seed_research_period": [d["date"].min().date().isoformat(), f"{SEED_RESEARCH_END}-12-31"],
        "annual_validation_years": validation_years,
        "protocol": "For each year Y from 2015-2026, select among a predeclared 36-rule grid using only 2010..Y-1; apply the selected rule only in Y. If no candidate satisfies the strict win-rate and trade-count gates, do not open new positions and retain only the previous eligible exit rule for an existing position.",
        "selection_rule": "maximize historical-prefix CAGR among candidates with at least five completed trades and win rate >= 86.9565%; ties prefer higher win rate then lower absolute drawdown",
        "win_rate_floor": WIN_FLOOR,
        "initial_capital": INITIAL_CAPITAL,
        "account_policy": "one continuous long-only all-in/all-out account across all validation years; no annual reset or year-end liquidation; remaining position is liquidated only at the final data close",
        "signal_execution": "signal at close t; execute next available session open; Taiwan fees, sell tax, configured adverse slippage, official price-limit execution constraints and cash dividends",
        "folds": fold_rows,
        "continuous_oof_results": results,
        "annual_oof_metrics_10bps": annual_rows,
        "pooled_oof_cagr_10bps": results.get("10bps", {}).get("cagr"),
        "pooled_oof_win_rate_10bps": results.get("10bps", {}).get("win_rate"),
        "target_status": "NOT_ESTABLISHED unless both pooled OOF CAGR >= 40% and pooled OOF win rate >= 86.9565%",
        "limitations": [
            "2010-2014 are initial research history, not blind validation years.",
            "2015-2026 are annual expanding walk-forward validation; the latest year may be partial.",
            "Annual parameter selection is retrospective within each training prefix; pooled OOF performance is the relevant executable evaluation.",
            "Daily OHLC cannot model queue priority or exact intraday fills.",
            "Historical price, corporate-action and institutional data still require official-source reconciliation before live use.",
        ],
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_annual_rule_oof_backtest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    pd.DataFrame(fold_rows).to_csv(out / "tsmc_2330_annual_rule_oof_folds.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(out / "tsmc_2330_annual_rule_oof_annual.csv", index=False)
    pd.DataFrame(trade_rows).to_csv(out / "tsmc_2330_annual_rule_oof_trades.csv", index=False)
    pd.DataFrame(equity_rows).to_csv(out / "tsmc_2330_annual_rule_oof_equity.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--limits", required=True)
    parser.add_argument("--dividends", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.limits, args.dividends, args.output),
                     ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
