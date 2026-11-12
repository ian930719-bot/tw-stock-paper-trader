"""Causal fixed-percentage trailing-exit and re-entry search for TSMC 2330.

Entry and high-point rules are unchanged. A close-based trailing drawdown may
schedule an exit for the next session open. Normal LOW signals can re-enter
only after the exit has actually filled. Candidate selection is research-only.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_atr_reentry_rule_search import (
    LOW_RETURN, RESEARCH, VALIDATION, compact, load_inputs, make_signals,
)
from tsmc_2330_execution_backtest import (
    INITIAL_CAPITAL, SLIPPAGE_BPS, _simulate_period,
)

WIN_FLOOR = 0.8695652173913043
TRAILING_DRAWDOWNS = (None, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20)


def run(features_path: str, limits_path: str, dividends_path: str, outdir: str) -> dict:
    d, limits, dividends = load_inputs(features_path, limits_path, dividends_path)
    test_end = d.date.max().date().isoformat()
    grid = list(itertools.product(
        (1.1, 1.2), (0.70, 0.75, 0.80), (0.08, 0.10, 0.12),
        TRAILING_DRAWDOWNS,
    ))
    rows = []
    for low_rv, high_loc, high_dist, trail in grid:
        low, high, _ = make_signals(d, low_rv, high_loc, high_dist)
        research = _simulate_period(
            d, low, high, limits, dividends, *RESEARCH, 10,
            INITIAL_CAPITAL, True, trailing_drawdown_pct=trail,
        )
        rows.append({
            "params": {
                "low_return": LOW_RETURN, "low_relative_volume": low_rv,
                "high_close_location": high_loc, "high_distance_sma200": high_dist,
                "trailing_drawdown_pct": trail,
            },
            "research": compact(research),
            "research_trades": research["trade_count"],
            "research_win_rate": research["win_rate"],
        })

    eligible = [
        row for row in rows
        if row["research_trades"] >= 5
        and row["research_win_rate"] is not None
        and row["research_win_rate"] >= WIN_FLOOR
    ]
    selected = max(
        eligible,
        key=lambda row: (
            row["research"]["cagr"] if row["research"]["cagr"] is not None else -1,
            row["research_win_rate"],
            -abs(row["research"]["max_drawdown"] or -1),
        ),
    ) if eligible else None

    for row in rows:
        p = row["params"]
        low, high, _ = make_signals(
            d, p["low_relative_volume"], p["high_close_location"],
            p["high_distance_sma200"],
        )
        for label, (start, end) in (
            ("validation", VALIDATION),
            ("test", ("2021-01-01", test_end)),
        ):
            metrics = _simulate_period(
                d, low, high, limits, dividends, start, end, 10,
                INITIAL_CAPITAL, True, trailing_drawdown_pct=p["trailing_drawdown_pct"],
            )
            row[label + "_10bps"] = compact(metrics)

    stress = {}
    if selected:
        p = selected["params"]
        low, high, _ = make_signals(
            d, p["low_relative_volume"], p["high_close_location"],
            p["high_distance_sma200"],
        )
        for label, (start, end) in (
            ("research", RESEARCH),
            ("validation", VALIDATION),
            ("test", ("2021-01-01", test_end)),
        ):
            stress[label] = {}
            for slip in SLIPPAGE_BPS:
                metrics = _simulate_period(
                    d, low, high, limits, dividends, start, end, slip,
                    INITIAL_CAPITAL, True, trailing_drawdown_pct=p["trailing_drawdown_pct"],
                )
                stress[label][f"{slip}bps"] = compact(metrics)

    result = {
        "symbol": "2330",
        "candidate_count": len(rows),
        "entry_rule": "fixed LOW return -0.8%; relative volume 1.1/1.2; Stoch K<=30; close location<=0.25; OBV flow ratio<=0",
        "exit_rule": "existing causal HIGH signal OR close-based trailing drawdown from highest close; next-session open execution",
        "trailing_drawdown_grid": list(TRAILING_DRAWDOWNS),
        "selection_period": list(RESEARCH),
        "validation_period": list(VALIDATION),
        "historical_test_period": ["2021-01-01", test_end],
        "win_rate_floor": WIN_FLOOR,
        "selection_rule": "research-only maximize CAGR subject to >=5 closed trades and win rate >=86.9565%; validation/test report-only",
        "selected_candidate": selected,
        "selected_cost_stress": stress,
        "eligible_candidate_count": len(eligible),
        "all_candidates": rows,
        "target_status": "NOT_ESTABLISHED unless historical test reaches CAGR>=40% and win_rate>=86.9565%; forward paper validation required",
        "execution": "long-only all-in/all-out; close signal at t and next-session open fill; integer shares; Taiwan commissions, sell tax, slippage, official price-limit lock handling and cash dividends; re-entry only after sell fill",
        "warning": "2021+ history was previously inspected and is not pristine OOS. Research-period selection does not establish future performance.",
    }
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_trailing_reentry_search.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    flat = []
    for row in rows:
        p = row["params"]
        flat.append({
            **p,
            "research_cagr": row["research"].get("cagr"),
            "research_win_rate": row["research_win_rate"],
            "research_trades": row["research_trades"],
            "research_eligible": row in eligible,
            "validation_cagr": row["validation_10bps"].get("cagr"),
            "validation_win_rate": row["validation_10bps"].get("win_rate"),
            "validation_trades": row["validation_10bps"].get("trade_count"),
            "test_cagr": row["test_10bps"].get("cagr"),
            "test_win_rate": row["test_10bps"].get("win_rate"),
            "test_trades": row["test_10bps"].get("trade_count"),
        })
    pd.DataFrame(flat).to_csv(out / "tsmc_2330_trailing_reentry_search.csv", index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--limits", required=True)
    parser.add_argument("--dividends", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.limits, args.dividends, args.output),
                     ensure_ascii=False, indent=2, default=float))
