"""ATR trailing-stop and causal re-entry search for TSMC 2330.

LOW entries retain the -0.8% candidate threshold. HIGH exits and ATR stop
multiples are predeclared. A stop is detected from close t and sold no earlier
than the next session open through the shared execution simulator. Once flat,
the normal causal LOW rule can re-enter on a later next-session open.

Selection is research-only (2010-2015); validation (2016-2020) and historical
test (2021+) are report-only. Historical test has been examined previously and
is not pristine OOS. Forward paper validation is still required.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_execution_backtest import (
    INITIAL_CAPITAL,
    SLIPPAGE_BPS,
    _simulate_period,
    trailing_atr_stop_hit,
)

RESEARCH = ("2010-01-04", "2015-12-31")
VALIDATION = ("2016-01-04", "2020-12-31")
WIN_FLOOR = 0.8695652173913043
LOW_RETURN = -0.008
ATR_MULTIPLES = (None, 2.0, 2.5, 3.0, 3.5, 4.0)


def load_inputs(features_path: str, limits_path: str, dividends_path: str):
    d = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    required = {
        "date", "raw_open", "raw_high", "raw_low", "raw_close", "return_1",
        "relative_volume20", "stoch_k14", "close_location", "obv_flow_ratio_5",
        "intraday_range_pct", "distance_sma_200", "macd_hist_change_3",
        "sma50_slope_10", "corporate_action_window", "atr_pct14",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Missing required causal/ATR features: {missing}")
    d["atr_pct14"] = pd.to_numeric(d["atr_pct14"], errors="coerce")
    if not np.isfinite(d["atr_pct14"].dropna()).all() or (d["atr_pct14"].dropna() <= 0).any():
        raise ValueError("ATR feature contains invalid non-positive values")
    if d["atr_pct14"].notna().mean() < 0.95:
        raise ValueError("ATR14 coverage below 95%; refusing incomplete stop research")

    lim = pd.read_csv(limits_path, dtype={"stock_id": str})
    lim = lim[lim.stock_id.astype(str).str.strip() == "2330"].copy()
    lim["date"] = pd.to_datetime(lim.date, errors="coerce").dt.normalize()
    lookup = {
        r.date.strftime("%Y-%m-%d"): (float(r.limit_up), float(r.limit_down))
        for r in lim.itertuples(index=False)
        if pd.notna(r.limit_up) and pd.notna(r.limit_down)
    }
    coverage = d["date"].dt.strftime("%Y-%m-%d").isin(lookup).mean()
    if coverage < 0.98:
        raise RuntimeError(f"Price-limit coverage below 98%: {coverage:.3%}")

    div = pd.read_csv(dividends_path)
    required_div = {"date", "stock_or_cache_dividend", "stock_and_cache_dividend"}
    if not required_div.issubset(div.columns):
        raise ValueError(f"Dividend file missing columns: {sorted(required_div-set(div.columns))}")
    div = div[div.stock_or_cache_dividend.astype(str).str.strip() == "息"].copy()
    div["date"] = pd.to_datetime(div.date, errors="coerce").dt.normalize()
    div["stock_and_cache_dividend"] = pd.to_numeric(div.stock_and_cache_dividend, errors="coerce")
    div = div.dropna(subset=["date", "stock_and_cache_dividend"])
    dividends = {
        r.date.strftime("%Y-%m-%d"): float(r.stock_and_cache_dividend)
        for r in div.itertuples(index=False)
    }
    return d, lookup, dividends


def make_signals(d: pd.DataFrame, low_rv: float, high_loc: float,
                 high_dist: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action = d.corporate_action_window.fillna(True).astype(bool).to_numpy()
    low = (
        (d.relative_volume20 >= low_rv)
        & (d.return_1 <= LOW_RETURN)
        & (d.stoch_k14 <= 30)
        & (d.close_location <= 0.25)
        & (d.obv_flow_ratio_5 <= 0)
    ).fillna(False).to_numpy(dtype=bool) & ~action
    high = (
        (d.return_1 >= 0.01)
        & (d.close_location >= high_loc)
        & (d.intraday_range_pct >= 0.016)
        & (d.distance_sma_200 >= high_dist)
        & (d.macd_hist_change_3 <= 0)
        & (d.sma50_slope_10 > 0)
    ).fillna(False).to_numpy(dtype=bool) & ~action
    atr_mask = ~action & d.atr_pct14.notna().to_numpy()
    return low, high, atr_mask


def compact(m: dict) -> dict:
    keys = (
        "period_start", "period_end", "final_equity", "total_return", "cagr",
        "max_drawdown", "trade_count", "win_rate", "exposure_ratio",
        "total_commission", "total_sell_tax", "estimated_slippage_cost",
        "blocked_limit_orders", "missing_price_limit_orders",
    )
    return {k: m.get(k) for k in keys}


def run(features_path: str, limits_path: str, dividends_path: str, outdir: str) -> dict:
    d, limits, dividends = load_inputs(features_path, limits_path, dividends_path)
    test_end = d.date.max().date().isoformat()
    grid = list(itertools.product((1.1, 1.2), (0.70, 0.75, 0.80),
                                  (0.08, 0.10, 0.12), ATR_MULTIPLES))
    rows = []
    for low_rv, high_loc, high_dist, atr_multiple in grid:
        low, high, atr_mask = make_signals(d, low_rv, high_loc, high_dist)
        research = _simulate_period(
            d, low, high, limits, dividends, *RESEARCH, 10,
            INITIAL_CAPITAL, True, atr_stop_multiple=atr_multiple,
            atr_stop_mask=atr_mask,
        )
        rows.append({
            "params": {
                "low_return": LOW_RETURN, "low_relative_volume": low_rv,
                "high_close_location": high_loc, "high_distance_sma200": high_dist,
                "atr_stop_multiple": atr_multiple,
            },
            "research": compact(research),
            "research_trades": research["trade_count"],
            "research_win_rate": research["win_rate"],
        })

    eligible = [
        x for x in rows
        if x["research_trades"] >= 5
        and x["research_win_rate"] is not None
        and x["research_win_rate"] >= WIN_FLOOR
    ]
    selected = max(
        eligible,
        key=lambda x: (
            x["research"]["cagr"] if x["research"]["cagr"] is not None else -1,
            x["research_win_rate"],
            -abs(x["research"]["max_drawdown"] or -1),
        ),
    ) if eligible else None

    # Evaluate every predeclared candidate on validation and historical test.
    for row in rows:
        p = row["params"]
        low, high, atr_mask = make_signals(
            d, p["low_relative_volume"], p["high_close_location"],
            p["high_distance_sma200"],
        )
        for label, (start, end) in (
            ("validation", VALIDATION),
            ("test", ("2021-01-01", test_end)),
        ):
            m = _simulate_period(
                d, low, high, limits, dividends, start, end, 10,
                INITIAL_CAPITAL, True, atr_stop_multiple=p["atr_stop_multiple"],
                atr_stop_mask=atr_mask,
            )
            row[label + "_10bps"] = compact(m)

    stress = {}
    if selected:
        p = selected["params"]
        low, high, atr_mask = make_signals(
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
                m = _simulate_period(
                    d, low, high, limits, dividends, start, end, slip,
                    INITIAL_CAPITAL, True, atr_stop_multiple=p["atr_stop_multiple"],
                    atr_stop_mask=atr_mask,
                )
                stress[label][f"{slip}bps"] = compact(m)

    no_stop = [r for r in rows if r["params"]["atr_stop_multiple"] is None]
    result = {
        "symbol": "2330",
        "candidate_count": len(rows),
        "low_return_fixed": LOW_RETURN,
        "atr_multiples": list(ATR_MULTIPLES),
        "selection_period": list(RESEARCH),
        "validation_period": list(VALIDATION),
        "historical_test_period": ["2021-01-01", test_end],
        "selection_rule": "research-only maximize CAGR subject to >=5 trades and win rate >=86.9565%; validation/test are report-only",
        "win_rate_floor": WIN_FLOOR,
        "selected_candidate": selected,
        "selected_cost_stress": stress,
        "no_atr_stop_candidates": len(no_stop),
        "all_candidates": rows,
        "target_status": "NOT_ESTABLISHED; require both historical test CAGR >=40% and win rate >=86.9565%, then forward paper validation",
        "execution": "close signal/ATR stop at t; next-session open execution; all-in/all-out long-only; locked-limit handling; commissions, sell tax, adverse slippage and cash dividends; LOW signals may re-enter after a completed stop exit",
        "atr_definition": "atr_pct14 feature; stop uses shared trailing_atr_stop_hit; stop decision at close and execution no earlier than next session open",
        "warning": "Historical test was previously examined and is not pristine OOS. This finite search does not guarantee future performance.",
    }
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_atr_reentry_search.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    flat = []
    for r in rows:
        p = r["params"]
        flat.append({
            **p,
            "research_cagr": r["research"].get("cagr"),
            "research_win_rate": r["research_win_rate"],
            "research_trades": r["research_trades"],
            "research_eligible": r in eligible,
            "validation_cagr": r["validation_10bps"].get("cagr"),
            "validation_win_rate": r["validation_10bps"].get("win_rate"),
            "test_cagr": r["test_10bps"].get("cagr"),
            "test_win_rate": r["test_10bps"].get("win_rate"),
            "test_trades": r["test_10bps"].get("trade_count"),
        })
    pd.DataFrame(flat).to_csv(out / "tsmc_2330_atr_reentry_search.csv", index=False)
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
