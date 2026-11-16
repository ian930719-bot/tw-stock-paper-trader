"""Event-driven account drawdown overlay research for TSMC 2330.

Uses the frozen candidate selected by the existing 2010-2015 rule search.
Overlay parameters are selected only on 2015-2021; 2022 onward is report-only.
All overlay exits and re-entries use the existing next-session-open simulator.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_execution_backtest import _simulate_period, INITIAL_CAPITAL
from tsmc_2330_40pct_rule_search import inputs, make_signals

TRAIN_START = "2015-01-05"
TRAIN_END = "2021-12-31"
OOS_START = "2022-01-01"
DRAWDOWN_GRID = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20)
REENTRY_MAS = (20, 50, 100, 200)
SLIPPAGE_BPS = 10


def _period_metrics(curve: pd.DataFrame, trades: list[dict], start: str,
                    end: str, *, use_continuous_drawdown: bool) -> dict:
    z = curve.copy()
    z["date"] = pd.to_datetime(z["date"])
    mask = z["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    p = z.loc[mask].reset_index(drop=True)
    if len(p) < 2:
        raise ValueError(f"Insufficient equity rows for {start} to {end}")
    first_equity = float(p["equity"].iloc[0])
    last_equity = float(p["equity"].iloc[-1])
    # The initial row is already marked to market. For the training period,
    # use the first available equity as the denominator; for OOS this is the
    # carried account value at the boundary, not a fresh NT$100,000 reset.
    elapsed_days = (p["date"].iloc[-1] - p["date"].iloc[0]).days
    years = elapsed_days / 365.25
    cagr = (last_equity / first_equity) ** (1 / years) - 1 if years > 0 and first_equity > 0 else None
    if use_continuous_drawdown:
        dd = pd.to_numeric(p["drawdown"], errors="coerce")
    else:
        dd = p["equity"] / p["equity"].cummax() - 1
    closed = [t for t in trades if pd.Timestamp(start) <= pd.Timestamp(t["exit_date"]) <= pd.Timestamp(end)]
    wins = sum(float(t["net_pnl"]) > 0 for t in closed)
    return {
        "period_start": p["date"].iloc[0].date().isoformat(),
        "period_end": p["date"].iloc[-1].date().isoformat(),
        "starting_equity": first_equity,
        "ending_equity": last_equity,
        "cagr": cagr,
        "max_drawdown": float(dd.min()),
        "trade_count": len(closed),
        "win_rate": wins / len(closed) if closed else None,
        "exposure_ratio": float((pd.to_numeric(p["shares"], errors="coerce") > 0).mean()),
    }


def run(features_path: str, limits_path: str, dividends_path: str,
        rule_search_path: str, output_dir: str) -> dict:
    d, limit_lookup, dividend_lookup = inputs(features_path, limits_path, dividends_path)
    rule_search = json.loads(Path(rule_search_path).read_text(encoding="utf-8"))
    selected = rule_search.get("selected_candidate")
    if not selected:
        raise RuntimeError("Existing frozen rule search selected no candidate; refusing to invent a baseline.")
    params = selected["params"]
    low, high = make_signals(
        d, params["low_return"], params["low_relative_volume"],
        params["high_close_location"], params["high_distance_sma200"],
    )
    test_end = pd.Timestamp(d["date"].max()).date().isoformat()
    start_i = np.flatnonzero(pd.to_datetime(d["date"]).ge(pd.Timestamp(TRAIN_START)).to_numpy())
    if len(start_i) == 0 or pd.Timestamp(test_end) <= pd.Timestamp(OOS_START):
        raise RuntimeError("Insufficient history for the declared train/OOS split.")
    full_end = test_end

    candidates = []
    equity_frames = {}
    for dd_stop, ma in itertools.product(DRAWDOWN_GRID, REENTRY_MAS):
        reentry = pd.to_numeric(d[f"distance_sma_{ma}"], errors="coerce").ge(0).fillna(False).to_numpy()
        result = _simulate_period(
            d, low, high, limit_lookup, dividend_lookup,
            TRAIN_START, full_end, SLIPPAGE_BPS, INITIAL_CAPITAL, True,
            account_drawdown_stop_pct=dd_stop,
            account_reentry_signal=reentry,
            account_reentry_cooldown=5,
        )
        curve = pd.DataFrame(result["equity_curve"])
        curve["peak"] = curve["equity"].cummax()
        curve["drawdown"] = curve["equity"] / curve["peak"] - 1
        train = _period_metrics(curve, result["trades"], TRAIN_START, TRAIN_END,
                                use_continuous_drawdown=False)
        oos = _period_metrics(curve, result["trades"], OOS_START, full_end,
                              use_continuous_drawdown=True)
        candidates.append({
            "drawdown_stop_pct": dd_stop, "reentry_ma": ma,
            "train": train, "oos": oos,
            "account_stop_count": result["account_drawdown_stop_count"],
            "account_reentry_count": result["account_drawdown_reentry_count"],
            "blocked_limit_orders": result["blocked_limit_orders"],
            "missing_price_limit_orders": result["missing_price_limit_orders"],
        })
        equity_frames[f"dd{int(dd_stop*100):02d}_ma{ma}"] = curve

    baseline = _simulate_period(
        d, low, high, limit_lookup, dividend_lookup,
        TRAIN_START, full_end, SLIPPAGE_BPS, INITIAL_CAPITAL, True,
    )
    baseline_curve = pd.DataFrame(baseline["equity_curve"])
    baseline_curve["peak"] = baseline_curve["equity"].cummax()
    baseline_curve["drawdown"] = baseline_curve["equity"] / baseline_curve["peak"] - 1
    baseline_metrics = {
        "train": _period_metrics(baseline_curve, baseline["trades"], TRAIN_START, TRAIN_END,
                                 use_continuous_drawdown=False),
        "oos": _period_metrics(baseline_curve, baseline["trades"], OOS_START, full_end,
                               use_continuous_drawdown=True),
    }

    eligible = [x for x in candidates
                if x["train"]["trade_count"] >= 5
                and x["train"]["max_drawdown"] >= -0.15
                and x["train"]["cagr"] is not None]
    chosen = max(eligible, key=lambda x: (x["train"]["cagr"], x["train"]["win_rate"] or -1,
                                          x["train"]["max_drawdown"])) if eligible else None
    chosen_curve_key = None
    if chosen:
        chosen_curve_key = f"dd{int(chosen['drawdown_stop_pct']*100):02d}_ma{chosen['reentry_ma']}"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "drawdown_stop_pct": x["drawdown_stop_pct"], "reentry_ma": x["reentry_ma"],
        **{f"train_{k}": v for k, v in x["train"].items() if k in {"cagr","max_drawdown","trade_count","win_rate","exposure_ratio"}},
        **{f"oos_{k}": v for k, v in x["oos"].items() if k in {"cagr","max_drawdown","trade_count","win_rate","exposure_ratio"}},
        "account_stop_count": x["account_stop_count"],
        "account_reentry_count": x["account_reentry_count"],
        "blocked_limit_orders": x["blocked_limit_orders"],
        "missing_price_limit_orders": x["missing_price_limit_orders"],
    } for x in candidates]).to_csv(out / "event_driven_drawdown_grid.csv", index=False)
    if chosen_curve_key:
        equity_frames[chosen_curve_key].to_csv(out / "selected_event_driven_equity.csv", index=False)
    baseline_curve.to_csv(out / "baseline_event_driven_equity.csv", index=False)
    report = {
        "symbol": "2330",
        "frozen_rule_source": str(rule_search_path),
        "frozen_rule_parameters": params,
        "train_period": [TRAIN_START, TRAIN_END],
        "oos_period": [OOS_START, full_end],
        "selection_rule": "training CAGR maximum subject to >=5 closed trades and training max drawdown >= -15%; OOS not used for selection",
        "slippage_bps_each_side": SLIPPAGE_BPS,
        "reentry_cooldown_sessions": 5,
        "reentry_signal": "causal adjusted close above the selected SMA (20/50/100/200), evaluated at close and executed next session open",
        "baseline": baseline_metrics,
        "selected_overlay": chosen,
        "selected_overlay_curve_key": chosen_curve_key,
        "candidate_count": len(candidates),
        "all_candidates": candidates,
        "target_status": "NOT_ESTABLISHED",
        "execution": "event-driven next-session-open fills, official daily price limits, integer shares, commission, sell tax, adverse slippage, cash dividends",
        "warnings": [
            "This is a historical research screen, not a live-trading guarantee.",
            "The 2022+ period has been examined in prior project research and is not a pristine blind holdout.",
            "A stop signal is generated at close and cannot fill until the next available open; locked limit-down exits remain pending.",
            "The account risk-control reference peak resets on successful re-entry; reported maximum drawdown remains measured against the continuous account equity high-water mark.",
            "No queue priority, intraday liquidity, or exact historical order-book fills are modeled."
        ],
    }
    (out / "event_driven_drawdown_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--limits", required=True)
    ap.add_argument("--dividends", required=True)
    ap.add_argument("--rule-search", required=True)
    ap.add_argument("--output", default="artifacts/tsmc_2330_event_driven_drawdown")
    args = ap.parse_args()
    print(json.dumps(run(args.features, args.limits, args.dividends, args.rule_search, args.output),
                     ensure_ascii=False, indent=2, default=float))
