"""Causal SMA200 structural-regime exit/recovery experiment for TSMC 2330.

The existing LOW/HIGH entry-exit signals are frozen. This experiment only tests
slow structural bear-regime exits and recovery gates. All signals are known at
close t and the shared simulator executes no earlier than the next session open.
Selection uses 2010-2015 only; 2016-2020 and 2021+ are report-only.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_execution_backtest import INITIAL_CAPITAL, SLIPPAGE_BPS, _simulate_period
from tsmc_2330_atr_reentry_rule_search import make_signals

RESEARCH = ("2010-01-04", "2015-12-31")
VALIDATION = ("2016-01-04", "2020-12-31")
WIN_FLOOR = 0.8695652173913043
MAX_DRAWDOWN_FLOOR = -0.10
BASE_PARAMS = {"low_rv": 1.1, "high_loc": 0.75, "high_dist": 0.08}
EXIT_CONFIRMATIONS = (2, 3, 5)
RECOVERY_CONFIRMATIONS = (1, 2, 3)


def confirm_consecutive(signal: pd.Series | np.ndarray, sessions: int) -> np.ndarray:
    if sessions < 1:
        raise ValueError("sessions must be >= 1")
    values = pd.Series(signal).fillna(False).astype(bool)
    return values.rolling(sessions, min_periods=sessions).sum().eq(sessions).to_numpy(dtype=bool)


def load_inputs(features_path: str, limits_path: str, dividends_path: str):
    d = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    required = {
        "date", "raw_open", "raw_high", "raw_low", "raw_close", "corporate_action_window",
        "return_1", "relative_volume20", "stoch_k14", "close_location",
        "obv_flow_ratio_5", "intraday_range_pct", "distance_sma_200",
        "macd_hist_change_3", "sma50_slope_10", "atr_pct14",
        "ema200_slope_20", "ema50_slope_10",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Missing causal SMA200 regime features: {missing}")

    lim = pd.read_csv(limits_path, dtype={"stock_id": str})
    lim = lim.loc[lim.stock_id.astype(str).str.strip() == "2330"].copy()
    lim["date"] = pd.to_datetime(lim.date, errors="coerce").dt.normalize()
    for col in ("limit_up", "limit_down"):
        lim[col] = pd.to_numeric(lim[col], errors="coerce")
    lim = lim.dropna(subset=["date", "limit_up", "limit_down"]).drop_duplicates("date", keep="last")
    limits = {r.date.strftime("%Y-%m-%d"): (float(r.limit_up), float(r.limit_down))
              for r in lim.itertuples(index=False)}
    coverage = d.date.dt.strftime("%Y-%m-%d").isin(limits).mean()
    if coverage < 0.98:
        raise RuntimeError(f"Official price-limit coverage below 98%: {coverage:.3%}")

    div = pd.read_csv(dividends_path)
    required_div = {"date", "stock_or_cache_dividend", "stock_and_cache_dividend"}
    if not required_div.issubset(div.columns):
        raise ValueError(f"Dividend file missing columns: {sorted(required_div-set(div.columns))}")
    div = div.loc[div.stock_or_cache_dividend.astype(str).str.strip() == "息"].copy()
    div["date"] = pd.to_datetime(div.date, errors="coerce").dt.normalize()
    div["stock_and_cache_dividend"] = pd.to_numeric(div.stock_and_cache_dividend, errors="coerce")
    div = div.dropna(subset=["date", "stock_and_cache_dividend"])
    dividends = {r.date.strftime("%Y-%m-%d"): float(r.stock_and_cache_dividend)
                 for r in div.itertuples(index=False)}
    return d, limits, dividends, float(coverage)


def build_candidates(d: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # Structural bear: price below SMA200 and the long EMA200 slope is negative.
    # Recovery: price regains SMA200 while the medium EMA50 slope is positive.
    bear_raw = (d.distance_sma_200 < 0) & (d.ema200_slope_20 < 0)
    recovery_raw = (d.distance_sma_200 > 0) & (d.ema50_slope_10 > 0)
    action = d.corporate_action_window.fillna(True).astype(bool).to_numpy()
    result = {}
    for exit_n, recovery_n in itertools.product(EXIT_CONFIRMATIONS, RECOVERY_CONFIRMATIONS):
        exit_signal = confirm_consecutive(bear_raw, exit_n) & ~action
        recovery_signal = confirm_consecutive(recovery_raw, recovery_n) & ~action
        result[f"exit{exit_n}_recover{recovery_n}"] = (exit_signal, recovery_signal)
    return result


def compact(m: dict) -> dict:
    keys = ("period_start", "period_end", "final_equity", "total_return", "cagr",
            "max_drawdown", "trade_count", "win_rate", "exposure_ratio",
            "total_commission", "total_sell_tax", "estimated_slippage_cost",
            "blocked_limit_orders", "missing_price_limit_orders")
    return {k: m.get(k) for k in keys}


def run(features_path: str, limits_path: str, dividends_path: str, outdir: str) -> dict:
    d, limits, dividends, limit_coverage = load_inputs(features_path, limits_path, dividends_path)
    low, high, atr_mask = make_signals(d, **BASE_PARAMS)
    candidates = build_candidates(d)
    end = d.date.max().date().isoformat()
    periods = {
        "research": RESEARCH,
        "validation": VALIDATION,
        "historical_test": ("2021-01-01", end),
    }
    rows = []
    for name, (exit_signal, recovery_signal) in candidates.items():
        item = {"candidate": name, "exit_signal_days": int(exit_signal.sum()),
                "recovery_signal_days": int(recovery_signal.sum()), "periods": {}}
        for period_name, (start, stop) in periods.items():
            metrics = _simulate_period(
                d, low, high, limits, dividends, start, stop, 10, INITIAL_CAPITAL, True,
                atr_stop_mask=atr_mask, regime_exit_signal=exit_signal,
                regime_entry_allowed=recovery_signal,
            )
            item["periods"][period_name] = compact(metrics)
            item.setdefault("trades", {})[period_name] = metrics["trades"]
        rows.append(item)

    # Research-only candidate selection. Validation/test never affect selection.
    eligible = [r for r in rows
                if r["periods"]["research"]["trade_count"] >= 5
                and r["periods"]["research"]["win_rate"] is not None
                and r["periods"]["research"]["win_rate"] >= WIN_FLOOR]
    selected = max(eligible, key=lambda r: (
        r["periods"]["research"]["cagr"] if r["periods"]["research"]["cagr"] is not None else -1,
        r["periods"]["research"]["max_drawdown"],
    )) if eligible else None
    target_eligible = [r for r in rows
        if r["periods"]["research"]["trade_count"] >= 5
        and r["periods"]["research"]["win_rate"] is not None
        and r["periods"]["research"]["win_rate"] >= WIN_FLOOR
        and r["periods"]["research"]["cagr"] is not None
        and r["periods"]["research"]["cagr"] >= 0.50
        and r["periods"]["research"]["max_drawdown"] is not None
        and r["periods"]["research"]["max_drawdown"] >= MAX_DRAWDOWN_FLOOR]

    stress = {}
    if selected:
        exit_signal, recovery_signal = candidates[selected["candidate"]]
        for period_name, (start, stop) in periods.items():
            stress[period_name] = {}
            for slip in SLIPPAGE_BPS:
                metrics = _simulate_period(
                    d, low, high, limits, dividends, start, stop, slip, INITIAL_CAPITAL, True,
                    atr_stop_mask=atr_mask, regime_exit_signal=exit_signal,
                    regime_entry_allowed=recovery_signal,
                )
                stress[period_name][f"{slip}bps"] = compact(metrics)

    result = {
        "symbol": "2330",
        "baseline_entry_exit_parameters": BASE_PARAMS,
        "candidate_count": len(rows),
        "candidates": rows,
        "selection_period": list(RESEARCH),
        "validation_period": list(VALIDATION),
        "historical_test_period": ["2021-01-01", end],
        "selection_rule": "research-only; >=5 closed trades and win rate >=86.9565%, then maximize CAGR; validation/test report-only",
        "selected_candidate": None if selected is None else selected["candidate"],
        "selected_cost_stress": stress,
        "research_candidates_meeting_all_three_targets": [r["candidate"] for r in target_eligible],
        "official_price_limit_coverage": limit_coverage,
        "target": {"cagr_min": 0.50, "win_rate_min": WIN_FLOOR,
                   "max_drawdown_min": MAX_DRAWDOWN_FLOOR},
        "target_status": "NOT_ESTABLISHED; no candidate meets all three research-period constraints" if not target_eligible else "RESEARCH_ONLY_CANDIDATE; must still pass validation and forward paper trading",
        "signal_timing": "bear/recovery conditions use close t; all exits and re-entries execute no earlier than next available session open",
        "bear_definition": "close below SMA200 and EMA200 20-session slope negative for 2/3/5 consecutive sessions",
        "recovery_definition": "close above SMA200 and EMA50 10-session slope positive for 1/2/3 consecutive sessions",
        "costs": "shared simulator: commission, minimum commission, sell tax, adverse slippage, cash dividends and official daily price-limit lock handling",
        "data_warning": "FinMind OHLCV is not independently reconciled row-by-row against TWSE official files",
        "caveats": [
            "2021+ has been examined in prior research and is not a pristine blind holdout",
            "daily OHLC cannot guarantee queue priority or exact fills",
            "SMA200 is a lagging regime filter; delayed exits and whipsaws remain possible",
            "a candidate selected on 2010-2015 is exploratory and requires future paper validation",
        ],
    }
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_sma200_regime_research.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    flat = []
    for r in rows:
        flat.append({
            "candidate": r["candidate"],
            "exit_signal_days": r["exit_signal_days"],
            "recovery_signal_days": r["recovery_signal_days"],
            **{f"{p}_{k}": v for p, m in r["periods"].items() for k, v in m.items()},
        })
    pd.DataFrame(flat).to_csv(out / "tsmc_2330_sma200_regime_research.csv", index=False)
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
