"""Predeclared causal regime-exit attribution experiment for TSMC 2330.

Uses one fixed entry/ordinary-high rule family and varies only regime exits.
All signals are close-confirmed at t and execute no earlier than next open.
Selection is restricted to 2010-2015; 2016-2020 and 2021+ are report-only.
The 2021+ segment is historically inspected and is not a pristine blind holdout.
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
BASE_PARAMS = {"low_rv": 1.1, "high_loc": 0.75, "high_dist": 0.08}
CANDIDATES = ("none", "sma50_2day", "macd_2day", "directional_2day",
              "two_of_three_2day", "two_of_three_1day")


def load_inputs(features_path: str, limits_path: str, dividends_path: str):
    d = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    required = {
        "date", "raw_open", "raw_high", "raw_low", "raw_close", "corporate_action_window",
        "return_1", "relative_volume20", "stoch_k14", "close_location",
        "obv_flow_ratio_5", "intraday_range_pct", "distance_sma_200",
        "macd_hist_change_3", "sma50_slope_10", "atr_pct14",
        "distance_sma_20", "distance_sma_50", "macd_hist_pct",
        "plus_di14", "minus_di14", "adx14",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Missing causal regime/execution features: {missing}")
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


def _bool(series: pd.Series) -> np.ndarray:
    return series.fillna(False).astype(bool).to_numpy()


def build_regime_rules(d: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    sma = (d.distance_sma_50 < 0) & (d.sma50_slope_10 < 0)
    macd = (d.macd_hist_pct < 0) & (d.macd_hist_change_3 < 0)
    directional = (
        (d.distance_sma_20 < 0) & (d.minus_di14 > d.plus_di14) & (d.adx14 >= 20)
    )
    votes = sma.astype(int) + macd.astype(int) + directional.astype(int)
    recovery = (d.distance_sma_20 > 0) & (d.macd_hist_pct > 0)
    raw = {
        "none": pd.Series(False, index=d.index),
        "sma50_2day": sma.rolling(2, min_periods=2).sum().eq(2),
        "macd_2day": macd.rolling(2, min_periods=2).sum().eq(2),
        "directional_2day": directional.rolling(2, min_periods=2).sum().eq(2),
        "two_of_three_2day": votes.rolling(2, min_periods=2).min().ge(2),
        "two_of_three_1day": votes.ge(2),
    }
    action = d.corporate_action_window.fillna(True).astype(bool)
    result = {}
    for name, signal in raw.items():
        result[name] = (_bool(signal & ~action), _bool(recovery & ~action))
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
    rules = build_regime_rules(d)
    end = d.date.max().date().isoformat()
    periods = {"research": RESEARCH, "validation": VALIDATION, "historical_test": ("2021-01-01", end)}
    rows = []
    for name in CANDIDATES:
        exit_signal, recovery = rules[name]
        item = {"candidate": name, "regime_exit_count": int(exit_signal.sum()), "periods": {}}
        for period, (start, stop) in periods.items():
            metrics = _simulate_period(
                d, low, high, limits, dividends, start, stop, 10, INITIAL_CAPITAL, True,
                atr_stop_mask=atr_mask, regime_exit_signal=None if name == "none" else exit_signal,
                regime_entry_allowed=None if name == "none" else recovery,
            )
            item["periods"][period] = compact(metrics)
            item.setdefault("trades", {})[period] = metrics["trades"]
        rows.append(item)

    eligible = [r for r in rows if r["periods"]["research"]["trade_count"] >= 5
                and r["periods"]["research"]["win_rate"] is not None
                and r["periods"]["research"]["win_rate"] >= WIN_FLOOR]
    selected = max(eligible, key=lambda r: (
        r["periods"]["research"]["cagr"] if r["periods"]["research"]["cagr"] is not None else -1,
        r["periods"]["research"]["max_drawdown"],
    )) if eligible else None

    stress = {}
    if selected:
        exit_signal, recovery = rules[selected["candidate"]]
        for period, (start, stop) in periods.items():
            stress[period] = {}
            for slip in SLIPPAGE_BPS:
                metrics = _simulate_period(
                    d, low, high, limits, dividends, start, stop, slip, INITIAL_CAPITAL, True,
                    atr_stop_mask=atr_mask,
                    regime_exit_signal=None if selected["candidate"] == "none" else exit_signal,
                    regime_entry_allowed=None if selected["candidate"] == "none" else recovery,
                )
                stress[period][f"{slip}bps"] = compact(metrics)

    result = {
        "symbol": "2330", "baseline_entry_exit_parameters": BASE_PARAMS,
        "candidate_count": len(rows), "candidates": rows,
        "selection_period": list(RESEARCH), "validation_period": list(VALIDATION),
        "historical_test_period": ["2021-01-01", end],
        "selection_rule": "research-only; >=5 trades and win rate >=86.9565%, then maximize research CAGR; validation/test are report-only",
        "selected_candidate": None if selected is None else selected["candidate"],
        "selected_cost_stress": stress, "official_price_limit_coverage": limit_coverage,
        "target": {"cagr_min": 0.50, "win_rate_min": WIN_FLOOR, "max_drawdown_min": -0.10},
        "target_status": "NOT_ESTABLISHED; historical test is previously inspected and not pristine OOS",
        "signal_timing": "all regime inputs use close t; exit and re-entry execute no earlier than next available session open",
        "recovery_gate": "after regime exit, re-entry requires close above SMA20 and positive MACD histogram; recovery close t permits next-session entry only",
        "costs": "shared simulator: commission, minimum commission, sell tax, adverse slippage, cash dividends and official daily price-limit lock handling",
        "data_warning": "input price history source is FinMind and manifest has official_exchange_reconciled=false; this run is not an official-data audit",
        "caveats": ["2021+ was inspected in prior research and is not a pristine blind holdout",
                    "daily OHLC cannot guarantee actual fills or queue priority",
                    "no candidate is eligible if research trade count or win-rate floor is not met"],
    }
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_regime_exit_attribution.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    flat = []
    for row in rows:
        flat.append({"candidate": row["candidate"], "regime_exit_count": row["regime_exit_count"],
                     **{f"{p}_{k}": v for p, m in row["periods"].items() for k, v in m.items()}})
    pd.DataFrame(flat).to_csv(out / "tsmc_2330_regime_exit_attribution.csv", index=False)
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
