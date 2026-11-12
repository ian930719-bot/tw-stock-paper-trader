"""Causal post-exit trend re-entry and false-high diagnostic research.

This version supersedes the initial scale-mismatched diagnostic. It does not tune or replace the frozen historical OOF strategy. It
replays the existing 2015-2026 OOF trade exits as event anchors and evaluates
predeclared, close-known re-entry rules on a separate diagnostic ledger.
Because the exit anchors were learned from previously inspected history, this
is exploratory diagnostics, NOT independent OOF or a candidate performance
claim. No forward extrema are used as features or triggers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from tsmc_2330_40pct_rule_search import inputs

REENTRY_RULES = (
    {"name": "break_prev_high", "cooldown": 0, "condition": "raw_close > prev_raw_high"},
    {"name": "ema20_reclaim", "cooldown": 0, "condition": "distance_ema_20 crosses from <= 0 to > 0"},
    {"name": "rsi_recovery", "cooldown": 0, "condition": "rsi14 >= 50 and rsi14 > rsi14_prev"},
    {"name": "macd_momentum", "cooldown": 0, "condition": "macd_hist_pct > macd_hist_pct_prev"},
    {"name": "ema_20_rsi_macd", "cooldown": 0, "condition": "raw_close > ema_20 and rsi14 >= 50 and macd_hist_pct > macd_hist_pct_prev"},
    {"name": "ema_20_rsi_macd_cd10", "cooldown": 10, "condition": "raw_close > ema_20 and rsi14 >= 50 and macd_hist_pct > macd_hist_pct_prev"},
    {"name": "ema_20_rsi_macd_cd20", "cooldown": 20, "condition": "raw_close > ema_20 and rsi14 >= 50 and macd_hist_pct > macd_hist_pct_prev"},
)

REQUIRED = {
    "raw_open", "raw_high", "raw_low", "raw_close", "distance_ema_20",
    "rsi14", "macd_hist_pct", "relative_volume20", "atr_pct14",
    "sma20_slope_5", "sma50_slope_10", "obv_flow_ratio_5",
    "close_location", "intraday_range_pct",
}


def _signals(bars: pd.DataFrame) -> dict[str, pd.Series]:
    prev_high = bars["raw_high"].shift(1)
    rsi_prev = bars["rsi14"].shift(1)
    macd_prev = bars["macd_hist_pct"].shift(1)
    common = bars["raw_close"].notna() & bars["raw_open"].notna()
    return {
        "break_prev_high": common & (bars["raw_close"] > prev_high),
        "ema20_reclaim": common & (bars["distance_ema_20"].shift(1) <= 0) & (bars["distance_ema_20"] > 0),
        "rsi_recovery": common & (bars["rsi14"] >= 50) & (bars["rsi14"] > rsi_prev),
        "macd_momentum": common & (bars["macd_hist_pct"] > macd_prev),
        "ema_20_rsi_macd": common & (bars["distance_ema_20"] > 0) &
                          (bars["rsi14"] >= 50) & (bars["macd_hist_pct"] > macd_prev),
        "ema_20_rsi_macd_cd10": common & (bars["distance_ema_20"] > 0) &
                               (bars["rsi14"] >= 50) & (bars["macd_hist_pct"] > macd_prev),
        "ema_20_rsi_macd_cd20": common & (bars["distance_ema_20"] > 0) &
                               (bars["rsi14"] >= 50) & (bars["macd_hist_pct"] > macd_prev),
    }


def _future_diagnostic(bars: pd.DataFrame, exit_i: int, horizon: int) -> dict:
    """Outcome-only post-exit audit; never passed to a trigger or selector."""
    stop = min(len(bars), exit_i + horizon + 1)
    future = bars.iloc[exit_i + 1:stop]
    if future.empty:
        return {"observations": 0, "future_max_return": None, "future_min_return": None}
    ref = float(bars.iloc[exit_i]["raw_close"])
    return {
        "observations": len(future),
        "future_max_return": float(future["raw_high"].max() / ref - 1),
        "future_min_return": float(future["raw_low"].min() / ref - 1),
    }


def run(features_path: str, limits_path: str, dividends_path: str,
        frozen_trades_path: str, output_dir: str) -> dict:
    bars, limit_lookup, dividend_lookup = inputs(features_path, limits_path, dividends_path)
    missing = sorted(REQUIRED - set(bars.columns))
    if missing:
        raise ValueError(f"Required causal features missing: {missing}")
    bars = bars.sort_values("date").reset_index(drop=True)
    bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
    exit_ledger = pd.read_csv(frozen_trades_path)
    if not {"slippage_bps_per_side", "trade_number", "entry_date", "exit_date"}.issubset(exit_ledger.columns):
        raise ValueError("Frozen trade ledger schema mismatch")
    base = exit_ledger.loc[exit_ledger.slippage_bps_per_side == 10].copy()
    idx = {d: i for i, d in enumerate(bars.date.dt.strftime("%Y-%m-%d"))}
    signals = _signals(bars)
    rows = []
    exit_rows = []
    for t in base.itertuples(index=False):
        exit_date = str(t.exit_date)
        if exit_date not in idx:
            continue
        exit_i = idx[exit_date]
        diag = _future_diagnostic(bars, exit_i, 90)
        exit_row = bars.iloc[exit_i]
        entry_row = {
            "trade_number": int(t.trade_number),
            "exit_date": exit_date,
            "exit_reference_close": float(exit_row.raw_close),
            "base_trade_return": float(t.net_return),
            "forward_90d_outcome_only": diag,
        }
        exit_rows.append(entry_row)
        for rule in REENTRY_RULES:
            start_i = exit_i + 1 + int(rule["cooldown"])
            trigger_i = None
            mask = signals[rule["name"]].to_numpy(bool)
            for i in range(start_i, len(bars) - 1):
                if mask[i]:
                    trigger_i = i
                    break
            row = {
                **entry_row,
                "rule": rule["name"],
                "cooldown_sessions": int(rule["cooldown"]),
                "trigger_date": bars.iloc[trigger_i].date.date().isoformat() if trigger_i is not None else None,
                "execution_date": bars.iloc[trigger_i + 1].date.date().isoformat() if trigger_i is not None else None,
                "execution_open": float(bars.iloc[trigger_i + 1].raw_open) if trigger_i is not None else None,
                "executable_next_open_available": trigger_i is not None,
            }
            if trigger_i is not None:
                exec_day = bars.iloc[trigger_i + 1].date.strftime("%Y-%m-%d")
                if exec_day not in limit_lookup:
                    row["execution_status"] = "blocked_missing_official_limit"
                else:
                    up, down = limit_lookup[exec_day]
                    op = float(bars.iloc[trigger_i + 1].raw_open)
                    row["execution_status"] = (
                        "blocked_locked_limit_up" if op >= up - 1e-8 else "eligible_for_execution_model"
                    )
                row["signal_known_at_close"] = True
                row["uses_future_price_for_signal"] = False
            else:
                row["execution_status"] = "no_trigger"
                row["signal_known_at_close"] = True
                row["uses_future_price_for_signal"] = False
            rows.append(row)

    result = {
        "symbol": "2330",
        "source_oof_trade_ledger": str(frozen_trades_path),
        "source_anchor": "historically inspected frozen OOF exits; diagnostic only",
        "purpose": "Post-exit causal trend-re-entry trigger timing and false-high outcome diagnostics",
        "reentry_rules": list(REENTRY_RULES),
        "trigger_timing": "condition calculated only from data available at close t; earliest order executes at t+1 open",
        "future_data_policy": "90-session future max/min returns are outcome-only audit fields, never used by trigger generation or rule selection",
        "candidate_status": "EXPLORATORY_DIAGNOSTIC_NOT_OOF",
        "corrected_price_scale_policy": "Raw OHLC is compared only with raw OHLC; EMA20 state uses the scale-neutral adjusted distance_ema_20 feature. Never compare raw_close directly with adjusted ema_20.",
        "base_exit_count": len(base),
        "rule_count": len(REENTRY_RULES),
        "event_rows": rows,
        "exit_outcome_diagnostics": exit_rows,
        "limitations": [
            "Exit anchors were generated by historical strategies and have been inspected; this is not a fresh independent sample.",
            "This diagnostic measures the first subsequent trigger per exit, not a full portfolio replay.",
            "No full-account CAGR, win rate or MDD claim is produced.",
            "Next-open limit-lock eligibility is checked, but fills, share sizing, fee, tax and slippage require the integrated portfolio replay.",
            "The frozen features and price-limit files require official source reconciliation before live trading."
        ]
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_trend_reentry_diagnostic.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(out / "tsmc_2330_trend_reentry_events.csv", index=False)
    pd.DataFrame(exit_rows).to_csv(out / "tsmc_2330_post_exit_outcomes_only.csv", index=False)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--limits", required=True)
    ap.add_argument("--dividends", required=True)
    ap.add_argument("--frozen-trades", required=True)
    ap.add_argument("--output", default="artifacts/tsmc_2330_50cagr_trend_reentry")
    a = ap.parse_args()
    summary = run(a.features, a.limits, a.dividends, a.frozen_trades, a.output)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"event_rows", "exit_outcome_diagnostics"}},
                     ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
