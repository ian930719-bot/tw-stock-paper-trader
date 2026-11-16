#!/usr/bin/env python3
"""Causal drawdown circuit-breaker screen for the frozen TSMC equity curve.

This is a screening proxy, not an execution-grade backtest: it applies the
frozen strategy's daily close-to-close return stream when exposed, and zero
return when flat. It does not invent next-open fills or model incremental
turnover costs. A passing candidate must be re-run in the full event-driven
simulator before acceptance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

SPLIT_DATE = "2022-01-03"
INITIAL_CAPITAL = 100_000.0


def load_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "equity", "shares", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for col in ["equity", "shares", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["date", "equity", "shares", "close"]].isna().any().any():
        raise ValueError("Missing or invalid date/equity/shares/close rows")
    if (df.equity <= 0).any() or (df.close <= 0).any():
        raise ValueError("Equity and close must be positive")
    if not df.date.is_monotonic_increasing or len(df) < 500:
        raise ValueError("Curve must be ordered and contain at least 500 sessions")
    return df


def run_overlay(df: pd.DataFrame, trigger: float, reentry_ma: int, min_off_days: int = 5):
    close = df.close
    ma = close.rolling(reentry_ma, min_periods=reentry_ma).mean()
    base_returns = df.equity.pct_change().fillna(0.0)
    equity = INITIAL_CAPITAL
    high_water = INITIAL_CAPITAL
    is_on = True
    off_days = 0
    rows = []
    for i in range(len(df)):
        if i > 0:
            prior_dd = equity / high_water - 1.0
            if is_on and prior_dd <= -trigger:
                is_on = False
                off_days = 0
            elif not is_on:
                off_days += 1
                # Re-entry uses only the prior close, prior baseline position,
                # and prior rolling average; execution is not assumed intraday.
                if (off_days >= min_off_days and
                    float(df.shares.iloc[i - 1]) > 0 and
                    float(close.iloc[i - 1]) > float(ma.iloc[i - 1])):
                    is_on = True
            daily_return = float(base_returns.iloc[i]) if is_on else 0.0
            equity *= 1.0 + daily_return
            high_water = max(high_water, equity)
        rows.append({
            "date": df.date.iloc[i],
            "equity": equity,
            "exposure": 1 if is_on else 0,
            "drawdown_from_full_history_peak": equity / high_water - 1.0,
        })
    return pd.DataFrame(rows)


def summarize(curve: pd.DataFrame, start: str | None = None, end: str | None = None):
    x = curve.copy()
    if start:
        x = x[x.date >= pd.Timestamp(start)]
    if end:
        x = x[x.date <= pd.Timestamp(end)]
    if len(x) < 2:
        raise ValueError("Insufficient rows in requested metric period")
    start_equity = float(x.equity.iloc[0])
    end_equity = float(x.equity.iloc[-1])
    days = max(1, (x.date.iloc[-1] - x.date.iloc[0]).days)
    cagr = (end_equity / start_equity) ** (365.25 / days) - 1.0
    # Drawdown uses the account's continuous high-water mark, including the
    # pre-test period; it is not reset at the OOS boundary.
    return {
        "period_start": x.date.iloc[0].date().isoformat(),
        "period_end": x.date.iloc[-1].date().isoformat(),
        "start_equity": start_equity,
        "end_equity": end_equity,
        "cagr": float(cagr),
        "max_drawdown_continuous_hwm": float(x.drawdown_from_full_history_peak.min()),
        "exposure_ratio": float(x.exposure.mean()),
        "sessions": int(len(x)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = load_curve(source)
    split = pd.Timestamp(SPLIT_DATE)
    if df.date.min() > pd.Timestamp("2015-01-05") or df.date.max() < pd.Timestamp("2026-11-11"):
        raise ValueError("Unexpected baseline period; refusing partial/stale input")

    base = pd.DataFrame({
        "date": df.date,
        "equity": df.equity / float(df.equity.iloc[0]) * INITIAL_CAPITAL,
        "exposure": (df.shares > 0).astype(int),
    })
    base["drawdown_from_full_history_peak"] = base.equity / base.equity.cummax() - 1.0

    candidates = []
    curves = {}
    for trigger in [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
        for ma_n in [20, 50, 100, 200]:
            c = run_overlay(df, trigger, ma_n, min_off_days=5)
            key = f"dd{int(trigger*100):02d}_ma{ma_n}"
            curves[key] = c
            train = summarize(c, end="2021-12-31")
            test = summarize(c, start=SPLIT_DATE)
            candidates.append({
                "candidate": key,
                "drawdown_trigger": trigger,
                "reentry_ma": ma_n,
                "min_off_sessions": 5,
                **{f"train_{k}": v for k, v in train.items() if k not in ["period_start", "period_end"]},
                **{f"test_{k}": v for k, v in test.items() if k not in ["period_start", "period_end"]},
                "test_start": test["period_start"],
                "test_end": test["period_end"],
            })
    grid = pd.DataFrame(candidates)
    eligible = grid[grid.train_max_drawdown_continuous_hwm >= -0.15].copy()
    if eligible.empty:
        raise RuntimeError("No training candidate met the predeclared -15% drawdown screen")
    # Freeze one candidate using training data only; no test-period selection.
    selected = eligible.sort_values(
        ["train_cagr", "train_max_drawdown_continuous_hwm"],
        ascending=[False, False],
    ).iloc[0]
    selected_key = str(selected.candidate)
    selected_curve = curves[selected_key]
    baseline_train = summarize(base, end="2021-12-31")
    baseline_test = summarize(base, start=SPLIT_DATE)
    selected_train = summarize(selected_curve, end="2021-12-31")
    selected_test = summarize(selected_curve, start=SPLIT_DATE)

    grid.to_csv(out / "drawdown_overlay_grid.csv", index=False)
    selected_curve.to_csv(out / "selected_overlay_equity.csv", index=False)
    result = {
        "status": "SCREEN_COMPLETE_NOT_EXECUTION_GRADE",
        "source_file": str(source),
        "baseline": {"train": baseline_train, "test": baseline_test},
        "selection_protocol": {
            "training_period": ["2015-01-05", "2021-12-31"],
            "untouched_test_period": [SPLIT_DATE, df.date.max().date().isoformat()],
            "predeclared_training_constraint": "continuous high-water-mark max drawdown >= -15%",
            "selection_objective": "highest training CAGR; ties prefer shallower training drawdown",
            "reentry": "prior session baseline shares > 0 and prior close > prior rolling MA",
            "minimum_off_sessions": 5,
            "test_period_not_used_for_selection": True,
        },
        "selected_candidate": {
            "candidate": selected_key,
            "drawdown_trigger": float(selected.drawdown_trigger),
            "reentry_ma": int(selected.reentry_ma),
            "train": selected_train,
            "test": selected_test,
        },
        "limits": [
            "Daily close-to-close return proxy only; no next-open order simulation.",
            "No incremental fees, tax, spread, market impact, or limit-queue cost for overlay exits/re-entries.",
            "The frozen strategy equity curve is used as a shadow return stream and for re-entry eligibility.",
            "Win rate is not recalculated because the overlay can split/alter trades; original trade win rate must not be reused as overlay win rate.",
            "This screen cannot establish a deployable max-drawdown guarantee; selected rule must be rerun in the full event-driven simulator.",
        ],
    }
    (out / "drawdown_overlay_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
