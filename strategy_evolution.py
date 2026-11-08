"""Data-driven strategy search with a strict chronological holdout.

This is a small, transparent research search—not a claim of machine-learning alpha.
Signals use only current/past completed daily bars; fills are handled by the shared
capital simulator on the next session open.
"""
from __future__ import annotations

from itertools import product
from typing import Mapping

import numpy as np
import pandas as pd

from research_engine import shared_capital_backtest
from analytics import trade_statistics


def candidate_grid():
    """Return a bounded, pre-declared search space to limit data snooping."""
    for breakout, volume_multiple, trend_period in product(
        (20, 40, 60), (1.2, 1.5), (50, 100)
    ):
        yield {
            "breakout": breakout,
            "volume_multiple": volume_multiple,
            "trend_period": trend_period,
        }


def build_signals(frame: pd.DataFrame, params: Mapping[str, float]):
    z = frame.sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(z["close"], errors="coerce")
    high = pd.to_numeric(z["high"], errors="coerce")
    volume = pd.to_numeric(z["volume"], errors="coerce")
    breakout = int(params["breakout"])
    trend_period = int(params["trend_period"])
    prior_high = high.shift(1).rolling(breakout, min_periods=breakout).max()
    prior_volume = volume.shift(1).rolling(20, min_periods=20).mean()
    trend = close.rolling(trend_period, min_periods=trend_period).mean()
    momentum = close.pct_change(20)
    ema21 = close.ewm(span=21, adjust=False).mean()
    entry = (
        (close > prior_high)
        & (volume > prior_volume * float(params["volume_multiple"]))
        & (close > trend)
        & (momentum > 0)
    ).fillna(False)
    exit_signal = ((close < ema21) | (close < trend)).fillna(False)
    return entry, exit_signal


def _prepare_overrides(frames, params):
    return {code: build_signals(frame, params) for code, frame in frames.items()}


def _score(result):
    metrics = result["metrics"]
    trades = result["trades"]
    stats = trade_statistics(trades)
    n = int(stats.get("交易筆數", len(trades)))
    pf = float(stats.get("Profit Factor", 0.0))
    if not np.isfinite(pf):
        pf = 3.0
    cagr = float(metrics.get("年化報酬", -1.0))
    drawdown = abs(min(0.0, float(metrics.get("最大回撤", 0.0))))
    sharpe = float(metrics.get("Sharpe", 0.0))
    # Penalize fragile candidates and severe drawdowns. Score is a ranking
    # heuristic only; holdout metrics are never used to select parameters.
    if n < 8 or not np.isfinite(cagr) or not np.isfinite(sharpe):
        return -10.0
    return cagr + 0.10 * np.clip(sharpe, -2, 3) + 0.08 * np.log(max(pf, 0.05)) - 0.75 * drawdown


def evolve_strategy(
    frames,
    initial_capital=100_000,
    max_positions=5,
    risk_fraction=0.005,
    exit_profile="baseline_atr2",
    warmup=260,
):
    """Select on first 70% chronologically; evaluate once on final 30%."""
    clean = {
        code: frame.sort_values("date").reset_index(drop=True).copy()
        for code, frame in frames.items()
        if frame is not None and not frame.empty
    }
    if not clean:
        return {"status": "no_data", "candidates": pd.DataFrame()}
    common_dates = sorted(set().union(*[
        set(pd.to_datetime(frame["date"])) for frame in clean.values()
    ]))
    if len(common_dates) < warmup + 80:
        return {
            "status": "insufficient_history",
            "required_sessions": warmup + 80,
            "available_sessions": len(common_dates),
            "candidates": pd.DataFrame(),
        }
    split_index = int(len(common_dates) * 0.70)
    split_date = pd.Timestamp(common_dates[split_index])
    train_frames = {
        code: frame[pd.to_datetime(frame["date"]) < split_date].copy()
        for code, frame in clean.items()
    }
    rows = []
    best = None
    best_score = -np.inf
    for params in candidate_grid():
        overrides = _prepare_overrides(train_frames, params)
        result = shared_capital_backtest(
            train_frames, "AI 演化候選",
            initial_capital=initial_capital,
            max_positions=max_positions,
            risk_fraction=risk_fraction,
            warmup=warmup,
            exit_profile=exit_profile,
            signal_overrides=overrides,
        )
        score = _score(result)
        stats = trade_statistics(result["trades"])
        row = {
            **params,
            "訓練年化報酬": result["metrics"].get("年化報酬", 0.0),
            "訓練最大回撤": result["metrics"].get("最大回撤", 0.0),
            "訓練Profit Factor": stats.get("Profit Factor", 0.0),
            "訓練交易筆數": stats.get("交易筆數", len(result["trades"])),
            "研究分數": score,
        }
        rows.append(row)
        if score > best_score:
            best_score, best = score, params

    # One untouched chronological holdout. No holdout metric influences selection.
    holdout_overrides = _prepare_overrides(clean, best)
    holdout = shared_capital_backtest(
        clean, "AI 演化候選",
        initial_capital=initial_capital,
        max_positions=max_positions,
        risk_fraction=risk_fraction,
        warmup=warmup,
        exit_profile=exit_profile,
        trading_start_date=split_date,
        report_start_date=split_date,
        signal_overrides=holdout_overrides,
    )
    holdout_stats = trade_statistics(holdout["trades"])
    annual_rows = []
    if not holdout["equity"].empty:
        annual_equity = holdout["equity"].copy()
        annual_equity["年度"] = pd.to_datetime(annual_equity["日期"]).dt.year
        for year, group in annual_equity.groupby("年度", sort=True):
            start_value = float(group["資產"].iloc[0])
            end_value = float(group["資產"].iloc[-1])
            annual_return = end_value / start_value - 1.0 if start_value > 0 else -1.0
            annual_rows.append({
                "年度": int(year),
                "年初資產（樣本外首日）": start_value,
                "年底資產": end_value,
                "年度報酬": annual_return,
                "100%目標達成": bool(annual_return >= 1.0),
            })
    annual_performance = pd.DataFrame(annual_rows)
    return {
        "status": "ok",
        "selected_params": dict(best),
        "selection_score": float(best_score),
        "split_date": split_date,
        "train_rows": pd.DataFrame(rows).sort_values("研究分數", ascending=False),
        "holdout": holdout,
        "holdout_stats": holdout_stats,\n        "annual_performance": annual_performance,\n        "annual_100pct_years": int(annual_performance["100%目標達成"].sum()) if not annual_performance.empty else 0,
        "search_count": len(rows),
        "symbols": len(clean),
    }
