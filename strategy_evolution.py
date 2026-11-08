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
from multifactor_engine import multifactor_signals
from corporate_actions import point_in_time_recalculate


def candidate_grid():
    """Return a bounded, pre-declared search space to limit data snooping."""
    for breakout, volume_multiple, trend_period in product(
        (20, 40, 60), (1.2, 1.5), (50, 100)
    ):
        yield {
            "signal_family": "breakout",
            "breakout": breakout,
            "volume_multiple": volume_multiple,
            "trend_period": trend_period,
            "min_entry_score": 4,
            "max_atr_pct": 0.12,
        }
    for min_score, atr_cap in product((3, 4, 5), (0.08, 0.12)):
        yield {
            "signal_family": "multifactor",
            "breakout": 0,
            "volume_multiple": 0.0,
            "trend_period": 50,
            "min_entry_score": min_score,
            "max_atr_pct": atr_cap,
        }
    for rsi_floor, band_multiple in product((25, 30, 35), (2.0, 2.5)):
        yield {
            "signal_family": "mean_reversion",
            "breakout": 0, "volume_multiple": 0.0, "trend_period": 200,
            "rsi_floor": rsi_floor, "band_multiple": band_multiple,
        }
    for squeeze_quantile, breakout in product((0.15, 0.20, 0.25), (20, 40)):
        yield {
            "signal_family": "squeeze_breakout",
            "breakout": breakout, "volume_multiple": 1.2, "trend_period": 50,
            "squeeze_quantile": squeeze_quantile,
        }


def build_signals(frame: pd.DataFrame, params: Mapping[str, float]):
    z = frame.sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(z["close"], errors="coerce")
    high = pd.to_numeric(z["high"], errors="coerce")
    volume = pd.to_numeric(z["volume"], errors="coerce")
    family = params.get("signal_family", "breakout")
    if family == "multifactor":
        entry, exit_signal, _ = multifactor_signals(
            z, min_entry_score=int(params["min_entry_score"]),
            max_atr_pct=float(params["max_atr_pct"]),
        )
        return entry, exit_signal
    if family == "mean_reversion":
        sma20 = close.rolling(20, min_periods=20).mean()
        sma200 = close.rolling(200, min_periods=200).mean()
        std20 = close.rolling(20, min_periods=20).std(ddof=0)
        lower = sma20 - float(params["band_multiple"]) * std20
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        entry = ((close > sma200) & (close < lower) &
                 (rsi < int(params["rsi_floor"]))).fillna(False)
        exit_signal = ((close > sma20) | (rsi > 55)).fillna(False)
        return entry, exit_signal
    if family == "squeeze_breakout":
        breakout = int(params["breakout"])
        sma20 = close.rolling(20, min_periods=20).mean()
        std20 = close.rolling(20, min_periods=20).std(ddof=0)
        bandwidth = 4 * std20 / sma20.replace(0, np.nan)
        quantile = float(params["squeeze_quantile"])
        threshold = bandwidth.shift(2).rolling(120, min_periods=60).quantile(quantile)
        recent_squeeze = (bandwidth.shift(1) <= threshold).rolling(10, min_periods=1).max().fillna(0).astype(bool)
        prior_high = high.shift(1).rolling(breakout, min_periods=breakout).max()
        prior_volume = volume.shift(1).rolling(20, min_periods=20).mean()
        trend = close.rolling(50, min_periods=50).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        momentum = close.pct_change(20)
        entry = ((close > prior_high) &
                 (volume > prior_volume * float(params["volume_multiple"])) &
                 (close > trend) & recent_squeeze & (momentum > 0)).fillna(False)
        exit_signal = ((close < ema21) | (close < trend)).fillna(False)
        return entry, exit_signal
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


def _candidate_features(frame, params):
    entry, exit_signal = build_signals(frame, params)
    return pd.DataFrame({
        "entry": pd.Series(entry).reset_index(drop=True).fillna(False).astype(bool),
        "exit": pd.Series(exit_signal).reset_index(drop=True).fillna(False).astype(bool),
    })


def _prepare_overrides(frames, params, corporate_actions=None):
    result = {}
    for code, frame in frames.items():
        events = (corporate_actions or {}).get(str(code))
        features = point_in_time_recalculate(
            frame, events, lambda adjusted: _candidate_features(adjusted, params)
        )
        result[code] = (
            features["entry"].astype(bool),
            features["exit"].astype(bool),
        )
    return result


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



def select_candidate(
    frames,
    *,
    initial_capital=100_000,
    max_positions=5,
    risk_fraction=0.005,
    exit_profile="baseline_atr2",
    warmup=260,
    min_validation_trades=8,
    signal_cache=None,
    fee=0.001425,
    sell_tax=0.003,
    slippage=0.001,
    corporate_actions=None,
):
    """Choose a candidate using an inner chronological validation slice only.

    The final outer walk-forward test window is never passed into this function.
    Candidates are ranked on the later part of the supplied training history;
    the earlier part supplies indicator warm-up and initial training context.
    """
    clean = {
        code: frame.sort_values("date").reset_index(drop=True).copy()
        for code, frame in frames.items()
        if frame is not None and not frame.empty and "date" in frame
    }
    if not clean:
        return {"status": "no_data", "selected_params": None, "ranking": pd.DataFrame()}
    dates = sorted(set().union(*[
        set(pd.to_datetime(frame["date"])) for frame in clean.values()
    ]))
    if len(dates) < int(warmup) + 100:
        return {"status": "insufficient_history", "selected_params": None,
                "available_sessions": len(dates), "ranking": pd.DataFrame()}
    split = int(len(dates) * 0.70)
    validation_start = pd.Timestamp(dates[split])
    rows = []
    best_params, best_score = None, -np.inf
    for params in candidate_grid():
        if signal_cache is None:
            overrides = _prepare_overrides(clean, params, corporate_actions)
        else:
            key = tuple(sorted(params.items()))
            overrides = {}
            for code, frame in clean.items():
                cached = signal_cache.get((code, key))
                if cached is None:
                    overrides[code] = _prepare_overrides({code: frame}, params, corporate_actions)[code]
                else:
                    dates_index = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
                    overrides[code] = tuple(
                        series.reindex(dates_index).fillna(False).reset_index(drop=True)
                        for series in cached
                    )
        result = shared_capital_backtest(
            clean, "AI 候選內部驗證",
            initial_capital=initial_capital,
            max_positions=max_positions,
            risk_fraction=risk_fraction,
            fee=fee, sell_tax=sell_tax, slippage=slippage,
            warmup=warmup,
            exit_profile=exit_profile,
            trading_start_date=validation_start,
            report_start_date=validation_start,
            signal_overrides=overrides,
            corporate_actions=corporate_actions,
        )
        stats = trade_statistics(result["trades"])
        trades = int(stats.get("交易筆數", len(result["trades"])))
        score = _score(result) if trades >= int(min_validation_trades) else -10.0
        rows.append({
            **params,
            "驗證年化報酬": result["metrics"].get("年化報酬", 0.0),
            "驗證最大回撤": result["metrics"].get("最大回撤", 0.0),
            "驗證Sharpe": result["metrics"].get("Sharpe", 0.0),
            "驗證Profit Factor": stats.get("Profit Factor", 0.0),
            "驗證交易筆數": trades,
            "研究分數": score,
        })
        if score > best_score:
            best_score, best_params = score, dict(params)
    ranking = pd.DataFrame(rows).sort_values("研究分數", ascending=False)
    if best_params is None or best_score <= -10.0:
        return {"status": "no_robust_candidate", "selected_params": None,
                "validation_start": validation_start, "ranking": ranking}
    return {"status": "ok", "selected_params": best_params,
            "selection_score": float(best_score),
            "validation_start": validation_start, "ranking": ranking}

def evolve_strategy(
    frames,
    initial_capital=100_000,
    max_positions=5,
    risk_fraction=0.005,
    exit_profile="baseline_atr2",
    warmup=260,
    corporate_actions=None,
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
    selection = select_candidate(
        train_frames,
        initial_capital=initial_capital,
        max_positions=max_positions,
        risk_fraction=risk_fraction,
        exit_profile=exit_profile,
        warmup=warmup,
        corporate_actions=corporate_actions,
    )
    if selection.get("status") != "ok":
        return {
            "status": selection.get("status", "no_robust_candidate"),
            "split_date": split_date,
            "available_sessions": len(common_dates),
            "required_sessions": warmup + 100,
            "candidates": selection.get("ranking", pd.DataFrame()),
        }
    best = selection["selected_params"]
    best_score = selection["selection_score"]
    candidate_rows = selection["ranking"].copy()
    candidate_rows = candidate_rows.rename(columns={
        "驗證年化報酬": "訓練年化報酬",
        "驗證最大回撤": "訓練最大回撤",
        "驗證Profit Factor": "訓練Profit Factor",
        "驗證交易筆數": "訓練交易筆數",
    })
    # One untouched chronological holdout. No holdout metric influences selection.
    holdout_overrides = _prepare_overrides(clean, best, corporate_actions)
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
        corporate_actions=corporate_actions,
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
            year_start = pd.Timestamp(f"{int(year)}-01-01")
            year_end = pd.Timestamp(f"{int(year)}-12-31")
            observed_start = pd.Timestamp(group["日期"].iloc[0])
            observed_end = pd.Timestamp(group["日期"].iloc[-1])
            full_year = observed_start <= year_start + pd.Timedelta(days=7) and observed_end >= year_end - pd.Timedelta(days=7)
            annual_rows.append({
                "年度": int(year),
                "期間完整": bool(full_year),
                "樣本外起日": observed_start.date().isoformat(),
                "樣本外迄日": observed_end.date().isoformat(),
                "年初資產（樣本外首日）": start_value,
                "年底資產": end_value,
                "年度報酬": annual_return,
                "100%目標達成": bool(full_year and annual_return >= 1.0),
            })
    annual_performance = pd.DataFrame(annual_rows)
    return {
        "status": "ok",
        "selected_params": dict(best),
        "selection_score": float(best_score),
        "selection_method": "前70%歷史中的內部時間序列驗證；最後30%完全保留",
        "split_date": split_date,
        "train_rows": candidate_rows.sort_values("研究分數", ascending=False),
        "holdout": holdout,
        "holdout_stats": holdout_stats,
        "annual_performance": annual_performance,
        "annual_100pct_years": int(annual_performance["100%目標達成"].sum()) if not annual_performance.empty else 0,
        "complete_years": int(annual_performance["期間完整"].sum()) if not annual_performance.empty else 0,
        "search_count": len(candidate_rows),
        "symbols": len(clean),
    }
