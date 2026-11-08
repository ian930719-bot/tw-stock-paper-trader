"""Continuous-portfolio walk-forward validation.

Unlike rolling_walk_forward(), this module does not liquidate and reset the
account at every OOS boundary. Adaptive parameters are selected using only
pre-window history, then their signals are spliced into one causal timeline.
"""
from __future__ import annotations

import ast
from typing import Mapping

import numpy as np
import pandas as pd

from analytics import equity_risk_metrics, trade_statistics
from research_engine import STRATEGIES, shared_capital_backtest
from strategy_evolution import _prepare_overrides
from walk_forward import rolling_walk_forward


def _make_windows(calendar, lookback_days, test_days, warmup_bars, min_test_days):
    first_start = max(int(lookback_days), int(warmup_bars))
    windows = []
    start = first_start
    while start < len(calendar):
        end = min(start + int(test_days), len(calendar))
        if end - start < int(min_test_days):
            break
        windows.append((pd.Timestamp(calendar[start]), pd.Timestamp(calendar[end - 1])))
        start += int(test_days)
    return windows


def _window_metrics(equity: pd.DataFrame, trades: pd.DataFrame,
                    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
                    initial_capital: float) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    eq = equity.copy()
    eq["日期"] = pd.to_datetime(eq["日期"])
    rows = []
    for number, (start, end) in enumerate(windows, start=1):
        part = eq.loc[eq["日期"].between(start, end)].copy()
        if part.empty:
            continue
        prior = eq.loc[eq["日期"] < start, "資產"]
        starting_equity = float(prior.iloc[-1]) if not prior.empty else float(initial_capital)
        ending_equity = float(part["資產"].iloc[-1])
        path = np.r_[starting_equity, pd.to_numeric(part["資產"], errors="coerce").to_numpy()]
        peaks = np.maximum.accumulate(path)
        drawdown = path[1:] / peaks[1:] - 1.0
        selected_trades = trades
        if not trades.empty and "出場日" in trades:
            exit_dates = pd.to_datetime(trades["出場日"], errors="coerce")
            selected_trades = trades.loc[exit_dates.between(start, end)]
        stats = trade_statistics(selected_trades)
        daily = pd.to_numeric(part["資產"], errors="coerce").pct_change()
        if len(daily):
            daily.iloc[0] = float(part["資產"].iloc[0]) / starting_equity - 1.0
        vol = float(daily.std(ddof=1) * np.sqrt(252)) if daily.notna().sum() > 1 else 0.0
        sharpe = (float(daily.mean() / daily.std(ddof=1) * np.sqrt(252))
                  if daily.notna().sum() > 1 and daily.std(ddof=1) > 0 else 0.0)
        rows.append({
            "區段": number, "測試開始": start.date().isoformat(),
            "測試結束": end.date().isoformat(), "交易日數": int(len(part)),
            "期初資產": starting_equity, "期末資產": ending_equity,
            "總報酬": ending_equity / starting_equity - 1.0 if starting_equity else 0.0,
            "最大回撤": float(np.nanmin(drawdown)) if len(drawdown) else 0.0,
            "年化波動率": vol, "Sharpe": sharpe, **stats,
        })
    return pd.DataFrame(rows)


def continuous_walk_forward(
    frames: Mapping[str, pd.DataFrame],
    strategy: str,
    *,
    initial_capital: float = 100_000,
    max_positions: int = 3,
    risk_fraction: float = 0.005,
    lookback_days: int = 756,
    test_days: int = 126,
    warmup_bars: int = 200,
    fee: float = 0.001425,
    sell_tax: float = 0.003,
    slippage: float = 0.001,
    exit_profile: str = "baseline_atr2",
    min_test_days: int = 40,
    corporate_actions=None,
) -> dict:
    """Run a single continuously funded portfolio over non-overlapping OOS windows.

    Fixed strategies keep their rule set throughout the OOS period. The adaptive
    strategy selects a candidate from each window's pre-test history only.
    Existing positions and cash carry across all OOS boundaries.
    """
    if strategy not in STRATEGIES and strategy != "AI 自適應策略":
        raise ValueError("未知策略")
    clean = {
        str(code): frame.copy().sort_values("date").reset_index(drop=True)
        for code, frame in frames.items()
        if frame is not None and not frame.empty and "date" in frame
    }
    if not clean:
        return {"windows": pd.DataFrame(), "equity": pd.DataFrame(),
                "trades": pd.DataFrame(), "metrics": {}, "coverage": 0}
    calendar = sorted(set().union(*[
        set(pd.to_datetime(frame["date"])) for frame in clean.values()
    ]))
    windows = _make_windows(calendar, lookback_days, test_days,
                            warmup_bars, min_test_days)
    if not windows:
        return {"windows": pd.DataFrame(), "equity": pd.DataFrame(),
                "trades": pd.DataFrame(), "metrics": {}, "coverage": len(clean),
                "note": "可用歷史不足以建立樣本外區段"}

    signal_overrides = None
    selection_rows = []
    if strategy == "AI 自適應策略":
        selection_run = rolling_walk_forward(
            clean, strategy, initial_capital=initial_capital,
            max_positions=max_positions, risk_fraction=risk_fraction,
            lookback_days=lookback_days, test_days=test_days,
            step_days=test_days, warmup_bars=warmup_bars,
            fee=fee, sell_tax=sell_tax, slippage=slippage,
            exit_profile=exit_profile, min_test_days=min_test_days,
            corporate_actions=corporate_actions,
        )
        selection_rows = selection_run.get("windows", []).to_dict("records")
        composite = {
            code: [pd.Series(False, index=frame.index),
                   pd.Series(False, index=frame.index)]
            for code, frame in clean.items()
        }
        params_cache = {}
        for row in selection_rows:
            start, end = pd.Timestamp(row["測試開始"]), pd.Timestamp(row["測試結束"])
            raw_params = row.get("選中策略參數")
            try:
                params = ast.literal_eval(raw_params) if raw_params else None
            except (ValueError, SyntaxError):
                params = None
            status = row.get("選擇狀態", "unknown")
            if status == "ok" and isinstance(params, dict):
                key = tuple(sorted(params.items()))
                if key not in params_cache:
                    params_cache[key] = _prepare_overrides(clean, params, corporate_actions)
                candidate = params_cache[key]
                for code, frame in clean.items():
                    entries, exits = candidate[code]
                    mask = pd.to_datetime(frame["date"]).between(start, end)
                    composite[code][0].loc[mask] = pd.Series(entries).reset_index(drop=True).loc[mask].to_numpy()
                    composite[code][1].loc[mask] = pd.Series(exits).reset_index(drop=True).loc[mask].to_numpy()
        signal_overrides = {
            code: (pair[0].reset_index(drop=True), pair[1].reset_index(drop=True))
            for code, pair in composite.items()
        }

    first_oos = windows[0][0]
    last_oos = windows[-1][1]
    # Exclude a trailing partial window: only complete, predeclared OOS windows
    # belong in the reported sample. Keep all pre-OOS bars for indicator warm-up.
    backtest_frames = {}
    for code, frame in clean.items():
        mask = pd.to_datetime(frame["date"]) <= last_oos
        backtest_frames[code] = frame.loc[mask].reset_index(drop=True)
    backtest_overrides = None
    if signal_overrides is not None:
        backtest_overrides = {}
        for code, frame in clean.items():
            mask = pd.to_datetime(frame["date"]) <= last_oos
            backtest_overrides[code] = tuple(
                pd.Series(values).reset_index(drop=True).loc[mask].reset_index(drop=True)
                for values in signal_overrides[code]
            )
    result = shared_capital_backtest(
        backtest_frames, strategy,
        initial_capital=initial_capital,
        max_positions=max_positions,
        risk_fraction=risk_fraction,
        fee=fee, sell_tax=sell_tax, slippage=slippage,
        warmup=0,
        report_start_date=first_oos,
        trading_start_date=first_oos,
        exit_profile=exit_profile,
        signal_overrides=backtest_overrides,
        corporate_actions=corporate_actions,
    )
    equity = result["equity"].copy()
    trades = result["trades"].copy()
    window_table = _window_metrics(equity, trades, windows, initial_capital)
    if strategy == "AI 自適應策略" and not window_table.empty and selection_rows:
        selected = pd.DataFrame(selection_rows)
        cols = ["測試開始", "測試結束", "選擇狀態", "選中策略參數", "選擇分數"]
        selected = selected[[c for c in cols if c in selected.columns]]
        window_table = window_table.merge(selected, on=["測試開始", "測試結束"], how="left")

    metrics = dict(result["metrics"])
    metrics.update(equity_risk_metrics(equity, initial_capital))
    metrics.update(trade_statistics(trades))
    return {
        "windows": window_table, "equity": equity, "trades": trades,
        "metrics": metrics, "coverage": result["coverage"]["symbols_used"],
        "corporate_actions_applied": result.get("corporate_actions_applied", 0),
        "note": ("單一連續資金帳戶；現金、持倉與風險狀態跨樣本外區段延續。"
                 "自適應候選只由每個測試區段之前的歷史選出；未通過內部驗證的區段保持現金。"),
    }
