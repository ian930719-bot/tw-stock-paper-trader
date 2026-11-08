"""Rolling, non-overlapping out-of-sample evaluation for shared-capital strategies.

Each test window starts flat with the same nominal capital. Prior bars are
retained only to warm indicators; trading is disabled before the OOS boundary.
Window returns are chained for an aggregate OOS curve, not treated as one
continuously held account.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics import equity_risk_metrics, trade_statistics
from research_engine import STRATEGIES, shared_capital_backtest


def rolling_walk_forward(
    frames: dict[str, pd.DataFrame],
    strategy: str,
    *,
    initial_capital: float = 1_000_000,
    max_positions: int = 5,
    risk_fraction: float = 0.005,
    lookback_days: int = 756,
    test_days: int = 126,
    step_days: int | None = None,
    warmup_bars: int = 200,
    fee: float = 0.001425,
    sell_tax: float = 0.003,
    slippage: float = 0.001,
    exit_profile: str = "baseline_atr2",
    min_test_days: int = 40,
) -> dict:
    """Evaluate fixed rules across rolling, non-overlapping OOS windows.

    No parameter optimization is performed. Each window uses only historical
    bars before its test start for indicator warm-up and permits orders only
    from the test start onward. Capital is reset per window; aggregate returns
    chain the independent window outcomes.
    """
    if strategy not in STRATEGIES:
        raise ValueError("未知策略")
    if initial_capital <= 0 or not np.isfinite(initial_capital):
        raise ValueError("initial_capital 必須是有限正數")
    if max_positions < 1 or not 0 <= risk_fraction <= 0.10:
        raise ValueError("持倉上限或風險預算不合理")
    if lookback_days < warmup_bars or test_days < min_test_days or min_test_days < 2:
        raise ValueError("回看、測試或最小測試區間長度不合理")
    step = test_days if step_days is None else int(step_days)
    if step < test_days:
        raise ValueError("為避免樣本外區間重疊，step_days 不可小於 test_days")

    valid = {
        str(code): frame.copy().sort_values("date").reset_index(drop=True)
        for code, frame in frames.items()
        if frame is not None and not frame.empty and "date" in frame
    }
    if not valid:
        return {"windows": pd.DataFrame(), "equity": pd.DataFrame(),
                "trades": pd.DataFrame(), "metrics": {}, "coverage": 0}

    calendar = sorted(set().union(*[
        set(pd.to_datetime(frame["date"])) for frame in valid.values()
    ]))
    first_start = max(int(lookback_days), int(warmup_bars))
    window_rows, curves, trade_frames = [], [], []
    chained_equity = float(initial_capital)

    start_idx = first_start
    while start_idx < len(calendar):
        end_idx = min(start_idx + int(test_days), len(calendar))
        if end_idx - start_idx < min_test_days:
            break
        start_date = pd.Timestamp(calendar[start_idx])
        end_date = pd.Timestamp(calendar[end_idx - 1])
        window_frames = {}
        for code, frame in valid.items():
            dates = pd.to_datetime(frame["date"])
            history = frame.loc[dates < start_date].tail(int(lookback_days))
            active = frame.loc[(dates >= start_date) & (dates <= end_date)]
            if len(active) < 2:
                continue
            window_frames[code] = pd.concat([history, active], ignore_index=True)

        result = shared_capital_backtest(
            window_frames, strategy, initial_capital=float(initial_capital),
            max_positions=int(max_positions), risk_fraction=float(risk_fraction),
            fee=fee, sell_tax=sell_tax, slippage=slippage, warmup=0,
            report_start_date=start_date, trading_start_date=start_date,
            exit_profile=exit_profile,
        )
        eq = result["equity"].copy()
        trades = result["trades"].copy()
        metrics = result["metrics"].copy()
        metrics.update(equity_risk_metrics(eq, float(initial_capital)))
        metrics.update(trade_statistics(trades))
        window_rows.append({
            "區段": len(window_rows) + 1,
            "測試開始": start_date.date().isoformat(),
            "測試結束": end_date.date().isoformat(),
            "交易日數": int(len(eq)),
            "標的數": int(result["coverage"]["symbols_used"]),
            **metrics,
            "交易筆數": int(len(trades)),
        })
        if not eq.empty:
            scale = chained_equity / float(initial_capital)
            segment = eq[["日期", "資產"]].copy()
            segment["區段"] = len(window_rows)
            segment["區段資產"] = segment["資產"]
            segment["資產"] = segment["資產"] * scale
            curves.append(segment)
            chained_equity = float(segment["資產"].iloc[-1])
        if not trades.empty:
            trades.insert(0, "樣本外區段", len(window_rows))
            trade_frames.append(trades)
        start_idx += step

    windows = pd.DataFrame(window_rows)
    equity = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=["日期", "資產"])
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    if len(equity) > 1:
        returns = equity["資產"].pct_change().fillna(
            equity["資產"].iloc[0] / float(initial_capital) - 1.0
        )
        metrics = {
            "期末資產（鏈結）": float(equity["資產"].iloc[-1]),
            "總報酬（鏈結）": float(equity["資產"].iloc[-1] / float(initial_capital) - 1.0),
            "最大回撤（鏈結）": float((equity["資產"] / np.maximum.accumulate(
                np.r_[float(initial_capital), equity["資產"].to_numpy()]
            )[1:] - 1.0).min()),
            "Sharpe（鏈結）": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
                if returns.std(ddof=1) > 0 else 0.0,
        }
    else:
        metrics = {}
    return {
        "windows": windows, "equity": equity, "trades": trades,
        "metrics": metrics, "coverage": len(valid),
        "note": "各測試區段獨立重設本金並清空持倉；鏈結曲線用於串接區段報酬，不代表同一帳戶連續持倉。",
    }


def classify_market_regime(
    benchmark: pd.DataFrame,
    *,
    ema_window: int = 200,
    slope_window: int = 20,
) -> pd.Series:
    """Classify each benchmark session using only that session's close.

    Bull: close and EMA slope are positive versus the long-term trend.
    Bear: close and EMA slope are negative versus the long-term trend.
    Otherwise sideways/transition. This is a descriptive regime proxy.
    """
    if benchmark is None or benchmark.empty or not {"date", "close"}.issubset(benchmark.columns):
        return pd.Series(dtype="object", name="盤勢")
    if ema_window < 2 or slope_window < 1:
        raise ValueError("ema_window 與 slope_window 必須為正整數")
    z = benchmark[["date", "close"]].copy()
    z["date"] = pd.to_datetime(z["date"])
    z["close"] = pd.to_numeric(z["close"], errors="coerce")
    z = z.dropna().sort_values("date").drop_duplicates("date", keep="last")
    ema = z["close"].ewm(span=int(ema_window), adjust=False, min_periods=int(ema_window)).mean()
    rising = ema > ema.shift(int(slope_window))
    falling = ema < ema.shift(int(slope_window))
    labels = np.select(
        [(z["close"] > ema) & rising, (z["close"] < ema) & falling],
        ["多頭", "空頭"], default="盤整／轉換"
    )
    return pd.Series(labels, index=pd.DatetimeIndex(z["date"]), name="盤勢")


def regime_performance(
    equity: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    initial_capital: float = 1_000_000,
    ema_window: int = 200,
    slope_window: int = 20,
) -> pd.DataFrame:
    """Summarize OOS daily returns conditional on a benchmark regime.

    If equity contains 區段資產 and 區段, daily returns are computed inside
    each independently reset test window, avoiding artificial returns across
    gaps between windows. The first session of each window is measured from
    its nominal starting capital.
    """
    columns = ["盤勢", "交易日數", "累積報酬", "年化波動率", "Sharpe", "上漲日比例"]
    if equity is None or equity.empty or benchmark is None or benchmark.empty:
        return pd.DataFrame(columns=columns)
    regime = classify_market_regime(
        benchmark, ema_window=ema_window, slope_window=slope_window
    )
    if regime.empty:
        return pd.DataFrame(columns=columns)
    z = equity.copy()
    z["日期"] = pd.to_datetime(z["日期"])
    if {"區段資產", "區段"}.issubset(z.columns):
        z["報酬"] = z.groupby("區段", sort=False)["區段資產"].pct_change()
        first = z.groupby("區段", sort=False).head(1).index
        z.loc[first, "報酬"] = (
            pd.to_numeric(z.loc[first, "區段資產"], errors="coerce") /
            float(initial_capital) - 1.0
        )
    else:
        z["報酬"] = pd.to_numeric(z["資產"], errors="coerce").pct_change()
        if len(z):
            z.loc[z.index[0], "報酬"] = float(z["資產"].iloc[0]) / float(initial_capital) - 1.0
    z["盤勢"] = z["日期"].map(regime)
    z["報酬"] = pd.to_numeric(z["報酬"], errors="coerce")
    z = z.dropna(subset=["盤勢", "報酬"])
    rows = []
    for label in ["多頭", "空頭", "盤整／轉換"]:
        values = z.loc[z["盤勢"] == label, "報酬"]
        if values.empty:
            continue
        vol = float(values.std(ddof=1) * np.sqrt(252)) if len(values) > 1 else 0.0
        sharpe = float(values.mean() / values.std(ddof=1) * np.sqrt(252)) if len(values) > 1 and values.std(ddof=1) > 0 else 0.0
        rows.append({
            "盤勢": label,
            "交易日數": int(len(values)),
            "累積報酬": float((1.0 + values).prod() - 1.0),
            "年化波動率": vol,
            "Sharpe": sharpe,
            "上漲日比例": float((values > 0).mean()),
        })
    return pd.DataFrame(rows, columns=columns)


def parameter_sensitivity(
    frames: dict[str, pd.DataFrame],
    strategy: str,
    *,
    risk_fractions=(0.0025, 0.005, 0.01),
    exit_profiles=("baseline_atr2", "atr_1_5", "atr_3"),
    **walk_forward_kwargs,
) -> pd.DataFrame:
    """Run a small, predeclared robustness grid; does not select a winner."""
    risks = tuple(float(x) for x in risk_fractions)
    profiles = tuple(exit_profiles)
    if not risks or not profiles:
        raise ValueError("風險比例與出場設定不可為空")
    if any(not np.isfinite(x) or x < 0 or x > 0.10 for x in risks):
        raise ValueError("風險比例必須介於 0 至 10%")
    allowed = {"baseline_atr2", "oneil_8pct", "minervini_structure", "atr_1_5", "atr_2", "atr_2_5", "atr_3"}
    if any(x not in allowed for x in profiles):
        raise ValueError("包含未知出場設定")
    rows = []
    for profile in profiles:
        for risk in risks:
            result = rolling_walk_forward(
                frames, strategy, risk_fraction=risk,
                exit_profile=profile, **walk_forward_kwargs
            )
            metrics = result.get("metrics", {})
            rows.append({
                "出場規則": profile,
                "單筆風險預算": risk,
                "有效區段": int(len(result.get("windows", []))),
                "交易筆數": int(len(result.get("trades", []))),
                "鏈結總報酬": metrics.get("總報酬（鏈結）", np.nan),
                "鏈結最大回撤": metrics.get("最大回撤（鏈結）", np.nan),
                "鏈結 Sharpe": metrics.get("Sharpe（鏈結）", np.nan),
            })
    return pd.DataFrame(rows)
