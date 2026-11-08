"""General multi-factor signal research; no investment-style assumptions.

Signals combine trend, Bollinger position, RSI, MACD, volume confirmation,
volatility and optional benchmark-relative strength. Regime labels modify
signal interpretation only; they never hard-disable entries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def factor_frame(frame: pd.DataFrame, benchmark: pd.Series | None = None) -> pd.DataFrame:
    required = {"date", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"缺少必要欄位：{sorted(missing)}")
    z = frame.sort_values("date").reset_index(drop=True).copy()
    close = pd.to_numeric(z["close"], errors="coerce")
    high = pd.to_numeric(z["high"], errors="coerce")
    low = pd.to_numeric(z["low"], errors="coerce")
    volume = pd.to_numeric(z["volume"], errors="coerce")

    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    std20 = close.rolling(20, min_periods=20).std(ddof=0)
    z["bb_upper"] = sma20 + 2.0 * std20
    z["bb_lower"] = sma20 - 2.0 * std20
    z["bb_percent_b"] = (close - z["bb_lower"]) / (z["bb_upper"] - z["bb_lower"]).replace(0, np.nan)
    z["bb_bandwidth"] = (z["bb_upper"] - z["bb_lower"]) / sma20.replace(0, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    z["rsi14"] = (100 - 100 / (1 + rs)).fillna(50)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    z["macd"] = ema12 - ema26
    z["macd_signal"] = z["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    z["macd_hist"] = z["macd"] - z["macd_signal"]

    true_range = pd.concat([
        high - low, (high - close.shift()).abs(), (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    z["atr14_pct"] = true_range.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / close.replace(0, np.nan)
    z["volume_ratio20"] = volume / volume.shift(1).rolling(20, min_periods=20).mean()
    z["momentum20"] = close.pct_change(20)
    z["momentum60"] = close.pct_change(60)
    z["trend_strength"] = (close / sma50 - 1.0)
    z["long_trend"] = close > sma200

    if benchmark is not None:
        b = pd.Series(benchmark).copy()
        b.index = pd.to_datetime(b.index)
        dates = pd.to_datetime(z["date"])
        benchmark_returns = b.sort_index().pct_change(20)
        mapped = pd.Series(dates.map(benchmark_returns), index=z.index, dtype=float)
        z["relative_strength20"] = z["momentum20"] - mapped.to_numpy()
    else:
        z["relative_strength20"] = np.nan

    # A regime is descriptive, not a buy/no-buy gate.
    z["market_regime"] = np.select(
        [close > sma200, (close <= sma200) & (close > sma50)],
        ["趨勢偏強", "弱勢反彈／轉換"],
        default="趨勢偏弱／盤整",
    )
    return z


def multifactor_signals(
    frame: pd.DataFrame,
    *,
    min_entry_score: int = 4,
    max_atr_pct: float = 0.12,
    benchmark: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Return entry, exit and auditable factor columns using current/past bars."""
    if not 1 <= int(min_entry_score) <= 7:
        raise ValueError("min_entry_score 必須介於 1 至 7")
    if not np.isfinite(float(max_atr_pct)) or not 0 < float(max_atr_pct) <= 1:
        raise ValueError("max_atr_pct 必須介於 0 與 1")
    z = factor_frame(frame, benchmark)
    close = pd.to_numeric(z["close"], errors="coerce")
    checks = pd.DataFrame({
        "趨勢": close > close.rolling(50, min_periods=50).mean(),
        "布林通道": (z["bb_percent_b"] > 0.65) & (z["bb_percent_b"] < 1.15),
        "RSI動能": z["rsi14"].between(50, 75),
        "MACD動能": z["macd_hist"] > 0,
        "成交量確認": z["volume_ratio20"] >= 1.2,
        "中期動能": z["momentum60"] > 0,
        "相對強度": z["relative_strength20"] > 0,
    }).fillna(False)
    # If no benchmark was supplied, relative strength is neutral, not a failure.
    if benchmark is None:
        checks["相對強度"] = True
    score = checks.sum(axis=1)
    entry = ((score >= int(min_entry_score)) & (z["atr14_pct"] <= float(max_atr_pct))).fillna(False)
    exit_signal = (
        (z["macd_hist"] < 0)
        | (z["rsi14"] < 42)
        | (z["bb_percent_b"] < 0.15)
        | (z["atr14_pct"] > float(max_atr_pct) * 1.5)
    ).fillna(False)
    audit = z[["date", "close", "bb_percent_b", "bb_bandwidth", "rsi14",
               "macd", "macd_signal", "macd_hist", "volume_ratio20",
               "momentum20", "momentum60", "relative_strength20",
               "atr14_pct", "market_regime"]].copy()
    audit["因子得分"] = score
    return entry, exit_signal, audit
