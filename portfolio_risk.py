"""Cross-sectional risk snapshot and capped risk-parity allocation."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _return_panel(frames, min_history):
    columns = {}
    for symbol, frame in frames.items():
        if frame is None or frame.empty or not {"date", "close"}.issubset(frame.columns):
            continue
        z = frame[["date", "close"]].copy()
        z["date"] = pd.to_datetime(z["date"], errors="coerce")
        z["close"] = pd.to_numeric(z["close"], errors="coerce")
        z = z.dropna().sort_values("date").drop_duplicates("date", keep="last")
        z = z.loc[z["close"] > 0]
        columns[str(symbol)] = z.set_index("date")["close"].pct_change()
    if not columns:
        return pd.DataFrame()
    panel = pd.concat(columns, axis=1, join="inner").replace([np.inf, -np.inf], np.nan).dropna()
    if len(panel) < int(min_history):
        return pd.DataFrame()
    return panel.tail(756)


def _capped_weights(scores, total, cap):
    scores = np.maximum(np.asarray(scores, dtype=float), 1e-12)
    if len(scores) * cap < total - 1e-10:
        raise ValueError("單一標的權重上限過低，無法達到總曝險")
    lo, hi = 0.0, total / max(float(scores.min()), 1e-12) + total
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.minimum(cap, mid * scores).sum() > total:
            hi = mid
        else:
            lo = mid
    weights = np.minimum(cap, lo * scores)
    residual = total - weights.sum()
    for i in np.argsort(-scores):
        add = min(residual, cap - weights[i])
        weights[i] += add
        residual -= add
        if residual <= 1e-10:
            break
    return weights


def risk_parity_weights(returns, *, max_weight=0.30, target_exposure=0.90,
                        shrinkage=0.20, max_iterations=500, tolerance=1e-7):
    """Equalize approximate component risk subject to long-only exposure/name caps."""
    if returns is None or returns.empty or len(returns) < 30:
        raise ValueError("至少需要 30 個共同有效交易日")
    if not 0 < max_weight <= 1 or not 0 < target_exposure <= 1:
        raise ValueError("權重上限及總曝險必須介於 0 與 1")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage 必須介於 0 與 1")
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    vol = clean.std(ddof=1) * np.sqrt(252)
    keep = vol[vol > 1e-8].index
    clean, vol = clean[keep], vol[keep]
    n = len(keep)
    if n == 0:
        raise ValueError("所有標的波動率皆為零或無效")
    if n * max_weight < target_exposure - 1e-10:
        raise ValueError(f"目前 {n} 檔標的無法在單檔上限 {max_weight:.0%} 下達到總曝險 {target_exposure:.0%}")
    cov = clean.cov().to_numpy(dtype=float) * 252
    cov = (1 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))
    cov += np.eye(n) * 1e-10
    weights = _capped_weights(1 / vol.to_numpy(), target_exposure, max_weight)
    target_rc = 1.0 / n
    for _ in range(max_iterations):
        variance = float(weights @ cov @ weights)
        if variance <= 1e-16:
            break
        rc = weights * (cov @ weights) / variance
        if not np.all(np.isfinite(rc)) or np.any(rc <= 0):
            break
        if np.max(np.abs(rc - target_rc)) < tolerance:
            break
        weights = _capped_weights(weights * np.sqrt(target_rc / np.maximum(rc, 1e-10)),
                                  target_exposure, max_weight)
    variance = float(weights @ cov @ weights)
    rc = weights * (cov @ weights) / variance if variance > 1e-16 else np.zeros(n)
    w = pd.Series(weights, index=keep, name="配置權重").sort_values(ascending=False)
    risk = pd.Series(rc, index=keep, name="風險貢獻").reindex(w.index)
    corr = clean.corr().reindex(index=w.index, columns=w.index)
    return w, risk, corr


def dynamic_strategy_ensemble(strategy_equities, *, initial_capital=100000,
                              lookback=60, rebalance_frequency=20,
                              max_weight=0.40, target_exposure=0.90):
    """Combine strategy OOS equity curves using lagged rolling risk parity.

    Weights for date t use returns strictly before t. The first lookback period
    uses equal weights. This is a sequential ensemble, not a retroactive fit.
    """
    if initial_capital <= 0 or lookback < 30 or rebalance_frequency < 1:
        raise ValueError("本金、回看期或再平衡頻率設定不合法")
    series = {}
    for name, curve in strategy_equities.items():
        if curve is None or curve.empty or not {"日期", "資產"}.issubset(curve.columns):
            continue
        z = curve[["日期", "資產"]].copy()
        z["日期"] = pd.to_datetime(z["日期"], errors="coerce")
        z["資產"] = pd.to_numeric(z["資產"], errors="coerce")
        z = z.dropna().sort_values("日期").drop_duplicates("日期", keep="last")
        z = z.loc[z["資產"] > 0]
        if len(z) < lookback + 2:
            continue
        r = z.set_index("日期")["資產"].pct_change()
        r.iloc[0] = float(z["資產"].iloc[0]) / float(initial_capital) - 1.0
        series[str(name)] = r
    if len(series) < 2:
        return {"status": "insufficient_strategies", "equity": pd.DataFrame(),
                "weights": pd.DataFrame(), "returns": pd.Series(dtype=float)}
    panel = pd.concat(series, axis=1, join="inner").replace([np.inf, -np.inf], np.nan).dropna()
    if len(panel) < lookback + 1:
        return {"status": "insufficient_history", "equity": pd.DataFrame(),
                "weights": pd.DataFrame(), "returns": pd.Series(dtype=float)}
    n = panel.shape[1]
    cap = max(float(max_weight), float(target_exposure) / n)
    cap = min(cap, 1.0)
    weights = np.full(n, float(target_exposure) / n)
    weight_rows = []
    blended = []
    for i, (day, current_returns) in enumerate(panel.iterrows()):
        if i >= lookback and (i - lookback) % rebalance_frequency == 0:
            history = panel.iloc[i - lookback:i]
            try:
                w, _, _ = risk_parity_weights(
                    history, max_weight=cap, target_exposure=target_exposure)
                weights = w.reindex(panel.columns).fillna(0).to_numpy(dtype=float)
            except ValueError:
                # If the trailing covariance is degenerate, retain the last valid weights.
                pass
        blended.append(float(np.dot(weights, current_returns.to_numpy(dtype=float))))
        weight_rows.append({"日期": day, **dict(zip(panel.columns, weights))})
    daily = pd.Series(blended, index=panel.index, name="組合日報酬")
    equity = pd.DataFrame({
        "日期": panel.index,
        "資產": float(initial_capital) * (1.0 + daily).cumprod().to_numpy(),
        "日報酬": daily.to_numpy(),
    })
    return {
        "status": "ok", "equity": equity,
        "weights": pd.DataFrame(weight_rows).set_index("日期"),
        "returns": daily, "strategies": list(panel.columns),
        "method": f"使用前 {lookback} 個交易日估計風險；每 {rebalance_frequency} 日再平衡；當日權重只使用此前資料；曝險上限 {target_exposure:.0%}；不使用槓桿",
    }


def portfolio_risk_report(frames, *, min_history=60, max_weight=0.30,
                          target_exposure=0.90, shrinkage=0.20):
    """Current as-of snapshot only; never use this output to rewrite past OOS."""
    panel = _return_panel(frames, min_history)
    if panel.empty:
        return {"status": "insufficient_history", "observations": 0, "as_of": None,
                "weights": pd.Series(dtype=float), "risk_contributions": pd.Series(dtype=float),
                "volatility": pd.Series(dtype=float), "correlation": pd.DataFrame()}
    weights, risk, corr = risk_parity_weights(
        panel, max_weight=max_weight, target_exposure=target_exposure, shrinkage=shrinkage)
    cov = panel.cov().to_numpy(dtype=float) * 252
    w = weights.reindex(panel.columns).fillna(0).to_numpy()
    return {"status": "ok", "observations": len(panel),
            "as_of": pd.Timestamp(panel.index.max()).date().isoformat(),
            "weights": weights, "risk_contributions": risk,
            "volatility": panel.std(ddof=1).mul(np.sqrt(252)).reindex(weights.index),
            "correlation": corr,
            "estimated_annual_volatility": float(np.sqrt(max(0, w @ cov @ w))),
            "cash_weight": float(max(0, 1 - weights.sum())),
            "method": "共同交易日完整樣本；20% 對角收縮協方差；等風險貢獻迭代；單檔權重上限；不使用槓桿"}
