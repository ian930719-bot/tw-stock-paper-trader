"""Supplemental portfolio and trade analytics for research reports."""
import numpy as np
import pandas as pd


def equity_risk_metrics(equity: pd.DataFrame, initial_capital: float) -> dict:
    """Calculate risk statistics from a marked-to-market equity curve."""
    if equity is None or equity.empty or "資產" not in equity:
        return {"年化波動率": 0.0, "Sortino": 0.0, "Calmar": 0.0, "CAGR": 0.0, "最大回撤": 0.0, "平均曝險": 0.0}
    values = pd.to_numeric(equity["資產"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty or not np.isfinite(float(initial_capital)) or initial_capital <= 0:
        return {"年化波動率": 0.0, "Sortino": 0.0, "Calmar": 0.0, "CAGR": 0.0, "平均曝險": 0.0}
    returns = values.pct_change()
    returns.iloc[0] = values.iloc[0] / float(initial_capital) - 1.0
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    daily_vol = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    annual_vol = daily_vol * np.sqrt(252.0)
    downside = returns.clip(upper=0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside.to_numpy())))) if len(downside) else 0.0
    sortino = float(returns.mean() / downside_dev * np.sqrt(252.0)) if downside_dev > 0 else 0.0
    peaks = pd.concat([pd.Series([float(initial_capital)]), values.reset_index(drop=True)]).cummax().iloc[1:].reset_index(drop=True)
    drawdowns = values.reset_index(drop=True) / peaks - 1.0
    max_dd = float(drawdowns.min()) if len(drawdowns) else 0.0
    if "日期" in equity.columns:
        dates = pd.to_datetime(equity["日期"], errors="coerce").dropna()
    elif isinstance(equity.index, pd.DatetimeIndex):
        dates = pd.to_datetime(equity.index, errors="coerce").dropna()
    else:
        dates = pd.Series(dtype="datetime64[ns]")
    if len(dates) >= 2:
        first_date, last_date = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
        if first_date.tzinfo is not None:
            first_date = first_date.tz_localize(None)
        if last_date.tzinfo is not None:
            last_date = last_date.tz_localize(None)
        elapsed_years = (last_date - first_date).total_seconds() / (365.25 * 86400)
        years = max(elapsed_years, 1.0 / 252.0)
    else:
        # Backward-compatible fallback for undated synthetic/unit-test curves.
        years = max(len(values) / 252.0, 1.0 / 252.0)
    ending = float(values.iloc[-1])
    cagr = (ending / float(initial_capital)) ** (1.0 / years) - 1.0 if ending > 0 else -1.0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
    exposure = 0.0
    if {"持倉市值", "資產"}.issubset(equity.columns):
        denominator = pd.to_numeric(equity["資產"], errors="coerce").replace(0, np.nan)
        exposure_series = (pd.to_numeric(equity["持倉市值"], errors="coerce") / denominator).replace([np.inf, -np.inf], np.nan).dropna()
        exposure = float(exposure_series.mean()) if not exposure_series.empty else 0.0
    return {
        "年化波動率": float(annual_vol),
        "Sortino": float(sortino),
        "Calmar": float(calmar),
        "CAGR": float(cagr),
        "最大回撤": float(max_dd),
        "平均曝險": float(exposure),
    }



def strategy_decay_report(
    equity_curves: dict[str, pd.DataFrame],
    initial_capital: float,
    *,
    recent_window: int = 63,
    min_observations: int = 40,
) -> pd.DataFrame:
    """Flag possible strategy deterioration from realized OOS curves.

    This is a diagnostic only: it never disables a strategy or changes past
    weights. The recent window is compared with the full supplied OOS sample.
    """
    columns = ["策略", "樣本日數", "近期Sharpe", "全期Sharpe", "近期最大回撤", "全期最大回撤", "資料截至", "狀態"]
    rows = []
    if initial_capital <= 0 or recent_window < 20 or min_observations < 20:
        raise ValueError("本金、近期視窗及最低樣本數設定不合理")
    for name, curve in (equity_curves or {}).items():
        if curve is None or curve.empty or not {"日期", "資產"}.issubset(curve.columns):
            continue
        z = curve[["日期", "資產"]].copy()
        z["日期"] = pd.to_datetime(z["日期"], errors="coerce")
        z["資產"] = pd.to_numeric(z["資產"], errors="coerce")
        z = z.dropna().sort_values("日期").drop_duplicates("日期", keep="last")
        z = z.loc[z["資產"] > 0]
        if z.empty:
            continue
        returns = z["資產"].pct_change()
        returns.iloc[0] = float(z["資產"].iloc[0]) / float(initial_capital) - 1.0
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        recent = returns.tail(int(recent_window))

        def sharpe(values):
            vol = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            return float(values.mean() / vol * np.sqrt(252)) if vol > 0 else 0.0

        def drawdown(values, start_value):
            path = float(start_value) * (1.0 + values).cumprod()
            peaks = np.maximum.accumulate(np.r_[float(start_value), path.to_numpy()])[1:]
            return float((path.to_numpy() / peaks - 1.0).min()) if len(path) else 0.0

        full_sharpe = sharpe(returns)
        recent_sharpe = sharpe(recent)
        full_dd = drawdown(returns, initial_capital)
        start_value = float(z["資產"].iloc[max(0, len(z) - len(recent) - 1)])
        recent_dd = drawdown(recent, start_value)
        if len(returns) < int(min_observations):
            status = "樣本不足"
        elif recent_sharpe < 0 and full_sharpe > 0:
            status = "退化警示"
        elif recent_sharpe < full_sharpe - 1.0 or recent_dd < full_dd * 1.5:
            status = "表現轉弱"
        else:
            status = "未見明顯退化"
        rows.append({
            "策略": str(name), "樣本日數": int(len(returns)),
            "近期Sharpe": recent_sharpe, "全期Sharpe": full_sharpe,
            "近期最大回撤": recent_dd, "全期最大回撤": full_dd,
            "資料截至": pd.Timestamp(z["日期"].iloc[-1]).date().isoformat(),
            "狀態": status,
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["狀態", "近期Sharpe"], ascending=[True, True]
    ).reset_index(drop=True) if rows else pd.DataFrame(columns=columns)

def trade_statistics(trades: pd.DataFrame) -> dict:
    """Summarize closed trades only; open positions are not included."""
    empty = {
        "交易筆數": 0, "勝率": 0.0, "平均獲利": 0.0, "平均虧損": 0.0,
        "Profit Factor": 0.0, "期望值": 0.0, "平均持有日數": 0.0,
    }
    if trades is None or trades.empty or "損益" not in trades:
        return empty
    pnl = pd.to_numeric(trades["損益"], errors="coerce").dropna()
    if pnl.empty:
        return empty
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_profit, gross_loss = float(wins.sum()), float(abs(losses.sum()))
    pf = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)
    if "持有交易日" in trades:
        holding = pd.to_numeric(trades["持有交易日"], errors="coerce").dropna()
    elif {"進場日", "出場日"}.issubset(trades.columns):
        holding = (pd.to_datetime(trades["出場日"], errors="coerce") - pd.to_datetime(trades["進場日"], errors="coerce")).dt.days.dropna()
    else:
        holding = pd.Series(dtype=float)
    return {
        "交易筆數": int(len(pnl)),
        "勝率": float((pnl > 0).mean()),
        "平均獲利": float(wins.mean()) if not wins.empty else 0.0,
        "平均虧損": float(losses.mean()) if not losses.empty else 0.0,
        "Profit Factor": pf,
        "期望值": float(pnl.mean()),
        "平均持有日數": float(holding.mean()) if not holding.empty else 0.0,
    }
