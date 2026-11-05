"""Supplemental portfolio and trade analytics for research reports."""
import numpy as np
import pandas as pd


def equity_risk_metrics(equity: pd.DataFrame, initial_capital: float) -> dict:
    """Calculate risk statistics from a marked-to-market equity curve."""
    if equity is None or equity.empty or "資產" not in equity:
        return {"年化波動率": 0.0, "Sortino": 0.0, "Calmar": 0.0, "最大回撤": 0.0, "平均曝險": 0.0}
    values = pd.to_numeric(equity["資產"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty or not np.isfinite(float(initial_capital)) or initial_capital <= 0:
        return {"年化波動率": 0.0, "Sortino": 0.0, "Calmar": 0.0, "平均曝險": 0.0}
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
        "最大回撤": float(max_dd),
        "平均曝險": float(exposure),
    }


def trade_statistics(trades: pd.DataFrame) -> dict:
    """Summarize closed trades only; open positions are not included."""
    empty = {
        "勝率": 0.0, "平均獲利": 0.0, "平均虧損": 0.0,
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
        "勝率": float((pnl > 0).mean()),
        "平均獲利": float(wins.mean()) if not wins.empty else 0.0,
        "平均虧損": float(losses.mean()) if not losses.empty else 0.0,
        "Profit Factor": pf,
        "期望值": float(pnl.mean()),
        "平均持有日數": float(holding.mean()) if not holding.empty else 0.0,
    }
