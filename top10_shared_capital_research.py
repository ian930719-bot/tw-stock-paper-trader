"""Reproducible walk-forward research for the ten liquid Taiwan stocks.

Signals are evaluated at a completed close and entered no earlier than the next
available open. The simulator uses one shared cash balance, integer shares,
ATR-based risk sizing, an ADV20 participation cap, gap-aware stops, trailing
stops based only on completed bars, and conservative Taiwan transaction costs.
All inputs must be adjusted OHLCV (FinMind TaiwanStockPriceAdj or Yahoo
Finance auto_adjust=True). Dividends are already reflected in adjusted prices
and must not be added a second time.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ["2330", "2454", "1303", "2327", "2408", "2308", "3037", "3008", "2383", "2303"]
NAMES = {"2330":"台積電", "2454":"聯發科", "1303":"南亞", "2327":"國巨", "2408":"南亞科", "2308":"台達電", "3037":"欣興", "3008":"大立光", "2383":"台光電", "2303":"聯電"}
PERIODS = {
    "train": ("2010-01-04", "2020-12-31"),
    "validation": ("2021-01-01", "2023-12-31"),
    "test": ("2024-01-01", None),
}
BUY_FEE = 0.001425
SELL_FEE_TAX = 0.001425 + 0.003
SLIPPAGE = 0.001
INITIAL_CAPITAL = 1_000_000.0
RISK_FRACTION = 0.005
MAX_POSITIONS = 5
MAX_POSITION_WEIGHT = 0.20
MAX_ADV_PARTICIPATION = 0.01
INITIAL_STOP_ATR = 3.0
TRAIL_ATR = 3.5
STRATEGIES = ["KD_low", "Oversold3", "Trend_RSI", "TrendBreakout", "SqueezeBreakout", "TrendPullback"]
EXIT_MODES = ["fixed_20", "fixed_40", "fixed_60"]


def prepare_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Build close-known, causal indicators and predeclared candidate signals."""
    d = frame.sort_values("date").copy().reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    c, h, l, v = d.close, d.high, d.low, d.volume
    d["sma20"] = c.rolling(20, min_periods=20).mean()
    d["sma50"] = c.rolling(50, min_periods=50).mean()
    d["sma100"] = c.rolling(100, min_periods=100).mean()
    d["sma200"] = c.rolling(200, min_periods=200).mean()
    d["ema8"] = c.ewm(span=8, adjust=False).mean()
    d["ema21"] = c.ewm(span=21, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ll, hh = l.rolling(14, min_periods=14).min(), h.rolling(14, min_periods=14).max()
    rsv = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    d["k"] = rsv.ewm(com=2, adjust=False).mean()
    d["d"] = d.k.ewm(com=2, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    d["bb_mid"] = d.sma20
    d["bb_low"] = d.sma20 - 2 * sd
    d["bb_up"] = d.sma20 + 2 * sd
    d["bb_pct"] = (c - d.bb_low) / (d.bb_up - d.bb_low).replace(0, np.nan)
    d["bb_width"] = (d.bb_up - d.bb_low) / d.bb_mid.replace(0, np.nan)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    mfv = (((c-l) - (h-c)) / (h-l).replace(0, np.nan)) * v
    d["cmf"] = mfv.rolling(20, min_periods=20).sum() / v.rolling(20, min_periods=20).sum().replace(0, np.nan)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    d["obv_slope"] = obv.diff(20) / v.rolling(20, min_periods=20).sum().replace(0, np.nan)
    d["xkd"] = (d.k > d.d) & (d.k.shift() <= d.d.shift())
    d["xema8"] = (c > d.ema8) & (c.shift() <= d.ema8.shift())
    d["xmacd"] = (d.macd_hist > 0) & (d.macd_hist.shift() <= 0)
    d["xrsi40"] = (d.rsi > 40) & (d.rsi.shift() <= 40)
    d["xcmf"] = (d.cmf > 0) & (d.cmf.shift() <= 0)
    d["xobv"] = (d.obv_slope > 0) & (d.obv_slope.shift() <= 0)
    oversold_score = (
        (d.rsi < 35).astype(int) + (d.bb_pct < .1).astype(int)
        + (d.k < 20).astype(int) + (d.cmf < 0).astype(int)
        + (d.obv_slope < 0).astype(int) + (c < d.bb_low).astype(int)
    )
    confirmation = d.xkd | d.xmacd | d.xema8 | d.xcmf | d.xobv
    d["momentum20"] = c.pct_change(20)
    d["momentum60"] = c.pct_change(60)
    d["momentum120"] = c.pct_change(120)
    d["adv20"] = (c * v).shift(1).rolling(20, min_periods=20).mean()
    prior20_high = h.shift(1).rolling(20, min_periods=20).max()
    volume20 = v.shift(1).rolling(20, min_periods=20).mean()
    trend_stack = (c > d.sma50) & (d.sma50 > d.sma100) & (d.sma100 > d.sma200)
    squeeze_cut = d.bb_width.shift(1).rolling(120, min_periods=60).quantile(.20)
    d["sig_KD_low"] = (d.xkd & (d.k < 25) & (d.rsi < 45)).fillna(False)
    d["sig_Oversold3"] = ((oversold_score >= 3) & confirmation).fillna(False)
    d["sig_Trend_RSI"] = ((c > d.sma200) & (c > d.sma50) & d.xrsi40 & (d.bb_pct < .6)).fillna(False)
    d["sig_TrendBreakout"] = ((c > prior20_high) & (v >= 1.2 * volume20) & trend_stack & (d.momentum20 > 0)).fillna(False)
    d["sig_SqueezeBreakout"] = ((c > prior20_high) & (v >= 1.2 * volume20) & (d.bb_width.shift(1) <= squeeze_cut) & (c > d.sma50)).fillna(False)
    d["sig_TrendPullback"] = ((c > d.sma200) & (c > d.sma50) & d.xrsi40 & (d.rsi < 60) & (c > d.ema8)).fillna(False)
    return d


def load_adjusted_frames(input_dir: str | Path) -> tuple[dict[str, pd.DataFrame], dict]:
    root = Path(input_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("缺少 manifest.json，不能確認歷史資料來源與還原股價狀態")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("collection_complete") or int(manifest.get("symbols_collected", 0)) != 10:
        raise ValueError("資料收集未完整完成，預期十檔股票全部成功")
    frames = {}
    for symbol in SYMBOLS:
        path = root / "bars" / f"{symbol}.csv"
        if not path.exists():
            raise ValueError(f"缺少股票資料：{symbol}")
        d = pd.read_csv(path)
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(d.columns):
            raise ValueError(f"{symbol} 缺少 OHLCV 欄位")
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        d = d.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        valid = (
            (d[["open", "high", "low", "close"]] > 0).all(axis=1)
            & (d["volume"] >= 0)
            & (d["high"] >= d[["open", "close", "low"]].max(axis=1))
            & (d["low"] <= d[["open", "close", "high"]].min(axis=1))
        )
        d = d.loc[valid].copy()
        if len(d) < 1000:
            raise ValueError(f"{symbol} 有效歷史資料不足 1,000 根")
        d["symbol"] = symbol
        d["name"] = NAMES[symbol]
        frames[symbol] = prepare_bars(d)
    return frames, manifest


def _trade_cost_return(entry: float, exit_: float) -> float:
    return (exit_ / entry) * (1 - BUY_FEE) * (1 - SELL_FEE_TAX) - 1


def simulate_portfolio(
    frames: dict[str, pd.DataFrame], strategy: str, exit_mode: str,
    start: str, end: str, *, initial_capital: float = INITIAL_CAPITAL,
    risk_fraction: float = RISK_FRACTION, max_positions: int = MAX_POSITIONS,
    max_weight: float = MAX_POSITION_WEIGHT,
    max_participation: float = MAX_ADV_PARTICIPATION,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Run a single-cash, non-overlapping portfolio with next-open execution."""
    if strategy not in STRATEGIES or exit_mode not in EXIT_MODES:
        raise ValueError("未知策略或出場設定")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    row_lookup = {
        s: {pd.Timestamp(r.date): r for r in d.itertuples(index=False)}
        for s, d in frames.items()
    }
    all_dates = sorted(set().union(*[set(x) for x in row_lookup.values()]))
    dates = [d for d in all_dates if start_ts <= d <= end_ts]
    last_dates = {s: max((d for d in lookup if start_ts <= d <= end_ts), default=None) for s, lookup in row_lookup.items()}
    schedules: dict[pd.Timestamp, list[dict]] = {}
    for symbol, d in frames.items():
        signal_col = f"sig_{strategy}"
        for i in np.flatnonzero(d[signal_col].fillna(False).to_numpy()):
            if i + 1 >= len(d):
                continue
            signal_date = pd.Timestamp(d.date.iloc[i])
            entry_date = pd.Timestamp(d.date.iloc[i + 1])
            if not (start_ts <= signal_date < entry_date <= end_ts):
                continue
            signal = d.iloc[i]
            if pd.isna(signal.atr) or signal.atr <= 0:
                continue
            schedules.setdefault(entry_date, []).append({
                "symbol": symbol, "signal_date": signal_date,
                "atr": float(signal.atr),
                "momentum": float(signal.momentum20) if pd.notna(signal.momentum20) else -999.0,
                "adv20": float(signal.adv20) if pd.notna(signal.adv20) else 0.0,
            })

    cash = float(initial_capital)
    positions: dict[str, dict] = {}
    last_close: dict[str, float] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []

    for day in dates:
        today = {s: lookup[day] for s, lookup in row_lookup.items() if day in lookup}

        # Exit orders from the prior close and overnight stop gaps are filled first.
        for symbol in list(positions):
            if symbol not in today:
                continue
            p, row = positions[symbol], today[symbol]
            open_px = float(row.open)
            if p["pending_exit"]:
                fill, reason = open_px, p["pending_reason"]
            elif open_px <= p["stop"]:
                fill, reason = open_px, "gap_stop"
            else:
                fill = None
                reason = ""
            if fill is not None:
                fill = fill * (1 - SLIPPAGE)
                proceeds = p["shares"] * fill * (1 - SELL_FEE_TAX)
                pnl = proceeds - p["entry_cost"]
                trades.append({**p, "symbol": symbol, "exit_date": day,
                               "exit_price": fill, "exit_reason": reason,
                               "net_return": _trade_cost_return(p["entry_price"], fill),
                               "pnl": pnl})
                cash += proceeds
                del positions[symbol]

        # Rank same-day candidates by momentum; position and ADV limits are applied.
        candidates = sorted(schedules.get(day, []), key=lambda x: x["momentum"], reverse=True)
        for candidate in candidates:
            symbol = candidate["symbol"]
            if symbol in positions or symbol not in today or len(positions) >= max_positions:
                continue
            row = today[symbol]
            entry = float(row.open) * (1 + SLIPPAGE)
            atr = max(candidate["atr"], entry * .005)
            stop = entry - INITIAL_STOP_ATR * atr
            if stop <= 0 or stop >= entry:
                continue
            equity_now = cash + sum(p["shares"] * last_close.get(s, p["entry_price"]) for s, p in positions.items())
            risk_budget = equity_now * risk_fraction
            shares_by_risk = math.floor(risk_budget / max(entry - stop, .01))
            shares_by_weight = math.floor(equity_now * max_weight / (entry * (1 + BUY_FEE)))
            shares_by_cash = math.floor(cash / (entry * (1 + BUY_FEE)))
            adv_limit = math.floor(max(0.0, candidate["adv20"]) * max_participation / entry) if candidate["adv20"] > 0 else 0
            shares = max(0, min(shares_by_risk, shares_by_weight, shares_by_cash, adv_limit))
            if shares <= 0:
                continue
            entry_cost = shares * entry * (1 + BUY_FEE)
            cash -= entry_cost
            positions[symbol] = {
                "signal_date": candidate["signal_date"], "entry_date": day,
                "entry_price": entry, "shares": shares, "entry_cost": entry_cost,
                "stop": stop, "high_water": entry, "atr": atr,
                "bars_held": 0, "pending_exit": False, "pending_reason": "",
                "entry_momentum": candidate["momentum"],
                "estimated_adv20_participation": shares * entry / candidate["adv20"] if candidate["adv20"] > 0 else np.nan,
            }

        # Intraday stop checks include positions entered at today's open.
        for symbol in list(positions):
            if symbol not in today:
                continue
            p, row = positions[symbol], today[symbol]
            open_px, low_px = float(row.open), float(row.low)
            if low_px <= p["stop"]:
                fill = min(open_px, p["stop"]) * (1 - SLIPPAGE)
                proceeds = p["shares"] * fill * (1 - SELL_FEE_TAX)
                pnl = proceeds - p["entry_cost"]
                reason = "gap_stop" if open_px <= p["stop"] else "atr_stop"
                trades.append({**p, "symbol": symbol, "exit_date": day,
                               "exit_price": fill, "exit_reason": reason,
                               "net_return": _trade_cost_return(p["entry_price"], fill),
                               "pnl": pnl})
                cash += proceeds
                del positions[symbol]

        # Only completed bars update trailing protection and exit instructions.
        for symbol in list(positions):
            if symbol not in today:
                continue
            p, row = positions[symbol], today[symbol]
            close, high = float(row.close), float(row.high)
            p["bars_held"] += 1
            p["high_water"] = max(p["high_water"], high, close)
            atr_now = float(row.atr) if pd.notna(row.atr) and float(row.atr) > 0 else p["atr"]
            p["stop"] = max(p["stop"], p["high_water"] - TRAIL_ATR * atr_now)
            p["atr"] = atr_now
            if p["bars_held"] >= int(exit_mode.split("_")[1]):
                p["pending_exit"] = True
                p["pending_reason"] = f"{exit_mode}_next_open"
            last_close[symbol] = close

        # Liquidate a symbol on its own final available bar if it ends before the portfolio calendar.
        for symbol in list(positions):
            if last_dates.get(symbol) != day or symbol not in today:
                continue
            p = positions[symbol]
            fill = float(today[symbol].close) * (1 - SLIPPAGE)
            proceeds = p["shares"] * fill * (1 - SELL_FEE_TAX)
            end_reason = "period_end_close" if day == end_ts else "symbol_data_end_close"
            trades.append({**p, "symbol": symbol, "exit_date": day,
                           "exit_price": fill, "exit_reason": end_reason,
                           "net_return": _trade_cost_return(p["entry_price"], fill),
                           "pnl": proceeds - p["entry_cost"]})
            cash += proceeds
            del positions[symbol]

        for symbol, row in today.items():
            last_close[symbol] = float(row.close)
        market_value = sum(p["shares"] * last_close.get(s, p["entry_price"]) for s, p in positions.items())
        equity = cash + market_value
        equity_rows.append({
            "date": day, "equity": equity, "cash": cash,
            "market_value": market_value,
            "exposure": market_value / equity if equity > 0 else 0.0,
            "positions": len(positions),
        })

    # Safety close if any symbol has no final row on the shared calendar.
    for symbol, p in list(positions.items()):
        symbol_dates = [d for d in dates if d in row_lookup[symbol]]
        if not symbol_dates:
            continue
        exit_date = symbol_dates[-1]
        row = row_lookup[symbol][exit_date]
        fill = float(row.close) * (1 - SLIPPAGE)
        proceeds = p["shares"] * fill * (1 - SELL_FEE_TAX)
        trades.append({**p, "symbol": symbol, "exit_date": exit_date,
                       "exit_price": fill, "exit_reason": "period_end_close",
                       "net_return": _trade_cost_return(p["entry_price"], fill),
                       "pnl": proceeds - p["entry_cost"]})
        cash += proceeds
    if equity_rows:
        equity_rows[-1].update({"equity": cash, "cash": cash, "market_value": 0.0, "exposure": 0.0, "positions": 0})
    trade_frame = pd.DataFrame(trades)
    equity_frame = pd.DataFrame(equity_rows)
    metrics = summarize_portfolio(equity_frame, trade_frame, initial_capital)
    metrics.update({"strategy": strategy, "exit_mode": exit_mode, "period_start": str(start_ts.date()), "period_end": str(end_ts.date())})
    return metrics, trade_frame, equity_frame


def summarize_portfolio(equity: pd.DataFrame, trades: pd.DataFrame, initial_capital: float) -> dict:
    if equity.empty:
        return {"trades": 0, "ending_equity": initial_capital, "total_return": 0.0, "CAGR": 0.0, "max_drawdown": 0.0, "Sharpe": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "avg_exposure": 0.0}
    values = equity.equity.astype(float)
    returns = values.pct_change().fillna(values.iloc[0] / initial_capital - 1)
    years = max((pd.Timestamp(equity.date.iloc[-1]) - pd.Timestamp(equity.date.iloc[0])).days / 365.25, 1/252)
    ending = float(values.iloc[-1])
    cagr = (ending / initial_capital) ** (1 / years) - 1 if ending > 0 else -1.0
    drawdown = values / np.maximum.accumulate(np.r_[initial_capital, values.to_numpy()])[1:] - 1
    daily_vol = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0.0
    pnl = pd.to_numeric(trades.get("pnl", pd.Series(dtype=float)), errors="coerce")
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else (float("inf") if len(wins) else 0.0)
    trade_returns = pd.to_numeric(trades.get("net_return", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "trades": int(len(trades)),
        "win_rate": float((trade_returns > 0).mean()) if len(trade_returns) else 0.0,
        "avg_trade_net": float(trade_returns.mean()) if len(trade_returns) else 0.0,
        "median_trade_net": float(trade_returns.median()) if len(trade_returns) else 0.0,
        "profit_factor": pf,
        "ending_equity": ending,
        "total_return": ending / initial_capital - 1,
        "CAGR": float(cagr),
        "max_drawdown": float(np.min(drawdown)) if len(drawdown) else 0.0,
        "Sharpe": sharpe,
        "avg_exposure": float(equity.exposure.mean()),
        "avg_positions": float(equity.positions.mean()),
        "avg_hold_bars": float(trades.bars_held.mean()) if len(trades) and "bars_held" in trades else 0.0,
    }


def equal_weight_buy_hold(frames: dict[str, pd.DataFrame], start: str, end: str,
                          initial_capital: float = INITIAL_CAPITAL) -> tuple[dict, pd.DataFrame]:
    """10-stock equal-dollar buy-and-hold benchmark with the same costs."""
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    selected = {}
    all_dates = set()
    for symbol, frame in frames.items():
        d = frame.loc[(frame.date >= a) & (frame.date <= b)].sort_values("date")
        if d.empty:
            continue
        selected[symbol] = d
        all_dates.update(pd.to_datetime(d.date).tolist())
    dates = sorted(all_dates)
    if not dates:
        return {}, pd.DataFrame()
    selected_rows = {
        s: {pd.Timestamp(r.date): r for r in d.itertuples(index=False)}
        for s, d in selected.items()
    }
    cash = float(initial_capital)
    shares, last_prices = {}, {}
    allocation = initial_capital / max(len(selected), 1)
    rows = []
    last_dates = {s: max(lookup) for s, lookup in selected_rows.items()}
    for day in dates:
        for symbol, lookup in selected_rows.items():
            if symbol in shares or day not in lookup:
                continue
            row = lookup[day]
            price = float(row.open) * (1 + SLIPPAGE)
            quantity = math.floor(allocation / (price * (1 + BUY_FEE)))
            cost = quantity * price * (1 + BUY_FEE)
            if quantity > 0 and cost <= cash:
                shares[symbol] = quantity
                cash -= cost
        for symbol, lookup in selected_rows.items():
            if day in lookup:
                last_prices[symbol] = float(lookup[day].close)
        for symbol in list(shares):
            if last_dates.get(symbol) == day:
                proceeds = shares.pop(symbol) * last_prices[symbol] * (1 - SLIPPAGE) * (1 - SELL_FEE_TAX)
                cash += proceeds
        market_value = sum(q * last_prices.get(s, 0.0) for s, q in shares.items())
        rows.append({"date": day, "equity": cash + market_value, "cash": cash, "market_value": market_value})
    for symbol, quantity in shares.items():
        proceeds = quantity * last_prices[symbol] * (1 - SLIPPAGE) * (1 - SELL_FEE_TAX)
        cash += proceeds
    rows[-1].update({"equity": cash, "cash": cash, "market_value": 0.0})
    curve = pd.DataFrame(rows)
    vals = curve.equity.astype(float)
    years = max((pd.Timestamp(curve.date.iloc[-1]) - pd.Timestamp(curve.date.iloc[0])).days / 365.25, 1/252)
    dd = vals / np.maximum.accumulate(np.r_[initial_capital, vals.to_numpy()])[1:] - 1
    rets = vals.pct_change().fillna(vals.iloc[0] / initial_capital - 1)
    vol = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    metrics = {
        "strategy": "equal_weight_buy_hold", "stocks": len(selected),
        "ending_equity": float(vals.iloc[-1]), "total_return": float(vals.iloc[-1] / initial_capital - 1),
        "CAGR": float((vals.iloc[-1] / initial_capital) ** (1 / years) - 1),
        "max_drawdown": float(dd.min()),
        "Sharpe": float(rets.mean() / vol * np.sqrt(252)) if vol > 0 else 0.0,
    }
    return metrics, curve


def run_research(input_dir: str | Path, output_dir: str | Path) -> dict:
    frames, manifest = load_adjusted_frames(input_dir)
    latest = max(pd.Timestamp(d.date.max()) for d in frames.values())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metric_rows, trade_frames, equity_frames, benchmark_rows, benchmark_curves = [], [], [], [], []
    for period, (start, configured_end) in PERIODS.items():
        end = min(pd.Timestamp(configured_end), latest) if configured_end else latest
        if end <= pd.Timestamp(start):
            continue
        for strategy in STRATEGIES:
            for exit_mode in EXIT_MODES:
                metrics, trades, equity = simulate_portfolio(
                    frames, strategy, exit_mode, start, str(end.date()),
                )
                metrics["period"] = period
                metric_rows.append(metrics)
                if not trades.empty:
                    trades = trades.copy(); trades["strategy"] = strategy; trades["exit_mode"] = exit_mode; trades["period"] = period
                    trade_frames.append(trades)
                if not equity.empty:
                    equity = equity.copy(); equity["strategy"] = strategy; equity["exit_mode"] = exit_mode; equity["period"] = period
                    equity_frames.append(equity)
        bm, curve = equal_weight_buy_hold(frames, start, str(end.date()))
        bm["period"] = period
        benchmark_rows.append(bm)
        if not curve.empty:
            curve = curve.copy(); curve["period"] = period
            benchmark_curves.append(curve)
    metrics_frame = pd.DataFrame(metric_rows)
    validation = metrics_frame.loc[(metrics_frame.period == "validation") & (metrics_frame.trades >= 30)].copy()
    validation = validation.sort_values(["profit_factor", "CAGR"], ascending=False)
    selected = None
    test_for_selected = None
    if not validation.empty:
        best = validation.iloc[0]
        selected = {"strategy": best.strategy, "exit_mode": best.exit_mode,
                    "selection_basis": "validation period only; test period excluded from selection",
                    "validation_CAGR": float(best.CAGR), "validation_profit_factor": float(best.profit_factor),
                    "validation_trades": int(best.trades)}
        match = metrics_frame.loc[(metrics_frame.period == "test") &
                                  (metrics_frame.strategy == best.strategy) &
                                  (metrics_frame.exit_mode == best.exit_mode)]
        if not match.empty:
            row = match.iloc[0]
            test_for_selected = {k: (float(row[k]) if isinstance(row[k], (float, np.floating, int, np.integer)) else row[k])
                                 for k in ["trades", "win_rate", "profit_factor", "ending_equity", "total_return", "CAGR", "max_drawdown", "Sharpe", "avg_exposure"]}
    benchmark_by_period = [
        {k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v)
         for k, v in row.items()}
        for row in benchmark_rows
    ]
    test_benchmark = next((x for x in benchmark_by_period if x.get("period") == "test"), None)
    test_comparison = None
    if test_for_selected is not None and test_benchmark is not None:
        test_comparison = {
            "selected_strategy_CAGR": test_for_selected.get("CAGR"),
            "equal_weight_buy_hold_CAGR": test_benchmark.get("CAGR"),
            "CAGR_difference": test_for_selected.get("CAGR", 0.0) - test_benchmark.get("CAGR", 0.0),
            "selected_strategy_max_drawdown": test_for_selected.get("max_drawdown"),
            "equal_weight_buy_hold_max_drawdown": test_benchmark.get("max_drawdown"),
        }
    report = {
        "data_last_date": str(latest.date()),
        "data_provider_by_symbol": {x["symbol"]: x["provider"] for x in manifest.get("status_rows", [])},
        "data_selection_note": "十檔股票依 2026 年近期流動性事後選取，存在存活者及選股偏誤；不是歷史逐日可投資股票池。",
        "price_basis_note": "使用還原 OHLC；股利/拆併股已反映於調整價格，不另加股利，避免重複計算。",
        "execution_note": "收盤訊號、下一交易日開盤進場；停損採跳空開盤或觸價成交；固定持有期訊號於期滿後下一開盤出場；整數股、共用現金、最多五檔、每筆風險預算 0.5%、單檔權重上限 20%、ADV20 參與率上限 1%。",
        "cost_note": "買進手續費 0.1425% + 滑價 0.10%；賣出手續費 0.1425% + 證交稅 0.30% + 滑價 0.10%。未納入券商折扣、融資利息及漲跌停無法成交的額外衝擊。",
        "validation_selection_rule": "只使用 2021-2023 驗證期；至少 30 筆交易，依交易損益 Profit Factor 排序，再以 CAGR 作為次順位。2024 年以後測試期不參與選擇。",
        "validation_selected": selected,
        "locked_configuration_test_result": test_for_selected,
        "equal_weight_buy_hold_benchmark": benchmark_by_period,
        "locked_test_vs_benchmark": test_comparison,
    }
    metrics_frame.to_csv(output / "strategy_metrics.csv", index=False)
    pd.concat(trade_frames, ignore_index=True).to_csv(output / "trade_log.csv", index=False) if trade_frames else pd.DataFrame().to_csv(output / "trade_log.csv", index=False)
    pd.concat(equity_frames, ignore_index=True).to_csv(output / "equity_curves.csv", index=False) if equity_frames else pd.DataFrame().to_csv(output / "equity_curves.csv", index=False)
    pd.DataFrame(benchmark_rows).to_csv(output / "benchmark_metrics.csv", index=False)
    pd.concat(benchmark_curves, ignore_index=True).to_csv(output / "benchmark_curves.csv", index=False) if benchmark_curves else pd.DataFrame().to_csv(output / "benchmark_curves.csv", index=False)
    (output / "research_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/top10_adjusted")
    parser.add_argument("--output", default="artifacts/top10_shared_capital_research")
    args = parser.parse_args()
    print(json.dumps(run_research(args.input, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
