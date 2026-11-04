import numpy as np
import pandas as pd
import yfinance as yf

STRATEGIES = ["Minervini 趨勢模板", "O’Neil CAN SLIM 技術代理", "Livermore 順勢突破"]

def strategy_signals(frame):
    c = frame["close"]
    s50 = c.rolling(50).mean()
    s150 = c.rolling(150).mean()
    s200 = c.rolling(200).mean()
    h52 = frame["high"].rolling(252, min_periods=120).max()
    l52 = frame["low"].rolling(252, min_periods=120).min()
    r20, r60 = c.pct_change(20), c.pct_change(60)
    return {
        STRATEGIES[0]: (
            (c > s50) & (s50 > s150) & (s150 > s200) &
            (s200 > s200.shift(20)) & (c >= l52 * 1.25) &
            (c >= h52 * .75) & (r20 > 0) & (r60 > 0),
            (c < s50) | (c < s150)),
        STRATEGIES[1]: (
            (c > frame["h20"]) & (frame["volume"] > 1.4 * frame["v20"]) &
            (c > s50) & (r20 > 0),
            (c < s50) | (c < frame["ema21"])),
        STRATEGIES[2]: (
            (c > frame["h20"]) & (c > frame["ema50"]) &
            frame["up"] & (r20 > 0),
            (c < frame["ema21"]) | (c < frame["ema50"]))
    }

def daily_strategy_returns(frame, strategy, fee=0.001425, sell_tax=0.003, slippage=0.001):
    z = frame.reset_index(drop=True)
    entries, exits = strategy_signals(z)[strategy]
    state, positions = 0, []
    for i in range(len(z)):
        if state and bool(exits.iloc[i]):
            state = 0
        elif not state and bool(entries.iloc[i]):
            state = 1
        positions.append(state)
    position = pd.Series(positions).shift(1).fillna(0)
    returns = z["close"].pct_change().fillna(0)
    turnover = position.diff().abs().fillna(position.iloc[0])
    costs = turnover * (fee + slippage)
    # Add sell tax only on exits.
    exits_turnover = ((position.shift(1).fillna(0) == 1) & (position == 0)).astype(float)
    net = position * returns - costs - exits_turnover * sell_tax
    return pd.Series(net.to_numpy(), index=pd.to_datetime(z["date"]))

def performance(returns, initial_capital):
    if returns.empty:
        return {"期末資產": initial_capital, "總報酬": 0.0, "年化報酬": 0.0,
                "最大回撤": 0.0, "Sharpe": 0.0, "交易日數": 0}
    equity = initial_capital * (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    years = max(len(returns) / 252, 0.1)
    cagr = (equity.iloc[-1] / initial_capital) ** (1 / years) - 1
    vol = returns.std()
    return {"期末資產": float(equity.iloc[-1]),
            "總報酬": float(equity.iloc[-1] / initial_capital - 1),
            "年化報酬": float(cagr), "最大回撤": float(drawdown.min()),
            "Sharpe": float(returns.mean() / vol * np.sqrt(252)) if vol > 0 else 0.0,
            "交易日數": int(len(returns))}

def fundamental_snapshot(codes):
    rows = []
    for code in codes:
        info = {}
        for suffix in [".TW", ".TWO"]:
            try:
                info = yf.Ticker(code + suffix).get_info()
                if info:
                    break
            except Exception:
                continue
        rows.append({"代號": code, "ROE": info.get("returnOnEquity"),
                     "營收成長率": info.get("revenueGrowth"),
                     "獲利成長率": info.get("earningsGrowth"),
                     "自由現金流": info.get("freeCashflow"),
                     "負債／權益": info.get("debtToEquity"),
                     "本益比": info.get("trailingPE"),
                     "資料狀態": "有回傳" if info else "來源未提供"})
    return pd.DataFrame(rows)
