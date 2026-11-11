"""TSMC 2330 single-stock parameter sweep; research-only adjusted-price proxy."""
from __future__ import annotations
import argparse, itertools, json, math
from pathlib import Path
import numpy as np
import pandas as pd

BUY_FEE, SELL_FEE_TAX, SLIPPAGE = 0.001425, 0.004425, 0.001
INITIAL_CAPITAL = 100_000.0

def load_data(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    valid = ((d[["open", "high", "low", "close"]] > 0).all(axis=1)
             & (d.volume >= 0)
             & (d.high >= d[["open", "close", "low"]].max(axis=1))
             & (d.low <= d[["open", "close", "high"]].min(axis=1)))
    d = d.loc[valid].drop_duplicates("date", keep="last").reset_index(drop=True)
    c, h, l, v = d.close, d.high, d.low, d.volume
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["rsi"] = 100 - 100/(1 + gain/loss.replace(0, np.nan))
    for n in (20, 50, 100, 150, 200):
        d[f"sma{n}"] = c.rolling(n, min_periods=n).mean()
    for n in (20, 55):
        d[f"prior_high{n}"] = h.shift(1).rolling(n, min_periods=n).max()
    for n in (20, 50):
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    d["mom20"], d["mom60"], d["mom120"] = c.pct_change(20), c.pct_change(60), c.pct_change(120)
    d["volratio"] = v / v.shift(1).rolling(20, min_periods=20).mean()
    return d

def backtest(d: pd.DataFrame, p: tuple, start: str, end: str) -> dict:
    look, ema_n, atr_mult, max_hold, sma_n, vol_min, rsi_max, mode = p
    c, o, h, l = (d[x].to_numpy(float) for x in ("close", "open", "high", "low"))
    atr, rsi, vr = (d[x].to_numpy(float) for x in ("atr", "rsi", "volratio"))
    m20, m60, m120 = (d[x].to_numpy(float) for x in ("mom20", "mom60", "mom120"))
    trend = ((c > d[f"sma{sma_n}"].to_numpy(float))
             & (d.sma50.to_numpy(float) > d[f"sma{sma_n}"].to_numpy(float)) & (m60 > 0))
    if mode == 0:
        signal = trend & (c > d[f"prior_high{look}"].to_numpy(float)) & (vr >= vol_min) & (rsi <= rsi_max)
    else:
        signal = trend & (m20 > 0) & (m120 > 0) & (vr >= vol_min) & (rsi <= rsi_max)
    exit_signal = c < d[f"ema{ema_n}"].to_numpy(float)
    dates = d.date.to_numpy()
    lo, hi = np.datetime64(start), np.datetime64(end)
    cash, qty, entry_cost, entry_price, stop, highwater, held, pending = INITIAL_CAPITAL, 0, 0., 0., 0., 0., 0, False
    equity, pnls = [], []
    for i, day in enumerate(dates):
        if day < lo or day > hi:
            continue
        if qty and (pending or o[i] <= stop):
            fill = o[i] * (1-SLIPPAGE)
            proceeds = qty * fill * (1-SELL_FEE_TAX)
            pnls.append(proceeds-entry_cost); cash += proceeds; qty = 0; pending = False
        if qty == 0 and i > 0 and signal[i-1] and dates[i-1] >= lo:
            fill = o[i] * (1+SLIPPAGE)
            new_qty = math.floor(cash/(fill*(1+BUY_FEE)))
            if new_qty > 0:
                qty = new_qty; entry_price = fill
                entry_cost = qty*fill*(1+BUY_FEE); cash -= entry_cost
                stop = fill-atr_mult*max(atr[i-1], fill*.005)
                highwater, held = fill, 0
        if qty and l[i] <= stop:
            fill = min(o[i], stop)*(1-SLIPPAGE)
            proceeds = qty*fill*(1-SELL_FEE_TAX)
            pnls.append(proceeds-entry_cost); cash += proceeds; qty = 0
        if qty:
            held += 1; highwater = max(highwater, h[i], c[i])
            atr_now = max(atr[i] if np.isfinite(atr[i]) else 0, c[i]*.005)
            stop = max(stop, highwater-atr_mult*atr_now)
            if exit_signal[i] or held >= max_hold:
                pending = True
        equity.append((day, cash+qty*c[i]))
    if qty and equity:
        last_day = equity[-1][0]
        i = int(np.flatnonzero(dates == last_day)[0])
        fill = c[i]*(1-SLIPPAGE); proceeds = qty*fill*(1-SELL_FEE_TAX)
        pnls.append(proceeds-entry_cost); cash += proceeds
        equity[-1] = (last_day, cash)
    if not equity:
        return {}
    values = np.asarray([x[1] for x in equity])
    years = max((equity[-1][0]-equity[0][0]).astype("timedelta64[D]").astype(int)/365.25, 1/365.25)
    dd = values/np.maximum.accumulate(np.r_[INITIAL_CAPITAL, values])[1:]-1
    trades = np.asarray(pnls)
    loss_sum = abs(trades[trades < 0].sum())
    return {"CAGR": float((values[-1]/INITIAL_CAPITAL)**(1/years)-1),
            "max_drawdown": float(dd.min()), "trades": int(len(trades)),
            "profit_factor": float(trades[trades > 0].sum()/loss_sum) if loss_sum else None,
            "ending_equity": float(values[-1])}

def run(input_csv: str, output_dir: str) -> dict:
    d = load_data(input_csv)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    grid = itertools.product([20,55], [20,50], [2.5,3.5,4.5], [20,40,60],
                             [150,200], [1.0,1.2], [75,90], [0,1])
    rows = []
    for p in grid:
        train = backtest(d,p,"2010-01-04","2020-12-31")
        if train.get("trades",0) < 5:
            continue
        rows.append({"params":json.dumps(p), **{f"train_{k}":v for k,v in train.items()}})
    frame = pd.DataFrame(rows).sort_values("train_CAGR",ascending=False)
    frame.head(30).to_csv(out/"tsmc_top30_training.csv",index=False)
    selected = []
    for raw in frame.head(30).params:
        p = tuple(json.loads(raw))
        val = backtest(d,p,"2021-01-04","2023-12-29")
        test = backtest(d,p,"2024-01-02","2026-11-10")
        selected.append({"params":raw,**{f"validation_{k}":v for k,v in val.items()},
                         **{f"historical_test_{k}":v for k,v in test.items()}})
    selected_df = pd.DataFrame(selected).sort_values("validation_CAGR",ascending=False)
    selected_df.to_csv(out/"tsmc_validation_test.csv",index=False)
    best = selected_df.iloc[0].to_dict()
    report = {"symbol":"2330","name":"台積電","research_status":"RESEARCH_ONLY",
              "data_semantics":"Yahoo Finance auto_adjust=True adjusted OHLC; adjusted open is only a fill proxy, not raw exchange execution",
              "data_rows":int(len(d)),"data_start":str(d.date.min().date()),"data_end":str(d.date.max().date()),
              "grid_combinations":576,"training_candidates":int(len(frame)),"validation_selected":best}
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    return report

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="data/top10_adjusted_prices/bars/2330.csv")
    ap.add_argument("--output",default="artifacts/tsmc_2330_parameter_sweep")
    args=ap.parse_args()
    print(json.dumps(run(args.input,args.output),ensure_ascii=False,indent=2,default=str))
