"""V19 single-stock TSMC pullback research with conservative daily-bar execution.

Signals are evaluated at close and filled at next open. Raw OHLC is used for fills;
cash-dividend reference-price factors are used only to back-adjust indicator history.
This is exploratory research, not a production strategy.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

BUY_FEE, SELL_FEE_TAX, SLIPPAGE = .001425, .004425, .001
CAPITAL = 100_000.0
PERIODS = {
    "research": ("2010-01-04", "2017-12-29"),
    "validation": ("2018-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2026-11-06"),
}

def load_data(price_path: str, action_path: str):
    d = pd.read_csv(price_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    a = pd.read_csv(action_path, parse_dates=["date"]).sort_values("date")
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(d.columns):
        raise ValueError(f"Missing OHLCV columns: {required - set(d.columns)}")
    if d.date.duplicated().any():
        raise ValueError("Duplicate trading dates in 2330 price file")
    px = d[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(px).all() or (px <= 0).any():
        raise ValueError("Invalid raw OHLC values")
    if ((d.high < d[["open", "low", "close"]].max(axis=1)) |
        (d.low > d[["open", "high", "close"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC range")
    factor = np.ones(len(d), dtype=float)
    for ev in a.itertuples():
        f = float(ev.reference_price) / float(ev.before_price)
        factor[d.date.to_numpy() < np.datetime64(ev.date)] *= f
    for col in ("open", "high", "low", "close"):
        d[f"adj_{col}"] = d[col].to_numpy(float) * factor
    return d, a, factor

def features(d):
    c, h, l, v = d.adj_close, d.adj_high, d.adj_low, d.volume
    change = c.diff()
    up = change.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-change.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / down.replace(0, np.nan))
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    for n in (5, 10, 20, 50, 100, 200):
        d[f"ma{n}"] = c.rolling(n).mean()
        d[f"ret{n}"] = c.pct_change(n)
        d[f"dev{n}"] = c / d[f"ma{n}"] - 1
        d[f"high{n}"] = h.shift(1).rolling(n).max()
        d[f"dd{n}"] = c / d[f"high{n}"] - 1
        d[f"vol{n}"] = v / v.rolling(n).mean()
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["bbw"] = 4 * sd / mid
    return d

def make_signals(d):
    c = d.adj_close
    ma50up = d.ma50 > d.ma200
    loc = (c-d.adj_low) / (d.adj_high-d.adj_low).replace(0, np.nan)
    s = {
        "pullback_reclaim": ma50up & (c>d.ma200) & d.dev20.between(-.10,-.015) & (d.rsi.shift(1)<45) & (d.rsi>d.rsi.shift(1)) & (c>c.shift(1)),
        "capitulation_reversal": (d.ret5<-.04) & (d.vol20>1.25) & (loc>.70) & (c>d.adj_open) & (d.rsi<50),
        "ma10_reclaim": ma50up & (c>d.ma200) & (c>d.ma10) & (c.shift(1)<=d.ma10.shift(1)) & (d.dd20<-.025) & d.rsi.between(35,60),
        "squeeze_breakout": (d.bbw<d.bbw.rolling(120,min_periods=80).quantile(.25)) & (c>d.high20) & (c>d.ma50) & ma50up & (d.vol20>1.1),
        "momentum_restart": ma50up & (c>d.ma20) & (c.shift(1)<=d.ma20.shift(1)) & (d.ret20>0) & d.rsi.between(45,70),
        "gap_recovery": (d.adj_open/d.adj_close.shift(1)-1<-.012) & (c>d.adj_open) & (loc>.65) & (d.vol20>1.1),
        "trend_dip_3_7": (c>d.ma200) & d.dd20.between(-.07,-.03) & d.rsi.between(30,55) & (d.rsi>d.rsi.shift(1)) & (c>d.adj_open),
        "trend_dip_ma20": (c>d.ma200) & d.dev20.between(-.07,-.01) & d.rsi.between(30,52) & (c>d.adj_open),
        "trend_dip_3day": (c>d.ma200) & (d.ret5<-.025) & (c>d.adj_open) & (c>c.shift(1)) & (d.rsi<60),
        "trend_dip_lowvol": (c>d.ma200) & d.dd20.between(-.08,-.02) & (d.vol20<1.3) & (d.rsi<55) & (c>d.adj_open),
        "pullback_zone_1_5": (c>d.ma200) & d.dd20.between(-.05,-.01) & (c>c.shift(1)),
        "pullback_zone_1_5_bull": (c>d.ma200) & d.dd20.between(-.05,-.01) & (c>d.adj_open) & (c>c.shift(1)),
        "pullback_zone_1_5_rsi": (c>d.ma200) & d.dd20.between(-.05,-.01) & d.rsi.between(30,65) & (d.rsi>d.rsi.shift(1)),
        "pullback_zone_3_7": (c>d.ma200) & d.dd20.between(-.07,-.03) & (c>c.shift(1)),
    }
    return {k: v.fillna(False).to_numpy(bool) for k,v in s.items()}

def simulate(d, actions, factor, signal, atr_mult, stop_pct, target_pct, max_hold, exit_ma,
             start_date, end_date, cost_mult=1.0):
    idx = np.flatnonzero((d.date>=start_date) & (d.date<=end_date))
    start, end = int(idx[0]), int(idx[-1])
    op, hi, lo, cl = (d[x].to_numpy(float) for x in ("open","high","low","close"))
    adjc = d.adj_close.to_numpy(float)
    atr_raw = d.atr.to_numpy(float) / factor
    atr_prev = np.r_[np.nan, atr_raw[:-1]]
    ma = d[f"ma{exit_ma}"].to_numpy(float)
    dates = d.date.to_numpy()
    dividends = {np.datetime64(x.date): float(x.stock_and_cache_dividend) for x in actions.itertuples()}
    cash, qty, entry, entry_i, peak = CAPITAL, 0, 0.0, None, 0.0
    entry_fee, pending, trades, curve = 0.0, None, [], []
    for i in range(start, end+1):
        dt = dates[i]
        if qty and dt in dividends:
            cash += qty * dividends[dt]
        if pending == "buy" and qty == 0:
            fill = op[i] * (1 + SLIPPAGE*cost_mult)
            q = math.floor(cash / (fill*(1+BUY_FEE*cost_mult)))
            if q:
                qty, entry, entry_i, peak = q, fill, i, fill
                entry_fee = BUY_FEE*cost_mult
                cash -= qty*fill*(1+entry_fee)
                trades.append({"entry_date":str(pd.Timestamp(dt).date()),"entry_price":fill,"shares":qty})
        elif pending == "sell" and qty:
            fill = op[i]*(1-SLIPPAGE*cost_mult)
            proceeds = qty*fill*(1-SELL_FEE_TAX*cost_mult)
            cash += proceeds
            trades[-1].update({"exit_date":str(pd.Timestamp(dt).date()),"exit_price":fill,
                "net_return":proceeds/(qty*entry*(1+entry_fee))-1,"hold_sessions":i-entry_i,"exit":"next_open_signal"})
            qty, entry, entry_i, peak = 0, 0.0, None, 0.0
        pending = None
        if qty:
            stop = max(entry*(1-stop_pct), peak-atr_mult*atr_prev[i]) if np.isfinite(atr_prev[i]) else entry*(1-stop_pct)
            target = entry*(1+target_pct)
            fill, reason = None, None
            if op[i] <= stop:
                fill, reason = op[i]*(1-SLIPPAGE*cost_mult), "gap_stop"
            elif lo[i] <= stop and hi[i] >= target:
                fill, reason = stop*(1-SLIPPAGE*cost_mult), "both_hit_stop_first"
            elif lo[i] <= stop:
                fill, reason = stop*(1-SLIPPAGE*cost_mult), "stop"
            elif op[i] >= target:
                fill, reason = op[i]*(1-SLIPPAGE*cost_mult), "gap_target"
            elif hi[i] >= target:
                fill, reason = target*(1-SLIPPAGE*cost_mult), "target"
            if fill is not None:
                proceeds = qty*fill*(1-SELL_FEE_TAX*cost_mult)
                cash += proceeds
                trades[-1].update({"exit_date":str(pd.Timestamp(dt).date()),"exit_price":fill,
                    "net_return":proceeds/(qty*entry*(1+entry_fee))-1,"hold_sessions":i-entry_i,"exit":reason})
                qty, entry, entry_i, peak = 0, 0.0, None, 0.0
            else:
                peak = max(peak, hi[i])
                if i-entry_i >= max_hold or (np.isfinite(ma[i]) and adjc[i] < ma[i]):
                    pending = "sell"
        elif signal[i]:
            pending = "buy"
        curve.append((dt, cash+qty*cl[i], qty))
    if qty:
        fill = cl[end]*(1-SLIPPAGE*cost_mult)
        proceeds = qty*fill*(1-SELL_FEE_TAX*cost_mult)
        cash += proceeds
        trades[-1].update({"exit_date":str(pd.Timestamp(dates[end]).date()),"exit_price":fill,
            "net_return":proceeds/(qty*entry*(1+entry_fee))-1,"hold_sessions":end-entry_i,"exit":"period_end"})
        curve[-1] = (dates[end], cash, 0)
    eq = pd.DataFrame(curve,columns=["date","equity","shares"])
    years = (pd.Timestamp(dates[end])-pd.Timestamp(dates[start])).days/365.25
    cagr = (cash/CAPITAL)**(1/years)-1
    dd = (eq.equity/eq.equity.cummax()-1).min()
    daily = eq.equity.pct_change().dropna()
    vol = daily.std(ddof=1)*np.sqrt(252)*100
    sharpe = daily.mean()/daily.std(ddof=1)*np.sqrt(252) if daily.std(ddof=1)>0 else np.nan
    closed = [t for t in trades if "exit_date" in t]
    returns = np.array([t["net_return"] for t in closed])
    return {"final":cash,"cagr_pct":cagr*100,"mdd_pct":dd*100,"trades":len(closed),
        "win_pct":(returns>0).mean()*100 if len(returns) else np.nan,
        "avg_trade_pct":returns.mean()*100 if len(returns) else np.nan,
        "exposure_pct":(eq.shares>0).mean()*100,"ann_vol_pct":vol,"sharpe_rf0":sharpe,
        "calmar":cagr*100/abs(dd*100) if dd<0 else np.nan}, eq, closed

def buy_hold(d, actions, start_date, end_date, cost_mult=1.0):
    idx=np.flatnonzero((d.date>=start_date)&(d.date<=end_date)); start,end=int(idx[0]),int(idx[-1])
    fill=float(d.open.iloc[start])*(1+SLIPPAGE*cost_mult)
    qty=math.floor(CAPITAL/(fill*(1+BUY_FEE*cost_mult)))
    cash=CAPITAL-qty*fill*(1+BUY_FEE*cost_mult)
    divs={np.datetime64(x.date):float(x.stock_and_cache_dividend) for x in actions.itertuples()}
    eq=[]
    for i in range(start,end+1):
        if i>start and d.date.to_numpy()[i] in divs: cash+=qty*divs[d.date.to_numpy()[i]]
        value=cash+qty*float(d.close.iloc[i])
        eq.append(value)
    final=cash+qty*float(d.close.iloc[end])*(1-SLIPPAGE*cost_mult)*(1-SELL_FEE_TAX*cost_mult)
    eq[-1]=final
    years=(d.date.iloc[end]-d.date.iloc[start]).days/365.25
    series=pd.Series(eq); daily=series.pct_change().dropna()
    dd=(series/series.cummax()-1).min()
    cagr=((final/CAPITAL)**(1/years)-1)*100
    vol=daily.std(ddof=1)*np.sqrt(252)*100
    sharpe=daily.mean()/daily.std(ddof=1)*np.sqrt(252) if daily.std(ddof=1)>0 else np.nan
    return {"final":final,"cagr_pct":cagr,"mdd_pct":dd*100,"ann_vol_pct":vol,
        "sharpe_rf0":sharpe,"calmar":cagr/abs(dd*100) if dd<0 else np.nan}


def simulate_fixed_hold(d, actions, signal, start_date, end_date, hold_sessions, cost_mult=1.0):
    idx=np.flatnonzero((d.date>=start_date)&(d.date<=end_date)); start,end=int(idx[0]),int(idx[-1])
    op=d.open.to_numpy(float); cl=d.close.to_numpy(float); dates=d.date.to_numpy()
    divs={np.datetime64(x.date):float(x.stock_and_cache_dividend) for x in actions.itertuples()}
    cash,qty,entry,entry_i,pending=CAPITAL,0,0.0,None,False
    trades=[]; curve=[]
    for i in range(start,end+1):
        dt=dates[i]
        if qty and dt in divs: cash+=qty*divs[dt]
        if pending and qty==0:
            fill=op[i]*(1+SLIPPAGE*cost_mult); q=math.floor(cash/(fill*(1+BUY_FEE*cost_mult)))
            if q:
                qty,entry,entry_i=q,fill,i
                cash-=qty*fill*(1+BUY_FEE*cost_mult)
                trades.append({"entry_date":str(pd.Timestamp(dt).date()),"entry_price":fill,"shares":qty})
            pending=False
        if qty and i-entry_i>=hold_sessions:
            fill=op[i]*(1-SLIPPAGE*cost_mult); proceeds=qty*fill*(1-SELL_FEE_TAX*cost_mult)
            cash+=proceeds
            trades[-1].update({"exit_date":str(pd.Timestamp(dt).date()),"exit_price":fill,
                "net_return":proceeds/(qty*entry*(1+BUY_FEE*cost_mult))-1,
                "hold_sessions":i-entry_i,"exit":"fixed_horizon_open"})
            qty,entry,entry_i=0,0.0,None
        if qty==0 and signal[i]: pending=True
        curve.append((dt,cash+qty*cl[i],qty))
    if qty:
        fill=cl[end]*(1-SLIPPAGE*cost_mult); proceeds=qty*fill*(1-SELL_FEE_TAX*cost_mult)
        cash+=proceeds
        trades[-1].update({"exit_date":str(pd.Timestamp(dates[end]).date()),"exit_price":fill,
            "net_return":proceeds/(qty*entry*(1+BUY_FEE*cost_mult))-1,
            "hold_sessions":end-entry_i,"exit":"period_end"})
        curve[-1]=(dates[end],cash,0)
    eq=pd.DataFrame(curve,columns=["date","equity","shares"])
    years=(pd.Timestamp(dates[end])-pd.Timestamp(dates[start])).days/365.25
    cagr=(cash/CAPITAL)**(1/years)-1; dd=(eq.equity/eq.equity.cummax()-1).min()
    daily=eq.equity.pct_change().dropna()
    vol=daily.std(ddof=1)*np.sqrt(252)*100
    sharpe=daily.mean()/daily.std(ddof=1)*np.sqrt(252) if daily.std(ddof=1)>0 else np.nan
    closed=[t for t in trades if "exit_date" in t]; rets=np.array([t["net_return"] for t in closed])
    metrics={"final":cash,"cagr_pct":cagr*100,"mdd_pct":dd*100,"trades":len(closed),
        "win_pct":(rets>0).mean()*100 if len(rets) else np.nan,
        "avg_trade_pct":rets.mean()*100 if len(rets) else np.nan,
        "exposure_pct":(eq.shares>0).mean()*100,"ann_vol_pct":vol,"sharpe_rf0":sharpe}
    return metrics,eq,closed

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="data/taiwan_history/bars/2330.csv")
    p.add_argument("--actions",default="data/taiwan_history/corporate_actions/by_symbol/2330.csv")
    p.add_argument("--output",default="artifacts/tsmc_v19")
    args=p.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    d,a,factor=load_data(args.input,args.actions); d=features(d); signals=make_signals(d)
    rows=[]
    for family,sig in signals.items():
        for atr in (3.0,4.5,6.0):
            for stop in (.08,.12):
                for target in (.15,.99):
                    for hold in (100,252):
                        for exit_ma in (50,100,200):
                            row={"family":family,"atr":atr,"stop":stop,"target":target,"hold":hold,"exit_ma":exit_ma}
                            for period,(lo,hi) in PERIODS.items():
                                metrics,_,_=simulate(d,a,factor,sig,atr,stop,target,hold,exit_ma,lo,hi)
                                row.update({f"{period}_{k}":v for k,v in metrics.items()})
                            rows.append(row)
    grid=pd.DataFrame(rows)
    grid["min_train_val_cagr_pct"]=grid[["research_cagr_pct","validation_cagr_pct"]].min(axis=1)
    grid["mean_train_val_cagr_pct"]=grid[["research_cagr_pct","validation_cagr_pct"]].mean(axis=1)
    eligible=grid[(grid.research_cagr_pct>0)&(grid.validation_cagr_pct>0)]
    if eligible.empty: raise RuntimeError("No positive research/validation candidate")
    selected=eligible.sort_values(["min_train_val_cagr_pct","mean_train_val_cagr_pct"],ascending=False).iloc[0]
    grid.to_csv(out/"candidate_grid.csv",index=False)
    selected_signal=signals[selected.family]
    metrics_rows=[]; trade_rows=[]; curve_rows=[]; yearly=[]
    for period,(lo,hi) in PERIODS.items():
        metrics,eq,trades=simulate(d,a,factor,selected_signal,float(selected.atr),float(selected.stop),
            float(selected.target),int(selected.hold),int(selected.exit_ma),lo,hi)
        metrics_rows.append({"period":period,**metrics})
        trade_rows.extend({"period":period,**t} for t in trades)
        eq["period"]=period; eq["year"]=pd.to_datetime(eq.date).dt.year
        curve_rows.append(eq)
        prev=CAPITAL
        for year,g in eq.groupby("year",sort=True):
            ending=float(g.equity.iloc[-1])
            yearly.append({"period":period,"year":int(year),"return_pct":(ending/prev-1)*100,"ending_equity":ending})
            prev=ending
    pd.DataFrame(metrics_rows).to_csv(out/"selected_metrics.csv",index=False)
    pd.DataFrame(trade_rows).to_csv(out/"selected_trades.csv",index=False)
    pd.concat(curve_rows,ignore_index=True).to_csv(out/"selected_equity.csv",index=False)
    pd.DataFrame(yearly).to_csv(out/"selected_yearly.csv",index=False)
    bench=[{"period":name,**buy_hold(d,a,*span)} for name,span in PERIODS.items()]
    pd.DataFrame(bench).to_csv(out/"buy_hold.csv",index=False)
    fixed_rows=[]; fixed_trades=[]
    for family in ("pullback_zone_1_5","pullback_zone_3_7"):
        for hold_days in (10,20,40):
            row={"family":family,"hold_sessions":hold_days}
            for period,(lo,hi) in PERIODS.items():
                m,_,tr=simulate_fixed_hold(d,a,signals[family],lo,hi,hold_days)
                row.update({f"{period}_{k}":v for k,v in m.items()})
                fixed_trades.extend({"family":family,"hold_sessions":hold_days,"period":period,**t} for t in tr)
            fixed_rows.append(row)
    fixed_grid=pd.DataFrame(fixed_rows)
    fixed_grid["min_train_val_cagr_pct"]=fixed_grid[["research_cagr_pct","validation_cagr_pct"]].min(axis=1)
    fixed_eligible=fixed_grid[(fixed_grid.research_cagr_pct>0)&(fixed_grid.validation_cagr_pct>0)]
    fixed_selected=fixed_eligible.sort_values("min_train_val_cagr_pct",ascending=False).iloc[0]
    fixed_family=str(fixed_selected.family); fixed_hold=int(fixed_selected.hold_sessions)
    fixed_stress=[]
    for mult in (1.0,1.5,2.0):
        for period,(lo,hi) in PERIODS.items():
            m,_,_=simulate_fixed_hold(d,a,signals[fixed_family],lo,hi,fixed_hold,cost_mult=mult)
            fixed_stress.append({"cost_mult":mult,"period":period,**m})
    fixed_grid.to_csv(out/"fixed_hold_grid.csv",index=False)
    pd.DataFrame(fixed_trades).to_csv(out/"fixed_hold_trades.csv",index=False)
    pd.DataFrame(fixed_stress).to_csv(out/"fixed_hold_cost_stress.csv",index=False)
    fixed_summary={"family":fixed_family,"hold_sessions":fixed_hold,
        "selection":"highest minimum research/validation CAGR among six predeclared fixed-horizon candidates",
        "metrics":{period:{k:float(fixed_selected[f"{period}_{k}"]) for k in
            ("cagr_pct","mdd_pct","trades","win_pct","exposure_pct")} for period in PERIODS},
        "cost_stress":fixed_stress}
    (out/"fixed_hold_selected.json").write_text(json.dumps(fixed_summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    stress=[]
    for mult in (1.0,1.5,2.0):
        for name,(lo,hi) in PERIODS.items():
            m,_,_=simulate(d,a,factor,selected_signal,float(selected.atr),float(selected.stop),
                float(selected.target),int(selected.hold),int(selected.exit_ma),lo,hi,cost_mult=mult)
            stress.append({"cost_mult":mult,"period":name,**m})
    pd.DataFrame(stress).to_csv(out/"cost_stress.csv",index=False)
    summary={"version":"V19","symbol":"2330","source":"FinMind API v4 TaiwanStockPrice cached raw OHLCV and cached corporate-action reference prices",
        "rows":len(d),"start":str(d.date.min().date()),"end":str(d.date.max().date()),"corporate_action_rows":len(a),
        "candidate_count":len(grid),"selection":"maximize the minimum CAGR across research and validation, requiring both positive; test not used for selection",
        "selected_candidate":{k:(float(v) if isinstance(v,(np.floating,float)) else int(v) if isinstance(v,(np.integer,int)) else str(v)) for k,v in selected.to_dict().items()},
        "selected_metrics":metrics_rows,"buy_hold":bench,"cost_stress":stress,"fixed_horizon_candidate":fixed_summary,
        "execution":"close signal, next open fills; raw OHLC intraday stop/target; gap fills at open; stop-first if both barriers touched; ATR uses prior-session value; whole-share cash-only",
        "costs":{"buy_fee":BUY_FEE,"sell_fee_tax":SELL_FEE_TAX,"slippage_each_side":SLIPPAGE},
        "limitations":["FinMind cached vendor data; direct TWSE reconciliation not completed.",
            "Historical indicator prices are back-adjusted using cash-dividend reference-price factors; raw prices are used for fills.",
            "Gross cash dividends credited on ex-date; withholding tax/payment delay omitted.",
            "No price-limit queues, suspensions, partial fills, market impact or intraday path beyond daily OHLC.",
            "Daily OHLC cannot reveal whether stop or target occurred first; if both touched, stop is assumed first.",
            "The 2022-2026 test period has been examined in prior work and is not a pristine blind holdout.",
            "This is exploratory single-stock research, not a live-trading approval or return guarantee."]}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))

if __name__=="__main__":
    main()
