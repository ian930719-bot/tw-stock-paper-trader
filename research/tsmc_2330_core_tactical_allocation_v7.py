"""V7 comparison: V6 tactical sleeve versus core buy-and-hold allocation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from tsmc_2330_causal_warning_walkforward_v6 import build_data, fit_model, FEATURES, HORIZON

BUY_FEE, SELL_FEE_TAX, SLIPPAGE = .001425, .004425, .001
CAPITAL = 100000

def scores(d):
    d=d.copy(); d["p_low"]=np.nan; d["p_high"]=np.nan
    yearly=[]
    for year in sorted(d.date.dt.year.unique()):
        if year<2016: continue
        mask=d.date.dt.year.eq(year); start=int(d.index[mask][0])
        train=d.loc[(d.index+HORIZON<start)&d.y_low.notna()]
        ml=fit_model(train,"y_low"); mh=fit_model(train,"y_high")
        if ml is None or mh is None:
            yearly.append({"year":int(year),"status":"insufficient"}); continue
        ix=d.index[mask & d[FEATURES].notna().any(axis=1)]
        d.loc[ix,"p_low"]=ml.predict_proba(d.loc[ix,FEATURES])[:,1]
        d.loc[ix,"p_high"]=mh.predict_proba(d.loc[ix,FEATURES])[:,1]
        yearly.append({"year":int(year),"status":"fit","train_rows":len(train)})
    return d,yearly

def tactical(d,start,capital):
    cash=float(capital); qty=0; entry=0.; entry_i=None; pending=None
    curve=[]; trades=[]; invested=[]
    for i in range(start,len(d)):
        r=d.iloc[i]
        if pending:
            px=float(r.open)*(1+SLIPPAGE if pending=="buy" else 1-SLIPPAGE)
            if pending=="buy" and qty==0:
                q=int(cash/(px*(1+BUY_FEE)))
                if q: cash-=q*px*(1+BUY_FEE); qty=q; entry=px; entry_i=i; trades.append({"entry_date":str(r.date.date()),"entry_price":px,"shares":q})
            elif pending=="sell" and qty:
                proceeds=qty*px*(1-SELL_FEE_TAX); cash+=proceeds
                trades[-1].update({"exit_date":str(r.date.date()),"exit_price":px,"net_return":proceeds/(qty*entry*(1+BUY_FEE))-1,"hold_sessions":i-entry_i})
                qty=0; entry=0.; entry_i=None
            pending=None
        curve.append(cash+qty*float(r.close)); invested.append(qty>0)
        if qty:
            if r.close<=entry*.92 or i-entry_i>=60 or (pd.notna(r.p_high) and r.p_high>=.60): pending="sell"
        elif pd.notna(r.p_low) and r.p_low>=.65: pending="buy"
    if qty:
        r=d.iloc[-1]; px=float(r.close)*(1-SLIPPAGE); proceeds=qty*px*(1-SELL_FEE_TAX); cash+=proceeds
        trades[-1].update({"exit_date":str(r.date.date()),"exit_price":px,"net_return":proceeds/(qty*entry*(1+BUY_FEE))-1,"hold_sessions":len(d)-1-entry_i})
        curve[-1]=cash
    return np.asarray(curve),trades,float(np.mean(invested))

def stats(curve,dates):
    years=(dates.iloc[-1]-dates.iloc[0]).days/365.25
    peak=np.maximum.accumulate(curve)
    return {"final_equity":round(float(curve[-1]),2),"CAGR_pct":round(((curve[-1]/CAPITAL)**(1/years)-1)*100,2),
      "max_drawdown_pct":round(float(np.min(curve/peak-1))*100,2)}

def run(input_csv,outdir):
    d=build_data(input_csv); d,yearly=scores(d)
    valid=d.index[d.p_low.notna()&d.p_high.notna()]
    if not len(valid): raise RuntimeError("No out-of-sample model scores")
    start=int(valid[0]); dates=d.date.iloc[start:].reset_index(drop=True)
    rows=[]; curves={}
    # Benchmark: integer shares, buy at first OOS open, liquidate at final close.
    q=int(CAPITAL/(float(d.open.iloc[start])*(1+BUY_FEE)))
    cash=CAPITAL-q*float(d.open.iloc[start])*(1+BUY_FEE)
    bh=cash+q*d.close.iloc[start:].to_numpy(float); bh[-1]=cash+q*float(d.close.iloc[-1])*(1-SELL_FEE_TAX)
    rows.append({"variant":"100% buy-and-hold core",**stats(bh,dates),"tactical_allocation_pct":0,"tactical_exposure_pct":0,"closed_trades":0,"win_rate_pct":None})
    for weight in (1.0,.5,.3,.1):
        tac,tr,exposure=tactical(d,start,CAPITAL*weight)
        # Scale tactical sleeve to total-account-equivalent metrics for the pure tactical comparator.
        if weight==1:
            rows.append({"variant":"100% V6 tactical",**stats(tac,dates),"tactical_allocation_pct":100,"tactical_exposure_pct":round(exposure*100,2),"closed_trades":sum("exit_date" in x for x in tr),"win_rate_pct":round(np.mean([x["net_return"]>0 for x in tr if "exit_date" in x])*100,2)})
            curves["100pct_tactical"]=tac
            continue
        core_weight=1-weight; cq=int(CAPITAL*core_weight/(float(d.open.iloc[start])*(1+BUY_FEE)))
        cc=CAPITAL*core_weight-cq*float(d.open.iloc[start])*(1+BUY_FEE)
        hybrid=cc+cq*d.close.iloc[start:].to_numpy(float)+tac
        hybrid[-1]=cc+cq*float(d.close.iloc[-1])*(1-SELL_FEE_TAX)+tac[-1]
        rows.append({"variant":f"{core_weight:.0%} core + {weight:.0%} V6 tactical",**stats(hybrid,dates),
          "tactical_allocation_pct":weight*100,"tactical_exposure_pct":round(exposure*weight*100,2),
          "closed_trades":sum("exit_date" in x for x in tr),
          "win_rate_pct":round(np.mean([x["net_return"]>0 for x in tr if "exit_date" in x])*100,2)})
        curves[f"hybrid_{weight:.1f}"]=hybrid
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"allocation_comparison.csv",index=False)
    pd.DataFrame(yearly).to_csv(out/"yearly_model_training.csv",index=False)
    pd.DataFrame({"date":dates,**{k:v for k,v in curves.items()}}).to_csv(out/"equity_curves.csv",index=False)
    result={"symbol":"2330","oos_start":str(d.date.iloc[start].date()),"data_end":str(d.date.iloc[-1].date()),
      "data_source":"Yahoo Finance auto_adjust=True","initial_capital":CAPITAL,"allocations":rows,
      "method":"Fixed V6 probabilities and entry/exit rules; vary only capital split between buy-and-hold core and tactical sleeve. Next-open fills, integer shares, costs and slippage retained.",
      "caution":"Allocation comparison is a diagnostic, not a new optimized strategy. Single-stock adjusted data, hypothetical fills; not official execution validation."}
    (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input",default="data/top10_adjusted_prices/bars/2330.csv"); p.add_argument("--output",default="artifacts/tsmc_v7_allocation")
    a=p.parse_args(); print(json.dumps(run(a.input,a.output),ensure_ascii=False,indent=2))
