"""V8: hold V6 entries fixed and compare predeclared causal exits."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from tsmc_2330_causal_warning_walkforward_v6 import build_data,fit_model,FEATURES,HORIZON
BUY_FEE,SELL_FEE_TAX,SLIPPAGE,CAPITAL,BUY_THRESHOLD=.001425,.004425,.001,100000,.65

def score(d):
 d=d.copy(); d["p_low"]=np.nan; d["p_high"]=np.nan; training=[]
 for year in sorted(d.date.dt.year.unique()):
  if year<2016: continue
  mask=d.date.dt.year.eq(year); start=int(d.index[mask][0])
  train=d.loc[(d.index+HORIZON<start)&d.y_low.notna()]
  ml,mh=fit_model(train,"y_low"),fit_model(train,"y_high")
  if ml is None or mh is None:
   training.append({"year":int(year),"status":"insufficient","train_rows":len(train)}); continue
  ix=d.index[mask & d[FEATURES].notna().any(axis=1)]
  d.loc[ix,"p_low"]=ml.predict_proba(d.loc[ix,FEATURES])[:,1]
  d.loc[ix,"p_high"]=mh.predict_proba(d.loc[ix,FEATURES])[:,1]
  training.append({"year":int(year),"status":"fit","train_rows":len(train)})
 tr=pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1)
 d["atr14"]=tr.rolling(14).mean(); d["ma60"]=d.close.rolling(60).mean()
 return d,training

def simulate(d,start,policy):
 cash=float(CAPITAL); qty=0; entry=0.; entry_i=None; peak=0.; pending=None; curve=[]; trades=[]
 for i in range(start,len(d)):
  r=d.iloc[i]
  if pending:
   action=pending; px=float(r.open)*(1+SLIPPAGE if action=="buy" else 1-SLIPPAGE)
   if action=="buy" and qty==0:
    q=int(cash/(px*(1+BUY_FEE)))
    if q: cash-=q*px*(1+BUY_FEE); qty=q; entry=px; entry_i=i; peak=float(r.close); trades.append({"entry_date":str(r.date.date()),"entry_price":px,"shares":q})
   elif action=="sell" and qty:
    proceeds=qty*px*(1-SELL_FEE_TAX); cash+=proceeds
    trades[-1].update({"exit_date":str(r.date.date()),"exit_price":px,"net_return":proceeds/(qty*entry*(1+BUY_FEE))-1,"hold_sessions":i-entry_i})
    qty=0; entry=0.; entry_i=None; peak=0.
   pending=None
  curve.append(cash+qty*float(r.close))
  if qty:
   peak=max(peak,float(r.close)); held=i-entry_i
   if policy=="v6_baseline":
    stop=r.close<=entry*.92 or held>=60 or (pd.notna(r.p_high) and r.p_high>=.60)
   elif policy in ("atr_3x","atr_4x"):
    mult=3 if policy=="atr_3x" else 4
    stop=(pd.notna(r.atr14) and r.close<=peak-mult*float(r.atr14)) or held>=252
   elif policy=="warning_plus_ma60":
    stop=((pd.notna(r.p_high) and r.p_high>=.60 and pd.notna(r.ma60) and r.close<r.ma60) or r.close<=entry*.88 or held>=252)
   else: raise ValueError(policy)
   if stop: pending="sell"
  elif pd.notna(r.p_low) and r.p_low>=BUY_THRESHOLD: pending="buy"
 if qty:
  r=d.iloc[-1]; px=float(r.close)*(1-SLIPPAGE); proceeds=qty*px*(1-SELL_FEE_TAX); cash+=proceeds
  trades[-1].update({"exit_date":str(r.date.date()),"exit_price":px,"net_return":proceeds/(qty*entry*(1+BUY_FEE))-1,"hold_sessions":len(d)-1-entry_i}); curve[-1]=cash
 return np.asarray(curve,float),[t for t in trades if "exit_date" in t]

def stats(curve,dates):
 years=(dates.iloc[-1]-dates.iloc[0]).days/365.25
 return {"final_equity":round(float(curve[-1]),2),"CAGR_pct":round(((curve[-1]/CAPITAL)**(1/years)-1)*100,2),"max_drawdown_pct":round(float(np.min(curve/np.maximum.accumulate(curve)-1))*100,2)}

def run(input_csv,outdir):
 d,training=score(build_data(input_csv)); valid=d.index[d.p_low.notna()&d.p_high.notna()]
 if not len(valid): raise RuntimeError("No walk-forward scores")
 start=int(valid[0]); dates=d.date.iloc[start:].reset_index(drop=True)
 policies=["v6_baseline","atr_3x","atr_4x","warning_plus_ma60"]; rows=[]; curves={}; alltrades=[]
 for p in policies:
  curve,trades=simulate(d,start,p); curves[p]=curve
  rows.append({"policy":p,**stats(curve,dates),"closed_trades":len(trades),"win_rate_pct":round(np.mean([t["net_return"]>0 for t in trades])*100,2) if trades else None,"average_net_trade_pct":round(np.mean([t["net_return"] for t in trades])*100,2) if trades else None})
  alltrades += [{"policy":p,**t} for t in trades]
 q=int(CAPITAL/(float(d.open.iloc[start])*(1+BUY_FEE))); residual=CAPITAL-q*float(d.open.iloc[start])*(1+BUY_FEE)
 bh=residual+q*d.close.iloc[start:].to_numpy(float); bh[-1]=residual+q*float(d.close.iloc[-1])*(1-SELL_FEE_TAX)
 rows.append({"policy":"buy_and_hold",**stats(bh,dates),"closed_trades":0,"win_rate_pct":None,"average_net_trade_pct":None}); curves["buy_and_hold"]=bh
 out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows).to_csv(out/"exit_policy_comparison.csv",index=False); pd.DataFrame(alltrades).to_csv(out/"trades.csv",index=False)
 pd.DataFrame(training).to_csv(out/"yearly_training.csv",index=False); pd.DataFrame({"date":dates,**curves}).to_csv(out/"equity_curves.csv",index=False)
 summary={"version":"V8","symbol":"2330","history_rows":len(d),"history_start":str(d.date.min().date()),"oos_start":str(d.date.iloc[start].date()),"data_end":str(d.date.iloc[-1].date()),"data_source":"Yahoo Finance auto_adjust=True","initial_capital":CAPITAL,"entry":"V6 p_low >= 0.65; close signal, next-session-open fill","exit_policies":{"v6_baseline":"8% close stop, p_high >= 0.60, or 60-session cap","atr_3x":"3 x ATR(14) trailing from highest close; 252-session cap","atr_4x":"4 x ATR(14) trailing from highest close; 252-session cap","warning_plus_ma60":"p_high >= 0.60 AND close < MA60, or 12% close stop, or 252-session cap"},"costs":{"buy_fee":BUY_FEE,"sell_fee_and_tax":SELL_FEE_TAX,"slippage_each_side":SLIPPAGE},"results":rows,"limitations":["Exploratory reuse of previously examined 2016-2026 OOS; not a pristine holdout.","Single stock; no generalization or guaranteed return.","Yahoo adjusted OHLCV and simplified fills are not official execution-grade data.","Close exits fill next open; gaps, market impact, queues and partial fills are not modeled."]}
 (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
 return summary
if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--input",default="data/top10_adjusted_prices/bars/2330.csv"); p.add_argument("--output",default="artifacts/tsmc_v8_exit_policies"); a=p.parse_args()
 print(json.dumps(run(a.input,a.output),ensure_ascii=False,indent=2))
