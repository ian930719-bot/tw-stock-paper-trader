"""V14: trend-core plus trend-gated tactical sleeve; close signals, next-open fills."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

CAP=100000.; BF=.001425; SF=.004425; SLIP=.001
SPLITS={"research":("2010-01-04","2017-12-29"),"validation":("2018-01-01","2021-12-31"),"test":("2022-01-01","2026-11-10")}
FAMILIES=("squeeze_breakout","trend_pullback_reentry","momentum_breakout","short_rebound")

def prepare(path):
 d=pd.read_csv(path,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
 for k in ("open","high","low","close","volume"): d[k]=pd.to_numeric(d[k],errors="coerce")
 c,h,l,v=d.close,d.high,d.low,d.volume
 for n in (20,50,200): d[f"ma{n}"]=c.rolling(n).mean()
 d["ema5"]=c.ewm(span=5,adjust=False).mean(); d["ema10"]=c.ewm(span=10,adjust=False).mean()
 ch=c.diff(); up=ch.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); dn=(-ch.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
 d["rsi"]=100-100/(1+up/dn.replace(0,np.nan))
 tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
 d["atr"]=tr.rolling(14).mean(); d["slope50"]=d.ma50.pct_change(20)
 d["dd20"]=c/c.rolling(20).max()-1
 d["hi10"]=h.shift(1).rolling(10).max(); d["hi20"]=h.shift(1).rolling(20).max()
 mid=c.rolling(20).mean(); sd=c.rolling(20).std()
 d["bbw"]=4*sd/mid; d["bbwq"]=d.bbw.rolling(120,min_periods=60).quantile(.25)
 d["vr"]=v/v.rolling(20).mean(); d["regime"]=(d.ma50>d.ma200).fillna(False)
 c=d.close
 sig={
 "squeeze_breakout":(d.bbw<d.bbwq)&(c>d.hi20)&(c>d.ma50)&(d.ma50>d.ma200)&(d.vr>1.15),
 "trend_pullback_reentry":(c>d.ema10)&(c.shift()<=d.ema10.shift())&(d.ma50>d.ma200)&(c>d.ma200)&d.dd20.between(-.12,-.025)&d.rsi.between(40,65),
 "momentum_breakout":(c>d.hi10)&(c>d.ma50)&(d.slope50>0)&d.rsi.between(50,75),
 "short_rebound":(c>d.ema5)&(c.shift()<=d.ema5.shift())&(d.slope50>0)&d.rsi.between(35,60)&(d.dd20<-.02)}
 for k,x in sig.items(): d["sig_"+k]=x.fillna(False).to_numpy(bool)
 return d

def spans(d):
 out={}
 for k,(a,b) in SPLITS.items():
  ix=d.index[(d.date>=a)&(d.date<=b)]
  if not len(ix): raise ValueError("missing "+k)
  out[k]=(int(ix[0]),int(ix[-1]))
 return out

def stats(curve,final,d,s,e,n,exp,capital):
 years=(d.date.iloc[e]-d.date.iloc[s]).days/365.25; a=np.asarray(curve,float)
 return {"final_equity":float(final),"cagr_pct":float(((final/capital)**(1/years)-1)*100),
 "max_drawdown_pct":float((a/np.maximum.accumulate(a)-1).min()*100),"closed_trades":int(n),
 "exposure_pct":float(np.mean(exp)*100)}

def simulate(d,s,e,capital,kind,family=None):
 o=d.open.to_numpy(float); c=d.close.to_numpy(float); reg=d.regime.to_numpy(bool)
 cash=capital; q=0; pending=0; entry=peak=0.; entry_i=-1; n=0; curve=[]; exp=[]
 sig=d["sig_"+family].to_numpy(bool) if family else None
 for i in range(s,e+1):
  if kind=="bh" and i==s:
   px=o[i]*(1+SLIP); q=int(cash/(px*(1+BF))); cash-=q*px*(1+BF)
  elif pending:
   px=o[i]*(1+SLIP if pending==1 else 1-SLIP)
   if pending==1 and q==0:
    qty=int(cash/(px*(1+BF)))
    if qty: cash-=qty*px*(1+BF); q=qty; entry=px; entry_i=i; peak=c[i]
   elif pending==-1 and q:
    cash+=q*px*(1-SF); q=0; entry=peak=0.; entry_i=-1; n+=1
   pending=0
  curve.append(cash+q*c[i]); exp.append(q>0)
  if kind=="bh": continue
  if kind=="core":
   if q and not reg[i]: pending=-1
   elif not q and reg[i]: pending=1
  elif q:
   peak=max(peak,c[i]); atr=d.atr.iloc[i]; stop=(np.isfinite(atr) and atr>0 and c[i]<=peak-2.5*atr)
   if stop or c[i]<=entry*.92 or c[i]<d.ma20.iloc[i] or i-entry_i>=40 or not reg[i]: pending=-1
  elif reg[i] and sig[i]: pending=1
 if q:
  px=c[e]*(1-SLIP); cash+=q*px*(1-SF); q=0; n+=1; curve[-1]=cash
 return stats(curve,cash,d,s,e,n,exp,capital),np.asarray(curve),np.asarray(exp,bool)

def run(path,outdir):
 d=prepare(path); ss=spans(d); scores={}
 for f in FAMILIES: scores[f]=simulate(d,*ss["research"],CAP,"tactical",f)[0]
 selected=max(FAMILIES,key=lambda f:scores[f]["cagr_pct"])
 rows=[]
 for period,(s,e) in ss.items():
  for name,kind,fam,alloc in [("100pct_core","core",None,0),("100pct_buy_hold","bh",None,0),("100pct_tactical","tactical",selected,1)]:
   m,_,_=simulate(d,s,e,CAP,kind,fam); rows.append({"period":period,"strategy":name+(("_"+fam) if fam else ""),**m})
  for a in (.10,.20,.30):
   cm,cc,ce=simulate(d,s,e,CAP*(1-a),"core")
   tm,tc,te=simulate(d,s,e,CAP*a,"tactical",selected)
   combined=cc+tc; final=cm["final_equity"]+tm["final_equity"]
   m=stats(combined,final,d,s,e,cm["closed_trades"]+tm["closed_trades"],ce|te,CAP)
   m.update({"core_allocation_pct":(1-a)*100,"tactical_allocation_pct":a*100,
   "core_final_equity":cm["final_equity"],"tactical_final_equity":tm["final_equity"]})
   rows.append({"period":period,"strategy":f"{int((1-a)*100)}pct_core_{int(a*100)}pct_tactical",**m})
 out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows).to_csv(out/"allocation_comparison.csv",index=False)
 summary={"version":"V14","symbol":"2330","rows":len(d),"start":str(d.date.min().date()),"end":str(d.date.max().date()),
 "source":str(d.provider.iloc[0]) if "provider" in d else "adjusted OHLCV",
 "splits":{k:{"start":str(d.date.iloc[a].date()),"end":str(d.date.iloc[b].date())} for k,(a,b) in ss.items()},
 "core_rule":"MA50 > MA200; close signal, next open fill; exit when regime turns false.",
 "tactical_rule":f"{selected}; entries gated by MA50 > MA200; 2.5 ATR trailing stop, MA20 close exit, 40 sessions max, 8% close stop.",
 "selected_family":selected,"research_family_scores":scores,"results":rows,
 "costs":{"buy_fee":BF,"sell_fee_and_tax":SF,"slippage_each_side":SLIP},
 "limitations":["Yahoo Finance auto_adjust=True adjusted OHLCV, not official raw execution data.","Long-only, cash-only, integer shares; no leverage, market impact, limit queues or partial fills.","Each split resets to NT$100,000; tactical family selected only on 2010-2017 research CAGR.","2022-2026 test segment was examined in previous work and is not a pristine holdout.","Single-stock exploratory research; no return guarantee."]}
 (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
 return summary

if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--input",default="data/top10_adjusted_prices/bars/2330.csv"); p.add_argument("--output",default="artifacts/tsmc_v14_core_tactical"); a=p.parse_args()
 print(json.dumps(run(a.input,a.output),ensure_ascii=False,indent=2,default=str))
