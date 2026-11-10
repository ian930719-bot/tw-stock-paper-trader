"""V13: fixed trend-regime exposure filters for TSMC 2330.
Signals are close-based and filled next open; each regime is tested independently.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

CAPITAL=100_000.0
BUY_FEE,SELL_FEE_TAX,SLIPPAGE=.001425,.004425,.001
SPLITS={"research":("2010-01-04","2017-12-29"),"validation":("2018-01-01","2021-12-31"),"test":("2022-01-01","2026-11-10")}

def prepare(path):
 d=pd.read_csv(path,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
 for col in ("open","close"):d[col]=pd.to_numeric(d[col],errors="coerce")
 c=d.close
 for n in (20,50,100,200):
  d[f"ma{n}"]=c.rolling(n).mean();d[f"slope{n}"]=d[f"ma{n}"].pct_change(20)
 return d

def strategies(d):
 c=d.close
 raw={
  "close_above_ma20":(c>d.ma20,c<=d.ma20),
  "close_above_ma50":(c>d.ma50,c<=d.ma50),
  "close_above_ma100":(c>d.ma100,c<=d.ma100),
  "close_above_ma200":(c>d.ma200,c<=d.ma200),
  "ma50_above_ma200":(d.ma50>d.ma200,d.ma50<=d.ma200),
  "close_above_ma50_and_ma50_gt_ma200":((c>d.ma50)&(d.ma50>d.ma200),(c<=d.ma50)|(d.ma50<=d.ma200)),
  "ma50_gt_ma200_and_slope_positive":((d.ma50>d.ma200)&(d.slope50>0),(d.ma50<=d.ma200)|(d.slope50<=0)),
  "close_above_ma200_and_slope_positive":((c>d.ma200)&(d.slope200>0),(c<=d.ma200)|(d.slope200<=0)),
 }
 return {k:(a.fillna(False).to_numpy(bool),b.fillna(False).to_numpy(bool)) for k,(a,b) in raw.items()}

def simulate(d,buy,sell,start,end):
 o=d.open.to_numpy(float);c=d.close.to_numpy(float);cash=CAPITAL;shares=0;pending=0;curve=[];exposure=[];trades=0
 for i in range(start,end+1):
  if pending:
   if pending==1 and shares==0:
    px=o[i]*(1+SLIPPAGE);qty=int(cash/(px*(1+BUY_FEE)))
    if qty:cash-=qty*px*(1+BUY_FEE);shares=qty
   elif pending==-1 and shares:
    px=o[i]*(1-SLIPPAGE);cash+=shares*px*(1-SELL_FEE_TAX);shares=0;trades+=1
   pending=0
  curve.append(cash+shares*c[i]);exposure.append(shares>0)
  if shares and sell[i]:pending=-1
  elif not shares and buy[i]:pending=1
 if shares:
  px=c[end]*(1-SLIPPAGE);cash+=shares*px*(1-SELL_FEE_TAX);shares=0;trades+=1;curve[-1]=cash
 arr=np.asarray(curve,float);years=(d.date.iloc[end]-d.date.iloc[start]).days/365.25
 return {"final_equity":float(cash),"cagr_pct":float(((cash/CAPITAL)**(1/years)-1)*100),
  "max_drawdown_pct":float(np.min(arr/np.maximum.accumulate(arr)-1)*100),"closed_trades":trades,
  "market_exposure_pct":float(np.mean(exposure)*100)},arr

def benchmark(d,start,end):
 o=d.open.to_numpy(float);c=d.close.to_numpy(float);px=o[start]*(1+SLIPPAGE);qty=int(CAPITAL/(px*(1+BUY_FEE)))
 cash=CAPITAL-qty*px*(1+BUY_FEE);curve=cash+qty*c[start:end+1];curve[-1]=cash+qty*c[end]*(1-SLIPPAGE)*(1-SELL_FEE_TAX)
 years=(d.date.iloc[end]-d.date.iloc[start]).days/365.25
 return {"final_equity":float(curve[-1]),"cagr_pct":float(((curve[-1]/CAPITAL)**(1/years)-1)*100),"max_drawdown_pct":float(np.min(curve/np.maximum.accumulate(curve)-1)*100)}

def run(input_csv,output_dir):
 d=prepare(input_csv);ranges={}
 for name,(lo,hi) in SPLITS.items():
  ix=d.index[(d.date>=lo)&(d.date<=hi)]
  if not len(ix):raise ValueError(f"Missing split {name}")
  ranges[name]=(int(ix[0]),int(ix[-1]))
 rules=strategies(d);rows=[];curves={}
 for name,(buy,sell) in rules.items():
  row={"regime":name}
  for period,(start,end) in ranges.items():
   metrics,curve=simulate(d,buy,sell,start,end)
   for k,v in metrics.items():row[f"{period}_{k}"]=v
   if period=="test":curves[name]=curve
  rows.append(row)
 benchmarks={k:benchmark(d,*v) for k,v in ranges.items()}
 out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows).to_csv(out/"trend_regime_comparison.csv",index=False)
 test_start,test_end=ranges["test"]
 pd.DataFrame({"date":d.date.iloc[test_start:test_end+1].reset_index(drop=True),**curves}).to_csv(out/"test_equity_curves.csv",index=False)
 result={"version":"V13","symbol":"2330","history_rows":len(d),"history_start":str(d.date.min().date()),"data_end":str(d.date.max().date()),
  "source":str(d.provider.iloc[0]) if "provider" in d else "adjusted OHLCV","periods":{k:{"start":str(d.date.iloc[v[0]].date()),"end":str(d.date.iloc[v[1]].date())} for k,v in ranges.items()},
  "initial_capital":CAPITAL,"costs":{"buy_fee":BUY_FEE,"sell_fee_and_tax":SELL_FEE_TAX,"slippage_each_side":SLIPPAGE},
  "benchmarks":benchmarks,"regimes":rows,
  "limitations":["Fixed, small set of predeclared trend regimes; not a parameter sweep.","Signals at close, filled next open; integer shares and full-cash long-only account.","Adjusted Yahoo OHLCV is not official exchange execution data.","No market impact, limit-up/down queues, partial fills or broker constraints.","Historical test segment was examined in prior work and is not a pristine holdout."]}
 (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
 return result

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--input",default="data/top10_adjusted_prices/bars/2330.csv");p.add_argument("--output",default="artifacts/tsmc_v13_trend_regime");a=p.parse_args()
 print(json.dumps(run(a.input,a.output),ensure_ascii=False,indent=2))
