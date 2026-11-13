#!/usr/bin/env python3
"""Fixed-rule time-period robustness checks for UMC 2303 (MA5 + volume only)."""
import csv,json,math
from datetime import date
from pathlib import Path

OUT=Path("research/umc2303/output")
SRC=OUT/"umc2303_twse_daily_raw.csv"
INITIAL=100000.0
FEE=0.001425
TAX=0.003
SLIP=0.001

def sma(x,n):
    a=[None]*len(x)
    for i in range(n-1,len(x)):
        a[i]=sum(x[i-n+1:i+1])/n
    return a

def run(rows,ma,vol20,start,end,threshold,exit_n):
    cash=INITIAL; qty=0; pending=None; below=0
    cost=0.0; entry_i=None; trades=[]; equity=[]
    for i in range(start,end):
        r=rows[i]
        if pending=="exit" and qty:
            px=r["open"]*(1-SLIP)
            proceeds=qty*px*(1-FEE-TAX)
            pnl=proceeds-cost
            trades.append({"entry_date":rows[entry_i]["date"],"exit_date":r["date"],
                           "net_pnl":pnl,"net_return_pct":pnl/cost*100,
                           "holding_sessions":i-entry_i})
            cash+=proceeds; qty=0; cost=0; entry_i=None; below=0
        elif pending=="entry" and not qty:
            px=r["open"]*(1+SLIP)
            n=int(cash/(px*(1+FEE)))
            if n:
                gross=n*px; fee=gross*FEE
                cash-=gross+fee; qty=n; cost=gross+fee; entry_i=i
        pending=None
        equity.append((r["date"],cash+qty*r["close"]))
        if i==end-1: break
        if qty:
            below=below+1 if r["close"]<ma[i] else 0
            if below>=exit_n:
                pending="exit"; below=0
        elif i>0 and ma[i] is not None and ma[i-1] is not None and vol20[i-1]:
            if (r["close"]>ma[i] and rows[i-1]["close"]<=ma[i-1]
                and r["volume"]>=vol20[i-1]*threshold):
                pending="entry"
    if qty:
        r=rows[end-1]; px=r["close"]*(1-SLIP)
        proceeds=qty*px*(1-FEE-TAX); pnl=proceeds-cost
        trades.append({"entry_date":rows[entry_i]["date"],"exit_date":r["date"],
                       "net_pnl":pnl,"net_return_pct":pnl/cost*100,
                       "holding_sessions":end-1-entry_i,"forced_end":True})
        cash+=proceeds; equity[-1]=(r["date"],cash)
    vals=[v for _,v in equity]
    peak=INITIAL; dd=0
    for v in vals:
        peak=max(peak,v); dd=min(dd,v/peak-1)
    days=(date.fromisoformat(equity[-1][0])-date.fromisoformat(equity[0][0])).days
    years=max(days/365.25,1/365.25)
    cagr=(vals[-1]/INITIAL)**(1/years)-1 if vals[-1]>0 else -1
    wins=[t for t in trades if t["net_pnl"]>0]
    losses=[t for t in trades if t["net_pnl"]<0]
    gp=sum(t["net_pnl"] for t in wins); gl=-sum(t["net_pnl"] for t in losses)
    rets=[vals[i]/vals[i-1]-1 for i in range(1,len(vals)) if vals[i-1]>0]
    mu=sum(rets)/len(rets) if rets else 0
    sd=(sum((v-mu)**2 for v in rets)/max(1,len(rets)-1))**0.5 if rets else 0
    return {
      "period_start":equity[0][0],"period_end":equity[-1][0],
      "total_return_pct":round((vals[-1]/INITIAL-1)*100,2),
      "cagr_pct":round(cagr*100,2),"max_drawdown_pct":round(dd*100,2),
      "sharpe":round(mu/sd*math.sqrt(252),3) if sd else 0,
      "trades":len(trades),
      "net_win_rate_pct":round(100*len(wins)/len(trades),2) if trades else 0,
      "profit_factor":round(gp/gl,3) if gl else None,
      "short_trade_pct":round(100*sum(t["holding_sessions"]<=3 for t in trades)/len(trades),2) if trades else 0,
      "ending_equity":round(vals[-1],2)
    }

def main():
    with SRC.open(encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    for r in rows:
        for k in ("volume","open","high","low","close"): r[k]=float(r[k])
    rows=[r for r in rows if r["open"]>0 and r["close"]>0 and r["volume"]>0]
    close=[r["close"] for r in rows]; volume=[r["volume"] for r in rows]
    ma=sma(close,5); v20=sma(volume,20)
    periods=[("2013-2017","2013-01-01","2018-01-01"),
             ("2018-2021","2018-01-01","2022-01-01"),
             ("2022-2026_partial","2022-01-01","2026-11-13")]
    results=[]
    for pname,lo,hi in periods:
        start=next(i for i,r in enumerate(rows) if r["date"]>=lo)
        end=next((i for i,r in enumerate(rows) if r["date"]>=hi),len(rows))
        for threshold in (1.5,2.0):
            for exit_n in (1,2):
                m=run(rows,ma,v20,start,end,threshold,exit_n)
                results.append({"period":pname,"volume_threshold_prior20x":threshold,
                                "exit_confirmation_closes":exit_n,**m})
    report={
      "symbol":"2303","source":"TWSE official STOCK_DAY raw daily OHLCV",
      "data_first_date":rows[0]["date"],"data_last_date":rows[-1]["date"],
      "data_rows":len(rows),
      "method":"Three fixed chronological subperiods. Each starts with NT$100,000 cash and no position. MA5 and prior-20-session average volume only; signal at close, next-open execution; fee 0.1425% each side, sell tax 0.3%, slippage 0.1% each side. No parameter tuning within each segment.",
      "limitations":["Raw unadjusted prices; corporate actions and cash dividends not incorporated.",
                     "Chronological segments are robustness checks, not full nested walk-forward validation.",
                     "Each segment starts with fresh capital and no inherited position.",
                     "Daily OHLCV cannot model queue priority, partial fills, limit locks, or market impact."],
      "results":results}
    (OUT/"umc2303_period_robustness_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT/"umc2303_period_robustness_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=list(results[0].keys()); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(results)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
