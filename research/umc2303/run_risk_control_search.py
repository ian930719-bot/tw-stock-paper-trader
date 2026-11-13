#!/usr/bin/env python3
"""Risk-budgeted UMC 2303 strategy search; daily signals, next-open execution."""
import csv, json, math
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
    for i in range(n-1,len(x)): a[i]=sum(x[i-n+1:i+1])/n
    return a

def atr(rows,n=14):
    tr=[]
    for i,r in enumerate(rows):
        p=rows[i-1]["close"] if i else r["close"]
        tr.append(max(r["high"]-r["low"],abs(r["high"]-p),abs(r["low"]-p)))
    return sma(tr,n)

def prior_max(x,n):
    a=[None]*len(x)
    for i in range(n,len(x)): a[i]=max(x[i-n:i])
    return a

def metrics(eq,trades):
    vals=[v for _,v in eq]
    peak=INITIAL; dd=0
    for v in vals:
        peak=max(peak,v); dd=min(dd,v/peak-1)
    years=max((date.fromisoformat(eq[-1][0])-date.fromisoformat(eq[0][0])).days/365.25,1/365.25)
    cagr=(vals[-1]/INITIAL)**(1/years)-1 if vals[-1]>0 else -1
    wins=[t["pnl"] for t in trades if t["pnl"]>0]
    losses=[-t["pnl"] for t in trades if t["pnl"]<0]
    gp=sum(wins); gl=sum(losses)
    return {"cagr_pct":round(cagr*100,2),"total_return_pct":round((vals[-1]/INITIAL-1)*100,2),
      "max_drawdown_pct":round(dd*100,2),"trades":len(trades),
      "win_rate_pct":round(100*len(wins)/len(trades),2) if trades else 0,
      "profit_factor":round(gp/gl,3) if gl else None,"ending_equity":round(vals[-1],2)}

def run(rows,signal,ma_exit,atrv,start,end,risk_pct,stop_mult,max_exposure,dd_limit,max_hold,cooldown):
    cash=INITIAL; qty=0; cost=0; entry_i=None; stop=None; entry_atr=None
    pending=None; age=0; peak_equity=INITIAL; halted=False; cooldown_until=-1
    trades=[]; eq=[]; halt_reason=None
    for i in range(start,end):
        r=rows[i]
        if pending=="exit" and qty:
            px=r["open"]*(1-SLIP); proceeds=qty*px*(1-FEE-TAX)
            trades.append({"entry_date":rows[entry_i]["date"],"exit_date":r["date"],"pnl":proceeds-cost,"reason":"signal_or_guard"})
            cash+=proceeds; qty=0; cost=0; entry_i=None; stop=None; age=0
        elif pending=="entry" and not qty and not halted and i>cooldown_until:
            px=r["open"]*(1+SLIP)
            eq_open=cash
            a=atrv[i-1] if i>0 and atrv[i-1] else px*0.04
            stop_distance=max(stop_mult*a,px*0.015)
            risk_cash=eq_open*risk_pct
            risk_qty=int(risk_cash/stop_distance)
            exposure_qty=int(eq_open*max_exposure/px)
            cash_qty=int(cash/(px*(1+FEE)))
            n=max(0,min(risk_qty,exposure_qty,cash_qty))
            if n>0:
                gross=n*px; cost=gross*(1+FEE); cash-=cost; qty=n
                entry_i=i; entry_atr=a; stop=px-stop_distance; age=0
        pending=None
        stopped=False
        if qty and stop is not None and r["low"]<=stop:
            raw=min(r["open"],stop); px=raw*(1-SLIP)
            proceeds=qty*px*(1-FEE-TAX)
            trades.append({"entry_date":rows[entry_i]["date"],"exit_date":r["date"],"pnl":proceeds-cost,"reason":"atr_stop"})
            cash+=proceeds; qty=0; cost=0; entry_i=None; stop=None; age=0; stopped=True
            cooldown_until=i+cooldown
        if qty and not stopped:
            age+=1
            a=atrv[i] if atrv[i] else entry_atr
            stop=max(stop,r["close"]-stop_mult*a)
            if age>=max_hold or (ma_exit[i] is not None and r["close"]<ma_exit[i]):
                pending="exit"
        equity=cash+qty*r["close"]
        peak_equity=max(peak_equity,equity)
        dd=equity/peak_equity-1 if peak_equity else 0
        if dd<=-dd_limit and not halted:
            halted=True; halt_reason={"date":r["date"],"drawdown_pct":round(dd*100,2)}
            if qty: pending="exit"
        if not qty and not halted and i<end-1 and signal[i] and i>cooldown_until:
            pending="entry"
        eq.append((r["date"],equity))
    if qty:
        r=rows[end-1]; px=r["close"]*(1-SLIP); proceeds=qty*px*(1-FEE-TAX)
        trades.append({"entry_date":rows[entry_i]["date"],"exit_date":r["date"],"pnl":proceeds-cost,"reason":"period_end"})
        cash+=proceeds; eq[-1]=(r["date"],cash)
    return metrics(eq,trades),halt_reason

def main():
    with SRC.open(encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    for r in rows:
        for k in ("volume","open","high","low","close"): r[k]=float(r[k])
    rows=[r for r in rows if min(r["open"],r["high"],r["low"],r["close"],r["volume"])>0]
    c=[r["close"] for r in rows]; h=[r["high"] for r in rows]; v=[r["volume"] for r in rows]
    ma20=sma(c,20); ma50=sma(c,50); v20=sma(v,20); a14=atr(rows,14)
    highs={n:prior_max(h,n) for n in (10,20,40)}
    start=next(i for i,r in enumerate(rows) if r["date"]>="2013-01-01"); end=len(rows)
    signals={}
    for n in (10,20,40):
        for filt in ("none","ma20","ma50"):
            sig=[False]*len(rows)
            for i in range(101,len(rows)):
                trend=(filt=="none" or (filt=="ma20" and c[i]>ma20[i] and ma20[i]>ma20[i-1]) or
                       (filt=="ma50" and c[i]>ma50[i] and ma20[i]>ma50[i]))
                sig[i]=bool(highs[n][i] and c[i]>highs[n][i] and trend and v[i]>=v20[i-1]*0.8)
            signals[f"breakout{n}_{filt}"]=sig
    # A predeclared risk-control grid: risk budget, ATR stop, exposure cap, and permanent drawdown circuit breaker.
    results=[]
    for name,sig in signals.items():
      for risk in (0.0025,0.005,0.0075):
       for stop in (1.5,2.0,2.5):
        for exposure in (0.5,0.75,1.0):
         for dd_limit in (0.08,0.09):
          for hold in (10,20,40):
           m,halt=run(rows,sig,ma20,a14,start,end,risk,stop,exposure,dd_limit,hold,5)
           results.append({"strategy":name,"risk_per_trade_pct":risk*100,"atr_stop":stop,
             "max_exposure_pct":exposure*100,"drawdown_circuit_pct":dd_limit*100,"max_hold":hold,
             **m,"circuit_breaker":halt})
    # Filter only strategies that actually respected the 10% observed drawdown ceiling.
    eligible=[x for x in results if x["max_drawdown_pct"]>=-10.0]
    eligible.sort(key=lambda x:(x["cagr_pct"],x["profit_factor"] or 0,x["trades"]),reverse=True)
    results.sort(key=lambda x:(x["cagr_pct"],x["max_drawdown_pct"]),reverse=True)
    report={"symbol":"2303","source":"TWSE official STOCK_DAY raw daily OHLCV; unadjusted",
      "data_first_date":rows[0]["date"],"data_last_date":rows[-1]["date"],"data_rows":len(rows),
      "test_start":rows[start]["date"],"candidate_count":len(results),"eligible_under_10pct_mdd":len(eligible),
      "risk_method":"Whole-share, cash-only risk sizing: quantity is minimum of risk-budget/ATR stop distance, exposure cap and available cash. ATR trailing stop, MA20 close exit, max holding period, five-session cooldown after stop, permanent circuit breaker when marked-to-market drawdown reaches 8% or 9%; liquidation occurs at next open and can gap through the threshold.",
      "costs":{"commission_each_side_pct":0.1425,"sell_tax_pct":0.3,"slippage_each_side_pct":0.1},
      "limitations":["Raw unadjusted prices; corporate actions and dividends are not adjusted.",
       "Daily bars cannot guarantee a hard 10% maximum drawdown because overnight gaps can exceed the circuit threshold.",
       "Circuit breaker halts trading permanently after trigger; results must be interpreted alongside the halt date.",
       "Parameter search creates selection bias; top candidate requires untouched forward validation.",
       "No leverage, short selling, market impact, partial fills, queue or limit-lock model."],
      "ranking":"Primary shortlist: highest CAGR among candidates whose observed full-period MDD is no worse than -10%; this is exploratory, not out-of-sample proof.",
      "top_30_under_10pct_mdd":eligible[:30],"top_30_unfiltered":results[:30]}
    (OUT/"umc2303_risk_control_search.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT/"umc2303_risk_control_search.csv").open("w",newline="",encoding="utf-8-sig") as f:
      fields=list(results[0]); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(results)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
