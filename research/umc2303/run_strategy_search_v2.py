#!/usr/bin/env python3
"""UMC 2303 broad strategy search v2. Signals at close; fills next session open."""
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
    out=[None]*len(x)
    for i in range(n-1,len(x)):
        out[i]=sum(x[i-n+1:i+1])/n
    return out

def atr(rows,n=14):
    tr=[]
    for i,r in enumerate(rows):
        if i==0: tr.append(r["high"]-r["low"])
        else:
            p=rows[i-1]["close"]
            tr.append(max(r["high"]-r["low"],abs(r["high"]-p),abs(r["low"]-p)))
    return sma(tr,n)

def rolling_prior_max(x,n):
    out=[None]*len(x)
    for i in range(n,len(x)):
        out[i]=max(x[i-n:i])
    return out

def metrics(eq,trades):
    if not eq: return {}
    vals=[x[1] for x in eq]
    peak=INITIAL; mdd=0
    for v in vals:
        peak=max(peak,v); mdd=min(mdd,v/peak-1)
    days=(date.fromisoformat(eq[-1][0])-date.fromisoformat(eq[0][0])).days
    years=max(days/365.25,1/365.25)
    cagr=(vals[-1]/INITIAL)**(1/years)-1 if vals[-1]>0 else -1
    wins=[t for t in trades if t["pnl"]>0]
    losses=[t for t in trades if t["pnl"]<0]
    gp=sum(t["pnl"] for t in wins); gl=-sum(t["pnl"] for t in losses)
    daily=[vals[i]/vals[i-1]-1 for i in range(1,len(vals)) if vals[i-1]>0]
    mu=sum(daily)/len(daily) if daily else 0
    sd=(sum((v-mu)**2 for v in daily)/max(1,len(daily)-1))**0.5 if daily else 0
    return {
      "total_return_pct":round((vals[-1]/INITIAL-1)*100,2),
      "cagr_pct":round(cagr*100,2),
      "max_drawdown_pct":round(mdd*100,2),
      "sharpe":round(mu/sd*math.sqrt(252),3) if sd else 0,
      "trades":len(trades),
      "win_rate_pct":round(100*len(wins)/len(trades),2) if trades else 0,
      "profit_factor":round(gp/gl,3) if gl else None,
      "ending_equity":round(vals[-1],2)
    }

def backtest(rows,entry_signal,ma_exit,atr_values,stop_mult,start,end,max_hold):
    cash=INITIAL; qty=0; cost=0; entry_i=None; entry_atr=None
    stop=None; peak_close=None; pending=None; age=0; trades=[]; eq=[]
    for i in range(start,end):
        r=rows[i]
        # Previously confirmed signal executes at today's open.
        if pending=="exit" and qty:
            px=r["open"]*(1-SLIP)
            proceeds=qty*px*(1-FEE-TAX)
            trades.append({"entry_date":rows[entry_i]["date"],"exit_date":r["date"],
                           "pnl":proceeds-cost,"return_pct":(proceeds-cost)/cost*100,
                           "holding_sessions":age,"exit":"signal"})
            cash+=proceeds; qty=0; cost=0; entry_i=None; stop=None; peak_close=None; age=0
        elif pending=="entry" and not qty:
            px=r["open"]*(1+SLIP)
            n=int(cash/(px*(1+FEE)))
            if n>0:
                gross=n*px; cost=gross*(1+FEE); cash-=cost; qty=n
                entry_i=i; entry_atr=atr_values[i-1] if i>0 else None
                if not entry_atr or entry_atr<=0: entry_atr=px*0.04
                stop=px-stop_mult*entry_atr if stop_mult else None
                peak_close=px; age=0
        pending=None
        # Stop fixed from information available before this session; gaps fill at open.
        stopped=False
        if qty and stop is not None and r["low"]<=stop:
            raw=min(r["open"],stop)
            px=raw*(1-SLIP)
            proceeds=qty*px*(1-FEE-TAX)
            trades.append({"entry_date":rows[entry_i]["date"],"exit_date":r["date"],
                           "pnl":proceeds-cost,"return_pct":(proceeds-cost)/cost*100,
                           "holding_sessions":age+1,"exit":"ATR_stop"})
            cash+=proceeds; qty=0; cost=0; entry_i=None; stop=None; peak_close=None; age=0
            stopped=True
        if qty and not stopped:
            age+=1
            peak_close=max(peak_close,r["close"])
            a=atr_values[i] if atr_values[i] else entry_atr
            # Close-derived trailing stop becomes active on the next session.
            if stop is not None:
                stop=max(stop,peak_close-stop_mult*a)
            if age>=max_hold or (ma_exit[i] is not None and r["close"]<ma_exit[i]):
                pending="exit"
        elif not qty and not stopped and i<end-1 and entry_signal[i]:
            pending="entry"
        eq.append((r["date"],cash+qty*r["close"]))
    if qty:
        r=rows[end-1]; px=r["close"]*(1-SLIP)
        proceeds=qty*px*(1-FEE-TAX)
        trades.append({"entry_date":rows[entry_i]["date"],"exit_date":r["date"],
                       "pnl":proceeds-cost,"return_pct":(proceeds-cost)/cost*100,
                       "holding_sessions":age,"exit":"period_end"})
        cash+=proceeds; eq[-1]=(r["date"],cash)
    return metrics(eq,trades),trades

def main():
    if not SRC.exists(): raise RuntimeError("Official TWSE raw CSV missing; run run_research.py first")
    with SRC.open(encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    for r in rows:
        for k in ("volume","open","high","low","close"): r[k]=float(r[k])
    rows=[r for r in rows if r["open"]>0 and r["close"]>0 and r["high"]>0 and r["low"]>0 and r["volume"]>0]
    c=[r["close"] for r in rows]; h=[r["high"] for r in rows]; l=[r["low"] for r in rows]; v=[r["volume"] for r in rows]
    ma10=sma(c,10); ma20=sma(c,20); ma50=sma(c,50); ma100=sma(c,100)
    v20=sma(v,20); a14=atr(rows,14)
    prior_high={n:rolling_prior_max(h,n) for n in (10,20,40,60)}
    prior_close={n:rolling_prior_max(c,n) for n in (10,20,40)}
    # Three structurally different entry families; all conditions use data available at signal close.
    candidates=[]
    for family in ("breakout_high","momentum_close","trend_pullback"):
      for lookback in (10,20,40,60):
       if family=="momentum_close" and lookback==60: continue
       for vol_mult in (0.8,1.0,1.5):
        for trend in ("none","ma20","ma50"):
         sig=[False]*len(rows)
         for i in range(101,len(rows)):
          if not v20[i-1] or v[i]<v20[i-1]*vol_mult: continue
          trend_ok=(trend=="none" or (trend=="ma20" and c[i]>ma20[i] and ma20[i]>ma20[i-1]) or
                    (trend=="ma50" and c[i]>ma50[i] and ma20[i]>ma50[i]))
          if not trend_ok: continue
          if family=="breakout_high":
           sig[i]=prior_high[lookback][i] is not None and c[i]>prior_high[lookback][i]
          elif family=="momentum_close":
           sig[i]=prior_close[lookback][i] is not None and c[i]>prior_close[lookback][i]
          else:
           # Uptrend pullback: intraday test near MA20, then close reclaiming MA10.
           sig[i]=(l[i]<=ma20[i]*1.015 and c[i]>ma10[i] and c[i-1]<=ma10[i-1] and c[i]>ma20[i])
          if sig[i] and i>0 and sig[i-1]: sig[i]=False
         for stop_mult in (2.0,3.0,4.0):
          for hold in (10,20,40,80):
           for exit_ma_n in (10,20):
            exit_ma=ma10 if exit_ma_n==10 else ma20
            name=f"{family}|lb{lookback}|vol{vol_mult}|trend{trend}|atr{stop_mult}|hold{hold}|exitMA{exit_ma_n}"
            candidates.append((name,sig,exit_ma,stop_mult,hold))
    # Separate fixed historical windows. Each starts with fresh capital and no position.
    periods=[("2013-2017","2013-01-01","2018-01-01"),
             ("2018-2021","2018-01-01","2022-01-01"),
             ("2022-2026_partial","2022-01-01","2026-11-13"),
             ("full_2013_2026","2013-01-01","2026-11-13")]
    results=[]
    for name,sig,exit_ma,sm,hold in candidates:
      row={"strategy":name}
      for pname,lo,hi in periods:
       start=next(i for i,r in enumerate(rows) if r["date"]>=lo)
       end=next((i for i,r in enumerate(rows) if r["date"]>=hi),len(rows))
       m,_=backtest(rows,sig,exit_ma,a14,sm,start,end,hold)
       for k,val in m.items(): row[f"{pname}_{k}"]=val
      # Robustness score is deliberately not just the best CAGR.
      cs=[row.get(f"{p}_cagr_pct",-100) for p,_,_ in periods[:3]]
      row["median_segment_cagr_pct"]=round(sorted(cs)[1],2)
      row["worst_segment_cagr_pct"]=round(min(cs),2)
      results.append(row)
    results.sort(key=lambda x:(x["median_segment_cagr_pct"],x["worst_segment_cagr_pct"]),reverse=True)
    report={"symbol":"2303","source":"TWSE official STOCK_DAY raw daily OHLCV",
      "data_first_date":rows[0]["date"],"data_last_date":rows[-1]["date"],"data_rows":len(rows),
      "candidate_count":len(candidates),"initial_cash":INITIAL,"fee_each_side":FEE,
      "sell_tax":TAX,"slippage_each_side":SLIP,
      "execution":"Close-confirmed signals; next-session open fills. Stop levels are based only on prior-session information; gap-through stops fill at open. Whole shares, one position, no leverage, reinvested equity.",
      "entry_families":["Donchian high breakout","Close momentum breakout","MA20 trend pullback and MA10 reclaim"],
      "exit_families":["ATR trailing stop (2/3/4 ATR)","MA10 or MA20 close exit","maximum holding duration (10/20/40/80 sessions)"],
      "limitations":["Raw unadjusted daily OHLCV; dividends, splits, rights issues and other corporate actions are not adjusted.",
      "This is a broad parameter search; top-ranked candidates are exposed to selection bias and are not proof of future performance.",
      "Three chronological segments are robustness checks, not a nested walk-forward test; later validation is still required.",
      "Daily bars cannot model queue priority, partial fills, limit-up/down locks or market impact.",
      "No leverage; single-position whole-share sizing; no short selling."],
      "ranking":"Sorted by median CAGR across the three chronological segments, then worst-segment CAGR; full-period CAGR is reported but not used as sole ranking target.",
      "top_30":results[:30]}
    (OUT/"umc2303_strategy_search_v2.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT/"umc2303_strategy_search_v2.csv").open("w",newline="",encoding="utf-8-sig") as f:
      fields=list(results[0].keys()); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(results)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
