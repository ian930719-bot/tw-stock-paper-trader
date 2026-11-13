#!/usr/bin/env python3
"""MA5 + volume exit/re-entry experiments for UMC 2303.

Signals are evaluated at the close and filled at the next session open.
Only MA5 and volume are used. No look-ahead.
"""
import csv, json, math
from datetime import date
from pathlib import Path

ROOT = Path("research/umc2303/output")
SRC = ROOT / "umc2303_twse_daily_raw.csv"
INITIAL = 100000.0
FEE = 0.001425
TAX = 0.003
SLIP = 0.001
START = "2013-01-01"

def sma(xs, n):
    out=[None]*len(xs)
    for i in range(n-1,len(xs)):
        out[i]=sum(xs[i-n+1:i+1])/n
    return out

def metrics(eq, trades):
    vals=[v for _,v in eq]
    peak=INITIAL
    mdd=0.0
    for v in vals:
        peak=max(peak,v)
        mdd=min(mdd,v/peak-1)
    days=(date.fromisoformat(eq[-1][0])-date.fromisoformat(eq[0][0])).days
    years=max(days/365.25,1/365.25)
    cagr=(vals[-1]/INITIAL)**(1/years)-1 if vals[-1]>0 else -1
    rets=[vals[i]/vals[i-1]-1 for i in range(1,len(vals)) if vals[i-1]>0]
    mu=sum(rets)/len(rets) if rets else 0
    sd=(sum((x-mu)**2 for x in rets)/max(1,len(rets)-1))**0.5 if rets else 0
    wins=[t for t in trades if t["net_pnl"]>0]
    losses=[t for t in trades if t["net_pnl"]<0]
    gp=sum(t["net_pnl"] for t in wins)
    gl=-sum(t["net_pnl"] for t in losses)
    false=[t for t in trades if t["holding_sessions"]<=3]
    return {
        "total_return_pct":round((vals[-1]/INITIAL-1)*100,2),
        "cagr_pct":round(cagr*100,2),
        "max_drawdown_pct":round(mdd*100,2),
        "sharpe":round(mu/sd*math.sqrt(252),3) if sd else 0,
        "trades":len(trades),
        "net_win_rate_pct":round(100*len(wins)/len(trades),2) if trades else 0,
        "profit_factor":round(gp/gl,3) if gl else None,
        "false_signal_3session_pct":round(100*len(false)/len(trades),2) if trades else 0,
        "avg_holding_sessions":round(sum(t["holding_sessions"] for t in trades)/len(trades),2) if trades else 0,
        "ending_equity":round(vals[-1],2)
    }

def backtest(rows, ma5, vol20, start_i, threshold, exit_bars):
    cash=INITIAL
    qty=0
    pending=None
    below=0
    entry_cost=0
    entry_i=None
    trades=[]
    equity=[]
    for i in range(start_i,len(rows)):
        r=rows[i]
        # Apply decision made at the preceding close, at today's open.
        if pending=="exit" and qty:
            px=r["open"]*(1-SLIP)
            gross=qty*px
            proceeds=gross*(1-FEE-TAX)
            pnl=proceeds-entry_cost
            trades.append({
                "entry_date":rows[entry_i]["date"],"exit_date":r["date"],
                "entry_index":entry_i,"exit_index":i,"net_pnl":pnl,
                "net_return_pct":pnl/entry_cost*100,
                "holding_sessions":i-entry_i
            })
            cash+=proceeds
            qty=0
            entry_cost=0
            entry_i=None
            below=0
        elif pending=="entry" and qty==0:
            px=r["open"]*(1+SLIP)
            n=int(cash/(px*(1+FEE)))
            if n>0:
                gross=n*px
                fee=gross*FEE
                cash-=gross+fee
                qty=n
                entry_cost=gross+fee
                entry_i=i
                below=0
        pending=None

        equity.append((r["date"],cash+qty*r["close"]))

        # Today's close determines only the next session's order.
        if i+1>=len(rows):
            continue
        if qty:
            below = below+1 if r["close"] < ma5[i] else 0
            if below>=exit_bars:
                pending="exit"
                below=0
        else:
            if (i>0 and r["close"]>ma5[i] and rows[i-1]["close"]<=ma5[i-1]
                and r["volume"]>=vol20[i-1]*threshold):
                pending="entry"

    if qty:
        r=rows[-1]
        px=r["close"]*(1-SLIP)
        gross=qty*px
        proceeds=gross*(1-FEE-TAX)
        pnl=proceeds-entry_cost
        trades.append({
            "entry_date":rows[entry_i]["date"],"exit_date":r["date"],
            "entry_index":entry_i,"exit_index":len(rows)-1,"net_pnl":pnl,
            "net_return_pct":pnl/entry_cost*100,
            "holding_sessions":len(rows)-1-entry_i,
            "forced_end":True
        })
        cash+=proceeds
        equity[-1]=(r["date"],cash)
    return metrics(equity,trades),trades

def main():
    if not SRC.exists():
        raise RuntimeError("Baseline raw TWSE CSV missing; run run_research.py first")
    with SRC.open(encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for k in ("volume","open","high","low","close"):
            r[k]=float(r[k])
    rows=[r for r in rows if r["close"]>0 and r["open"]>0 and r["volume"]>0]
    closes=[r["close"] for r in rows]
    vols=[r["volume"] for r in rows]
    ma5=sma(closes,5)
    vol20=sma(vols,20)
    first=next(i for i,r in enumerate(rows) if r["date"]>=START)
    results=[]
    trade_rows=[]
    for exit_bars in (1,2):
        for threshold in (1.0,1.5,2.0):
            m,tr=backtest(rows,ma5,vol20,first,threshold,exit_bars)
            name=f"ma5_volume_{threshold:.1f}x_exit_{exit_bars}close"
            results.append({"strategy":name,"exit_confirmation_closes":exit_bars,
                            "volume_threshold_prior20x":threshold,**m})
            for t in tr:
                trade_rows.append({"strategy":name,**{k:v for k,v in t.items() if k not in ("entry_index","exit_index")}})
    report={
        "symbol":"2303",
        "source":"TWSE official STOCK_DAY raw daily OHLCV CSV from baseline run",
        "data_first_date":rows[0]["date"],"data_last_date":rows[-1]["date"],
        "data_rows":len(rows),"test_start":rows[first]["date"],
        "method":"MA5 close crossover plus current volume versus prior 20 completed sessions; exit after either one or two consecutive closes below MA5; decisions at close, fills next session open; after exit, re-entry only on a fresh close crossover above MA5 with volume threshold; whole-share sizing; commission 0.1425% each side, sell tax 0.3%, slippage 0.1% each side.",
        "false_signal_definition":"A completed trade exited within 3 trading sessions of its entry fill. This is a short-lived-trade proxy, not a claim that every such trade is objectively false.",
        "limitations":["Raw, unadjusted prices: cash dividends, stock dividends and splits are not adjusted.",
                       "No queue, partial-fill, limit-up/down or market-impact simulation.",
                       "Fixed rule variants are exploratory and not an untouched holdout.",
                       "No corporate-action adjustment means long-run results remain provisional."],
        "results":results
    }
    (ROOT/"umc2303_exit_reentry_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with (ROOT/"umc2303_exit_reentry_trades.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=["strategy","entry_date","exit_date","net_pnl","net_return_pct","holding_sessions","forced_end"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for t in trade_rows: w.writerow({k:t.get(k,"") for k in fields})
    with (ROOT/"umc2303_exit_reentry_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=list(results[0].keys())
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(results)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
