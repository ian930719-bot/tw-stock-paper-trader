#!/usr/bin/env python3
"""UMC 2303 reproducible swing research; signals at close, fills no earlier than next open."""
from __future__ import annotations
import csv, json, math, os, time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SYMBOL = "2303"
START_YEAR = 2010
END_YEAR = 2026
INITIAL_CASH = 100_000.0
FEE_RATE = 0.001425
SELL_TAX = 0.003
SLIPPAGE = 0.001
OUT = Path("research/umc2303/output")
OUT.mkdir(parents=True, exist_ok=True)

def fetch_month(year: int, month: int):
    date = f"{year}{month:02d}01"
    url = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?" + urlencode(
        {"date": date, "stockNo": SYMBOL, "response": "json"}
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("stat") != "OK":
        return []
    rows = []
    for row in payload.get("data", []):
        try:
            parts = row[0].split("/")
            if len(parts[0]) == 3:
                d = datetime(int(parts[0]) + 1911, int(parts[1]), int(parts[2])).date().isoformat()
            else:
                d = datetime.strptime(row[0].replace("/", ""), "%Y%m%d").date().isoformat()
            nums = [float(x.replace(",", "")) for x in row[1:6]]
            volume, turnover, opn, high, low = nums
            close = float(row[6].replace(",", ""))
            rows.append({"date": d, "volume": volume, "turnover": turnover,
                         "open": opn, "high": high, "low": low, "close": close})
        except (ValueError, IndexError):
            continue
    return rows

def load_data():
    cache = OUT / "umc2303_twse_daily_raw.csv"
    if cache.exists():
        with cache.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Always refresh official monthly data on each workflow run.
        # The previous year-only cache check could leave 2026 data stale for months.
    all_rows = {}
    errors = []
    for y in range(START_YEAR, END_YEAR + 1):
        for m in range(1, 13):
            if (y, m) > (datetime.now().year, datetime.now().month):
                continue
            try:
                for r in fetch_month(y, m):
                    all_rows[r["date"]] = r
                time.sleep(0.12)
            except Exception as e:
                errors.append(f"{y}-{m:02d}: {type(e).__name__}: {e}")
    rows = [all_rows[k] for k in sorted(all_rows)]
    if len(rows) < 500:
        raise RuntimeError(f"Insufficient TWSE data rows ({len(rows)}); errors={errors[:8]}")
    with cache.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (OUT / "data_fetch_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows

def sma(values, n):
    out = [None] * len(values)
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= n: s -= values[i-n]
        if i >= n-1: out[i] = s/n
    return out

def atr(rows, n=14):
    tr = []
    for i,r in enumerate(rows):
        if i == 0: tr.append(r["high"]-r["low"])
        else:
            p=rows[i-1]["close"]
            tr.append(max(r["high"]-r["low"], abs(r["high"]-p), abs(r["low"]-p)))
    out=[None]*len(rows)
    for i in range(n-1,len(rows)):
        out[i]=sum(tr[i-n+1:i+1])/n
    return out

def build_signals(rows):
    c=[r["close"] for r in rows]
    ma5=sma(c,5)
    vol=[r["volume"] for r in rows]
    vol20=sma(vol,20)
    sigs={}
    # Pure MA5 + volume strategies only; no ATR or other indicators in core rules.
    for threshold in (1.0,1.2,1.5,2.0):
        active=False; state=[]
        for i in range(len(rows)):
            if i>=20 and ma5[i] is not None and ma5[i-1] is not None and vol20[i-1] and vol20[i-1]>0:
                # Compare today's volume with the PRIOR 20 completed sessions.
                entry=(c[i]>ma5[i] and c[i-1]<=ma5[i-1] and vol[i]>=vol20[i-1]*threshold)
                if entry: active=True
                elif c[i]<ma5[i]: active=False
            state.append(active)
        sigs[f"ma5_cross_volume_{threshold:.1f}x"]=state
    active=False; state=[]
    for i in range(len(rows)):
        if i>=20 and ma5[i] is not None and ma5[i-1] is not None and vol20[i-1] and vol20[i-1]>0:
            if c[i]>ma5[i] and ma5[i]>ma5[i-1] and vol[i]>=vol20[i-1]*1.5:
                active=True
            elif c[i]<ma5[i]:
                active=False
        state.append(active)
    sigs["ma5_rising_volume_1.5x"]=state
    return sigs, None
def run_backtest(rows, signal, start_idx, end_idx, stop_atr=None):
    cash=INITIAL_CASH; qty=0; entry=0.0; stop=None; highwater=0.0
    equity=[]; trades=[]
    for i in range(start_idx,end_idx):
        r=rows[i]
        # A signal confirmed at yesterday's close can only trade at today's open.
        prior_signal = signal[i-1] if i > 0 else False
        if qty and not prior_signal:
            px=r["open"]*(1-SLIPPAGE); gross=qty*px
            cash+=gross-gross*(FEE_RATE+SELL_TAX)
            net_pnl=(gross-gross*(FEE_RATE+SELL_TAX))-trades[-1]["total_cost"]
            trades[-1].update({"exit_date":r["date"],"exit_price":px,"net_pnl":net_pnl,
                               "net_return_pct":net_pnl/trades[-1]["total_cost"]*100})
            qty=0; entry=0.0; stop=None; highwater=0.0
        if qty==0 and prior_signal:
            px=r["open"]*(1+SLIPPAGE)
            buy_qty=int(cash/(px*(1+FEE_RATE)))
            if buy_qty>0:
                gross=buy_qty*px; fee=gross*FEE_RATE
                cash-=gross+fee; qty=buy_qty; entry=px; highwater=px
                prev_atr=atr_values[i-1] if i>0 and atr_values is not None else None
                stop=px-stop_atr*prev_atr if stop_atr is not None and prev_atr else None
                trades.append({"entry_date":r["date"],"entry_price":px,"qty":qty,"total_cost":gross+fee})
        # Stop known before today's session; gap-through fills at open, otherwise at stop.
        if stop_atr is not None and qty and stop is not None and r["low"]<=stop:
            fill=min(r["open"],stop)*(1-SLIPPAGE)
            gross=qty*fill; cash+=gross-gross*(FEE_RATE+SELL_TAX)
            net_pnl=(gross-gross*(FEE_RATE+SELL_TAX))-trades[-1]["total_cost"]
            trades[-1].update({"exit_date":r["date"],"exit_price":fill,"net_pnl":net_pnl,
                               "net_return_pct":net_pnl/trades[-1]["total_cost"]*100,"stop_exit":True})
            qty=0; entry=0.0; stop=None; highwater=0.0
        elif stop_atr is not None and qty:
            highwater=max(highwater,r["high"])
            a=atr_values[i] if atr_values[i] is not None else entry*0.04
            # Updated at today's close; the new stop is effective from next session.
            stop=max(stop or 0.0,highwater-stop_atr*a)
        equity.append((r["date"],cash+qty*r["close"]))
    if qty:
        r=rows[end_idx-1]; fill=r["close"]*(1-SLIPPAGE); gross=qty*fill
        cash+=gross-gross*(FEE_RATE+SELL_TAX)
        net_pnl=(gross-gross*(FEE_RATE+SELL_TAX))-trades[-1]["total_cost"]
        trades[-1].update({"exit_date":r["date"],"exit_price":fill,"net_pnl":net_pnl,
                           "net_return_pct":net_pnl/trades[-1]["total_cost"]*100,"forced_end":True})
        if equity: equity[-1]=(equity[-1][0],cash)
    return equity,trades,cash

def metrics(equity,trades):
    vals=[x[1] for x in equity]
    if not vals: return {}
    peak=INITIAL_CASH; mdd=0.0
    for v in vals:
        peak=max(peak,v); mdd=min(mdd,v/peak-1)
    first_date=datetime.strptime(equity[0][0],"%Y-%m-%d").date()
    last_date=datetime.strptime(equity[-1][0],"%Y-%m-%d").date()
    years=max((last_date-first_date).days/365.25,1/365.25)
    total=vals[-1]/INITIAL_CASH-1
    cagr=(vals[-1]/INITIAL_CASH)**(1/years)-1 if vals[-1]>0 else -1
    daily=[vals[i]/vals[i-1]-1 for i in range(1,len(vals)) if vals[i-1]>0]
    mu=sum(daily)/len(daily) if daily else 0
    sd=(sum((x-mu)**2 for x in daily)/max(len(daily)-1,1))**0.5 if daily else 0
    wins=[t for t in trades if t.get("net_pnl",0)>0]
    losses=[t for t in trades if t.get("net_pnl",0)<0]
    gross_profit=sum(t["net_pnl"] for t in wins)
    gross_loss=-sum(t["net_pnl"] for t in losses)
    holds=[]
    for t in trades:
        if t.get("exit_date"):
            d0=datetime.strptime(t["entry_date"],"%Y-%m-%d").date()
            d1=datetime.strptime(t["exit_date"],"%Y-%m-%d").date()
            holds.append((d1-d0).days)
    return {"total_return_pct":round(total*100,2),"cagr_pct":round(cagr*100,2),
            "max_drawdown_pct":round(mdd*100,2),"sharpe":round((mu/sd*math.sqrt(252)) if sd else 0,3),
            "trades":len(trades),"win_rate_pct":round(100*len(wins)/len(trades),2) if trades else 0,
            "profit_factor":round(gross_profit/gross_loss,3) if gross_loss else None,
            "avg_win_pct":round(sum(t["net_return_pct"] for t in wins)/len(wins),2) if wins else 0,
            "avg_loss_pct":round(sum(t["net_return_pct"] for t in losses)/len(losses),2) if losses else 0,
            "avg_holding_days":round(sum(holds)/len(holds),2) if holds else 0,
            "ending_equity":round(vals[-1],2)}

def main():
    global atr_values
    rows=load_data()
    rows=[r for r in rows if r["close"]>0 and r["open"]>0 and r["high"]>0 and r["low"]>0]
    signals, atr_values=build_signals(rows)
    # Fixed, predeclared strategies; one continuous out-of-sample window from 2013 onward.
    first=next((i for i,r in enumerate(rows) if r["date"]>="2013-01-01"),None)
    if first is None: raise RuntimeError("No 2013+ observations")
    results=[]
    for name, sig in signals.items():
        eq,tr,_=run_backtest(rows,sig,first,len(rows),stop_atr=None)
        results.append({"strategy":name,**metrics(eq,tr)})
    # Buy-and-hold benchmark uses the same execution timing, sizing, and costs.
    bh_signal=[i>=first for i in range(len(rows))]
    eq,tr,_=run_backtest(rows,bh_signal,first,len(rows),stop_atr=None)
    results.append({"strategy":"buy_and_hold_2013+" ,**metrics(eq,tr)})
    report={"symbol":SYMBOL,"source":"TWSE official STOCK_DAY raw daily OHLCV","source_url":"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY",
            "generated_utc":datetime.utcnow().isoformat()+"Z","first_date":rows[0]["date"],"last_date":rows[-1]["date"],
            "rows":len(rows),"test_start":rows[first]["date"],"initial_cash":INITIAL_CASH,"fee_rate_each_side":FEE_RATE,
            "sell_tax":SELL_TAX,"slippage_each_side":SLIPPAGE,
            "corporate_action_adjustment":"NOT IMPLEMENTED; raw prices only; dividend/split distortions can affect indicators and returns",
            "method":"pure MA5 + volume rules only; entry confirmed at close and executed next open; exit after close below MA5 and execute next open; today's volume is compared with the prior 20 completed sessions' average; one continuous account from 2013; whole-share sizing; estimated fee, sell tax and slippage. Trade win rate and profit factor use net P&L after entry/exit costs. No ATR stop in core results.",
            "limitations":["No official corporate-action adjustment / dividends; do not interpret as total-return performance","No queue, partial fills, limit-up/down or market-impact model","Parameter candidates are fixed, not optimized on this test interval","Daily-bar approximation; close-confirmed exits execute at next session open","Raw prices are not corporate-action adjusted; long-run CAGR and signals remain provisional until adjusted data is integrated"],
            "results":results}
    (OUT/"umc2303_research_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT/"umc2303_strategy_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in results for k in r})); w.writeheader(); w.writerows(results)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
