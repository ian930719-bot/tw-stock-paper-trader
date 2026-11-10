from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

INITIAL=1_000_000.0
BUY_FEE=0.001425
SELL_FEE_TAX=0.004425
SLIPPAGE=0.001
REBALANCE_SESSIONS=20
TOP_N=5
SPLITS={
    "research":("2016-01-04","2019-12-31"),
    "validation":("2020-01-02","2021-12-31"),
    "test":("2022-01-03","2026-11-05"),
}

def load_full_panel(bars_dir, universe_json, events_csv):
    report=json.loads(Path(universe_json).read_text(encoding="utf-8"))
    universe={int(y):set(map(str,syms)) for y,syms in report["lagged_universe_years"].items()}
    symbols=sorted(set().union(*universe.values()))
    events=pd.read_csv(events_csv,dtype={"symbol":str})
    events["date"]=pd.to_datetime(events.date,errors="coerce")
    events["adjustment_ratio"]=pd.to_numeric(events.adjustment_ratio,errors="coerce")
    frames=[]
    for symbol in symbols:
        p=Path(bars_dir)/f"{symbol}.csv"
        if not p.exists(): continue
        d=pd.read_csv(p,dtype={"symbol":str})
        d["date"]=pd.to_datetime(d.date,errors="coerce")
        for c in ["open","high","low","close","volume","turnover"]:
            if c in d: d[c]=pd.to_numeric(d[c],errors="coerce")
        d=d.dropna(subset=["date","open","high","low","close"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
        ev=events[(events.symbol==symbol)&(~events.conflict.fillna(False))].sort_values("date")
        for e in ev.itertuples(index=False):
            mask=d.date<e.date
            for c in ["open","high","low","close"]:
                d.loc[mask,c]*=float(e.adjustment_ratio)
        d["mom20"]=d.close.pct_change(20)
        d["mom60"]=d.close.pct_change(60)
        d["mom120"]=d.close.pct_change(120)
        d["eligible"]=d.date.dt.year.map(lambda y: symbol in universe.get(int(y),set())).fillna(False)
        d["symbol"]=symbol
        frames.append(d[["symbol","date","open","high","low","close","volume","turnover","mom20","mom60","mom120","eligible"]])
    if not frames: raise ValueError("No historical symbol bars loaded")
    return pd.concat(frames,ignore_index=True).sort_values(["date","symbol"]).reset_index(drop=True)

def select(g,mode):
    x=g[g.eligible].dropna(subset=["mom20","mom60","mom120"]).copy().reset_index(drop=True)
    if x.empty:return []
    if mode=="mom20":x=x.sort_values("mom20",ascending=False)
    elif mode=="mom60":x=x.sort_values("mom60",ascending=False)
    elif mode=="mom120":x=x.sort_values("mom120",ascending=False)
    elif mode=="composite":
        x["score"]=(x.mom20.rank(pct=True)+x.mom60.rank(pct=True)+x.mom120.rank(pct=True))/3
        x=x.sort_values("score",ascending=False)
    elif mode=="annual_top20":x=x.sort_values("symbol")
    return x.symbol.astype(str).tolist()

def simulate(d,start,end,mode):
    a,b=pd.Timestamp(start),pd.Timestamp(end)
    q=d[(d.date>=a)&(d.date<=b)].copy()
    dates=sorted(pd.Timestamp(x) for x in q.date.unique())
    if len(dates)<2:raise ValueError(f"Not enough sessions for {start}..{end}")
    byday={day:g.set_index("symbol",drop=False) for day,g in q.groupby("date",sort=True)}
    cash=INITIAL; holdings={}; lastclose={}; pending=None; orders=0; rebals=0; curve=[]
    for i,day in enumerate(dates):
        today=byday[day]
        if pending is not None:
            can_sell=all(s in today.index and pd.notna(today.loc[s,"open"]) and float(today.loc[s,"open"])>0 for s in holdings)
            if can_sell:
                for s,qty in list(holdings.items()):
                    px=float(today.loc[s,"open"])*(1-SLIPPAGE)
                    cash+=qty*px*(1-SELL_FEE_TAX);orders+=1
                holdings={}
                available=[s for s in pending if s in today.index and pd.notna(today.loc[s,"open"]) and float(today.loc[s,"open"])>0]
                if available:
                    alloc=cash/len(available)
                    for s in available:
                        px=float(today.loc[s,"open"])*(1+SLIPPAGE)
                        qty=int(alloc/(px*(1+BUY_FEE)))
                        cost=qty*px*(1+BUY_FEE)
                        if qty>0 and cost<=cash:
                            cash-=cost;holdings[s]=qty;orders+=1
                rebals+=1
                pending=None
        mv=0.
        for s,qty in holdings.items():
            if s in today.index and pd.notna(today.loc[s,"close"]):
                lastclose[s]=float(today.loc[s,"close"])
            mv+=qty*lastclose.get(s,0.)
        equity=cash+mv
        curve.append({"date":day,"equity":equity,"cash":cash,"market_value":mv,"positions":len(holdings),"exposure":mv/equity if equity>0 else 0})
        if i<len(dates)-1:
            year_change=i>0 and day.year!=dates[i-1].year
            trigger=(year_change or i%REBALANCE_SESSIONS==0) if mode!="annual_top20" else (i==0 or year_change)
            if trigger and pending is None:
                pending=select(today,mode)[:(20 if mode=="annual_top20" else TOP_N)]
    for s,qty in list(holdings.items()):
        px=lastclose.get(s)
        if px and px>0:
            cash+=qty*px*(1-SLIPPAGE)*(1-SELL_FEE_TAX);orders+=1
    if curve:curve[-1].update(equity=cash,cash=cash,market_value=0.,positions=0,exposure=0.)
    eq=pd.DataFrame(curve);v=eq.equity.to_numpy(float)
    years=(eq.date.iloc[-1]-eq.date.iloc[0]).days/365.25
    mdd=float(np.min(v/np.maximum.accumulate(np.r_[INITIAL,v])[1:]-1))
    return {"final_equity":float(cash),"total_return_pct":float((cash/INITIAL-1)*100),
            "cagr_pct":float(((cash/INITIAL)**(1/years)-1)*100),"max_drawdown_pct":mdd*100,
            "orders":orders,"rebalances":rebals,"average_exposure_pct":float(eq.exposure.mean()*100),
            "sessions":len(eq)},eq

def main(bars_dir,universe_json,events_csv,outdir):
    d=load_full_panel(bars_dir,universe_json,events_csv)
    rows=[];test_curves={}
    for period,(start,end) in SPLITS.items():
        for mode in ["annual_top20","mom20","mom60","mom120","composite"]:
            m,eq=simulate(d,start,end,mode)
            rows.append({"period":period,"strategy":mode,**m})
            if period=="test":test_curves[mode]=eq
    table=pd.DataFrame(rows)
    r=table[table.period=="research"].set_index("strategy")
    selected=max(["mom20","mom60","mom120","composite"],key=lambda x:r.loc[x,"cagr_pct"])
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/"strategy_comparison.csv",index=False)
    for name,eq in test_curves.items():eq.to_csv(out/f"test_equity_{name}.csv",index=False)
    summary={"version":"V15","symbols":int(d.symbol.nunique()),"rows":len(d),
      "data_start":str(d.date.min().date()),"data_end":str(d.date.max().date()),
      "source":"FinMind TaiwanStockPrice raw OHLCV; historical OHLC adjusted only for available non-conflicting split/par-value ratios. Cash dividends and capital-reduction reference prices are not fully adjusted.",
      "universe":"Each calendar year uses prior calendar year's top 20 by turnover; universe JSON supplied from historical turnover ranking.",
      "execution":"Signal at close; next session open execution; integer shares; cash-only; long-only; whole portfolio rebalanced into top five names every 20 sessions; annual universe change forces rebalance.",
      "strategies":["annual top-20 equal-weight hold with annual rebalance","top five by 20-session momentum","top five by 60-session momentum","top five by 120-session momentum","top five by average cross-sectional percentile rank of 20/60/120-session momentum"],
      "costs":{"buy_fee":BUY_FEE,"sell_fee_and_tax":SELL_FEE_TAX,"slippage_each_side":SLIPPAGE},
      "periods":{k:{"start":v[0],"end":v[1]} for k,v in SPLITS.items()},
      "selected_by_research_only":selected,"results":rows,
      "limitations":["The lagged turnover universe is an approximation, not a fully verified point-in-time exchange constituent history.","Delisted security price coverage is not verified; the candidate pool includes current and delisting candidates but may be incomplete.","Corporate-action data is incomplete: cash dividend adjustment and capital-reduction reference-price coverage are not fully available.","Missing/suspended bars can delay a rebalance until all existing holdings have executable opens; no queue, limit-price or market-impact simulation.","The 2022-2026 period has been inspected in previous project research and is not a pristine holdout.","This is an exploratory cross-sectional diagnostic, not a live strategy or a claim of 200% annualized return."]}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--bars-dir",default="data/taiwan_history/bars")
    p.add_argument("--universe-json",default="artifacts/top20/extended_oos_report.json")
    p.add_argument("--events-csv",default="data/taiwan_history/corporate_actions/events.csv")
    p.add_argument("--output",default="artifacts/top20_v15_cross_sectional")
    a=p.parse_args();main(a.bars_dir,a.universe_json,a.events_csv,a.output)
