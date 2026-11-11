from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

INITIAL=1_000_000.0
BUY_FEE,SELL_FEE_TAX,SLIPPAGE=.001425,.004425,.001
REBALANCE_SESSIONS,TOP_N=20,5
SPLITS={"research":("2016-01-04","2019-12-31"),"validation":("2020-01-02","2021-12-31"),"test":("2022-01-03","2026-11-05")}

def load_panel(bars_dir,universe_json,events_csv):
    report=json.loads(Path(universe_json).read_text(encoding="utf-8"))
    universe={int(y):set(map(str,s)) for y,s in report["lagged_universe_years"].items()}
    symbols=sorted(set().union(*universe.values()))
    events=pd.read_csv(events_csv,dtype={"symbol":str})
    events["date"]=pd.to_datetime(events.date,errors="coerce")
    events["adjustment_ratio"]=pd.to_numeric(events.adjustment_ratio,errors="coerce")
    frames=[]
    for symbol in symbols:
        path=Path(bars_dir)/f"{symbol}.csv"
        if not path.exists():continue
        d=pd.read_csv(path,dtype={"symbol":str})
        d["date"]=pd.to_datetime(d.date,errors="coerce")
        for c in ["open","high","low","close","volume","turnover"]:
            if c in d:d[c]=pd.to_numeric(d[c],errors="coerce")
        d=d.dropna(subset=["date","open","high","low","close"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
        ev=events[(events.symbol==symbol)&(~events.conflict.fillna(False))].sort_values("date")
        for e in ev.itertuples(index=False):
            m=d.date<e.date
            for c in ["open","high","low","close"]:d.loc[m,c]*=float(e.adjustment_ratio)
        d["mom20"]=d.close.pct_change(20);d["mom60"]=d.close.pct_change(60);d["mom120"]=d.close.pct_change(120)
        d["ma200"]=d.close.rolling(200,min_periods=200).mean()
        d["eligible"]=d.date.dt.year.map(lambda y:symbol in universe.get(int(y),set())).fillna(False)
        d["symbol"]=symbol
        frames.append(d[["symbol","date","open","high","low","close","volume","turnover","mom20","mom60","mom120","ma200","eligible"]])
    if not frames:raise ValueError("No symbol bars loaded")
    panel=pd.concat(frames,ignore_index=True).sort_values(["date","symbol"]).reset_index(drop=True)
    breadth=panel.loc[panel.eligible&panel.ma200.notna()].groupby("date").apply(lambda g:float((g.close>g.ma200).mean()),include_groups=False)
    panel["breadth200"]=panel.date.map(breadth)
    return panel

def select(g,mode):
    x=g[g.eligible].dropna(subset=["mom20","mom60","mom120"]).copy().reset_index(drop=True)
    if mode in ("mom20_trend","mom20_breadth50_trend"):x=x[x.close>x.ma200]
    if mode in ("mom20_breadth50","mom20_breadth50_trend") and (g.breadth200.dropna().empty or float(g.breadth200.dropna().iloc[0])<.50):return []
    if mode=="mom20_breadth60" and (g.breadth200.dropna().empty or float(g.breadth200.dropna().iloc[0])<.60):return []
    if mode in ("mom20","mom20_trend","mom20_breadth50","mom20_breadth60","mom20_breadth50_trend"):x=x.sort_values("mom20",ascending=False)
    elif mode=="mom60":x=x.sort_values("mom60",ascending=False)
    elif mode=="mom120":x=x.sort_values("mom120",ascending=False)
    elif mode=="composite":
        x["score"]=(x.mom20.rank(pct=True)+x.mom60.rank(pct=True)+x.mom120.rank(pct=True))/3
        x=x.sort_values("score",ascending=False)
    elif mode=="annual_top20":x=x.sort_values("symbol")
    return x.symbol.astype(str).tolist()

def simulate(d,start,end,mode):
    q=d[(d.date>=pd.Timestamp(start))&(d.date<=pd.Timestamp(end))].copy()
    dates=sorted(pd.Timestamp(x) for x in q.date.unique())
    if len(dates)<2:raise ValueError("Insufficient sessions")
    byday={day:g.set_index("symbol",drop=False) for day,g in q.groupby("date",sort=True)}
    cash=INITIAL;holdings={};lastclose={};pending=None;orders=rebals=0;curve=[]
    for i,day in enumerate(dates):
        today=byday[day]
        if pending is not None:
            can_sell=all(s in today.index and pd.notna(today.loc[s,"open"]) and float(today.loc[s,"open"])>0 for s in holdings)
            if can_sell:
                for s,qty in list(holdings.items()):
                    px=float(today.loc[s,"open"])*(1-SLIPPAGE);cash+=qty*px*(1-SELL_FEE_TAX);orders+=1
                holdings={}
                available=[s for s in pending if s in today.index and pd.notna(today.loc[s,"open"]) and float(today.loc[s,"open"])>0]
                if available:
                    alloc=cash/len(available)
                    for s in available:
                        px=float(today.loc[s,"open"])*(1+SLIPPAGE);qty=int(alloc/(px*(1+BUY_FEE)));cost=qty*px*(1+BUY_FEE)
                        if qty>0 and cost<=cash:cash-=cost;holdings[s]=qty;orders+=1
                rebals+=1;pending=None
        mv=0.
        for s,qty in holdings.items():
            if s in today.index and pd.notna(today.loc[s,"close"]):lastclose[s]=float(today.loc[s,"close"])
            mv+=qty*lastclose.get(s,0.)
        equity=cash+mv
        curve.append({"date":day,"equity":equity,"cash":cash,"market_value":mv,"positions":len(holdings),"exposure":mv/equity if equity>0 else 0})
        if i<len(dates)-1:
            year_change=i>0 and day.year!=dates[i-1].year
            trigger=(year_change or i%REBALANCE_SESSIONS==0) if mode!="annual_top20" else (i==0 or year_change)
            if trigger and pending is None:pending=select(today,mode)[:(20 if mode=="annual_top20" else TOP_N)]
    for s,qty in list(holdings.items()):
        px=lastclose.get(s)
        if px and px>0:cash+=qty*px*(1-SLIPPAGE)*(1-SELL_FEE_TAX);orders+=1
    if curve:curve[-1].update(equity=cash,cash=cash,market_value=0.,positions=0,exposure=0.)
    eq=pd.DataFrame(curve);v=eq.equity.to_numpy(float);years=(eq.date.iloc[-1]-eq.date.iloc[0]).days/365.25
    mdd=float(np.min(v/np.maximum.accumulate(np.r_[INITIAL,v])[1:]-1))
    return {"final_equity":float(cash),"total_return_pct":float((cash/INITIAL-1)*100),"cagr_pct":float(((cash/INITIAL)**(1/years)-1)*100),
            "max_drawdown_pct":mdd*100,"orders":orders,"rebalances":rebals,"average_exposure_pct":float(eq.exposure.mean()*100),"sessions":len(eq)},eq

def main(bars_dir,universe_json,events_csv,outdir):
    d=load_panel(bars_dir,universe_json,events_csv)
    modes=["annual_top20","mom20","mom60","mom120","composite","mom20_trend","mom20_breadth50","mom20_breadth60","mom20_breadth50_trend"]
    rows=[];curves={}
    for period,(start,end) in SPLITS.items():
        for mode in modes:
            m,eq=simulate(d,start,end,mode);rows.append({"period":period,"strategy":mode,**m})
            if period=="test":curves[mode]=eq
    table=pd.DataFrame(rows);r=table[table.period=="research"].set_index("strategy")
    chosen=max(modes[1:],key=lambda x:r.loc[x,"cagr_pct"])
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True);table.to_csv(out/"strategy_comparison.csv",index=False)
    for name,eq in curves.items():eq.to_csv(out/f"test_equity_{name}.csv",index=False)
    summary={"version":"V16","symbols":int(d.symbol.nunique()),"rows":len(d),"data_start":str(d.date.min().date()),"data_end":str(d.date.max().date()),
      "source":"FinMind raw OHLCV; OHLC adjusted for available non-conflicting split/par-value ratios only; cash dividend and capital-reduction adjustments incomplete.",
      "universe":"Annual top 20 by prior calendar-year turnover; historical constituent and delisted-price completeness not verified.",
      "execution":"Close signal, next session open; integer shares, cash-only, long-only; top five momentum names rebalanced every 20 sessions; annual universe change forces rebalance.",
      "rules":modes,"costs":{"buy_fee":BUY_FEE,"sell_fee_and_tax":SELL_FEE_TAX,"slippage_each_side":SLIPPAGE},
      "periods":{k:{"start":v[0],"end":v[1]} for k,v in SPLITS.items()},"selected_by_research_only":chosen,"results":rows,
      "limitations":["Point-in-time turnover universe is an approximation, not fully verified historical exchange membership.","Delisted security price coverage is not verified.","Cash dividend and capital-reduction reference-price adjustments are incomplete.","Missing/suspended bars can delay rebalancing; no price-limit queues, partial fills or market impact.","2022-2026 has been inspected in prior work and is not a pristine holdout.","Exploratory historical test, not a live strategy or return guarantee."]}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--bars-dir",default="data/taiwan_history/bars");p.add_argument("--universe-json",default="artifacts/top20/extended_oos_report.json");p.add_argument("--events-csv",default="data/taiwan_history/corporate_actions/events.csv");p.add_argument("--output",default="artifacts/top20_v16_trend_breadth");a=p.parse_args()
    main(a.bars_dir,a.universe_json,a.events_csv,a.output)
