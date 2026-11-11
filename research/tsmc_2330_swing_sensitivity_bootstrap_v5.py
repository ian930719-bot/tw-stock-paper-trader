"""TSMC 2330 ZigZag threshold sensitivity and matched-event bootstrap."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

THRESHOLDS = (0.06, 0.08, 0.10, 0.12, 0.15)
FOCUS = {("low",5):("rsi","bbpct","dd20"),("low",1):("rsi","ret10"),
         ("high",1):("rsi","ma60dev","ret40","dd40"),("high",5):("rsi","ma60dev")}

def features(path):
    d=pd.read_csv(path,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    for c in ("open","high","low","close","volume"): d[c]=pd.to_numeric(d[c],errors="coerce")
    c,h,l,v=d.close,d.high,d.low,d.volume
    ch=c.diff(); up=ch.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); dn=(-ch.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    d["rsi"]=100-100/(1+up/dn.replace(0,np.nan))
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); d["atrp"]=tr.rolling(14).mean()/c
    for n in (5,10,20,40,60):
        d[f"ret{n}"]=c.pct_change(n); d[f"ma{n}dev"]=c/c.rolling(n).mean()-1
    d["dd20"]=c/h.rolling(20).max()-1; d["dd40"]=c/h.rolling(40).max()-1
    d["bbpct"]=(c-c.rolling(20).mean())/(2*c.rolling(20).std())
    return d

def pivots(d, rev):
    hi,lo,cl=d.high.to_numpy(float),d.low.to_numpy(float),d.close.to_numpy(float)
    state,ext,ix,out="high",float(hi[0]),0,[]
    for i in range(1,len(d)):
        if state=="high":
            if hi[i]>ext: ext,ix=hi[i],i
            if cl[i]<=ext*(1-rev):
                out.append((ix,"high")); state,ext,ix="low",float(lo[i]),i
        else:
            if lo[i]<ext: ext,ix=lo[i],i
            if cl[i]>=ext*(1+rev):
                out.append((ix,"low")); state,ext,ix="high",float(hi[i]),i
    return [(i,t) for i,t in out if i>=65]

def matched(d, ps):
    mask=np.ones(len(d),dtype=bool)
    for i,_ in ps: mask[max(0,i-35):min(len(d),i+36)]=False
    available=np.flatnonzero(mask); used=set(); out=[]
    for i,typ in ps:
        year=d.date.iloc[i].year
        cand=available[d.date.iloc[available].dt.year.to_numpy()==year]
        cand=np.array([j for j in cand if j not in used and pd.notna(d.loc[j,"ma60dev"]) and pd.notna(d.loc[j,"atrp"])])
        if not len(cand): continue
        dist=((d.loc[cand,"ma60dev"].to_numpy()-d.loc[i,"ma60dev"])/.08)**2+((d.loc[cand,"atrp"].to_numpy()-d.loc[i,"atrp"])/.01)**2
        j=int(cand[np.argmin(dist)]); used.add(j); out.append((i,j,typ))
    return out

def ci_median(x,rng,n):
    x=np.asarray(x,float)
    if not len(x): return (None,None)
    sample=rng.choice(x,size=(n,len(x)),replace=True)
    med=np.median(sample,axis=1)
    return float(np.quantile(med,.025)),float(np.quantile(med,.975))

def run(source,outdir,nboot=5000):
    d=features(source); rng=np.random.default_rng(23302026); counts=[]; rows=[]
    for rev in THRESHOLDS:
        ps=pivots(d,rev); pairs=matched(d,ps)
        counts.append({"reversal_pct":rev*100,"low_pivots":sum(t=="low" for _,t in ps),
          "high_pivots":sum(t=="high" for _,t in ps),"matched_low":sum(t=="low" for _,_,t in pairs),
          "matched_high":sum(t=="high" for _,_,t in pairs)})
        for (typ,lag),fs in FOCUS.items():
            for f in fs:
                dif=[]; ev=[]; co=[]
                for i,j,t in pairs:
                    if t!=typ or i<lag or j<lag: continue
                    a,b=d.loc[i-lag,f],d.loc[j-lag,f]
                    if pd.notna(a) and pd.notna(b): ev.append(float(a)); co.append(float(b)); dif.append(float(a-b))
                lo,up=ci_median(dif,rng,nboot)
                rows.append({"reversal_pct":rev*100,"pivot":typ,"lag":lag,"feature":f,"pairs":len(dif),
                  "event_median":float(np.median(ev)) if ev else None,"control_median":float(np.median(co)) if co else None,
                  "median_paired_diff":float(np.median(dif)) if dif else None,"bootstrap_95ci_low":lo,
                  "bootstrap_95ci_high":up,"bootstrap_resamples":nboot})
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(counts).to_csv(out/"zigzag_threshold_sensitivity.csv",index=False)
    pd.DataFrame(rows).to_csv(out/"event_paired_bootstrap.csv",index=False)
    summary={"symbol":"2330","rows":len(d),"start":str(d.date.min().date()),"end":str(d.date.max().date()),
      "source":"Yahoo Finance auto_adjust=True","threshold_sensitivity":counts,
      "bootstrap":"95% percentile CI, resampling matched event-control pairs with replacement; seed=23302026",
      "limitations":["ZigZag pivots are future-confirmed retrospective labels, not tradable signals.",
      "Unique same-year controls matched on MA60 deviation and ATR%; small event samples.",
      "Different thresholds change the event population; threshold results are not independent.",
      "Bootstrap quantifies event-pair sampling uncertainty only, not regime/data-source uncertainty.",
      "Adjusted Yahoo OHLCV is not official execution-grade exchange data."]}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input",default="data/top10_adjusted_prices/bars/2330.csv")
    p.add_argument("--output",default="artifacts/tsmc_swing_v5"); p.add_argument("--bootstrap",type=int,default=5000)
    a=p.parse_args(); print(json.dumps(run(a.input,a.output,a.bootstrap),ensure_ascii=False,indent=2))
