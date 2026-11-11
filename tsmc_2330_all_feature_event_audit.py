"""Cluster-aware paired audit of all technical and foreign/trust pivot features."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

LEADS=(1,3,5,10,20)
BOOTSTRAPS=2000
SEED=23302027

def bh_adjust(pvalues):
    p=np.asarray(pvalues,dtype=float)
    order=np.argsort(p)
    ranked=p[order]*len(p)/(np.arange(len(p))+1)
    ranked=np.minimum.accumulate(ranked[::-1])[::-1]
    q=np.empty_like(ranked);q[order]=np.minimum(ranked,1.0)
    return q

def paired_feature_stats(frame,feature,rng,n_boot):
    a=pd.to_numeric(frame[feature],errors="coerce")
    b=pd.to_numeric(frame["control_"+feature],errors="coerce")
    ok=a.notna()&b.notna()
    z=frame.loc[ok]
    a=a.loc[ok].to_numpy(float);b=b.loc[ok].to_numpy(float)
    n=len(z)
    if n<30:return None
    diff=a-b
    labels=z.control_date.astype(str).to_numpy()
    clusters=np.unique(labels)
    groups=[np.flatnonzero(labels==c) for c in clusters]
    sums=np.array([diff[ix].sum() for ix in groups],dtype=float)
    counts=np.array([len(ix) for ix in groups],dtype=float)
    sampled=rng.integers(0,len(groups),size=(n_boot,len(groups)))
    draws=sums[sampled].sum(axis=1)/counts[sampled].sum(axis=1)
    cm=sums/counts
    pos=int((cm>0).sum());neg=int((cm<0).sum())
    p=float(binomtest(pos,pos+neg,.5).pvalue) if pos+neg else 1.
    pooled=np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2)
    return {"n_pairs":n,"n_control_clusters":len(clusters),
      "event_mean":float(a.mean()),"control_mean":float(b.mean()),
      "mean_difference":float(diff.mean()),"median_difference":float(np.median(diff)),
      "standardized_difference":float(diff.mean()/pooled) if pooled>0 else np.nan,
      "cluster_bootstrap_ci_low":float(np.quantile(draws,.025)),
      "cluster_bootstrap_ci_high":float(np.quantile(draws,.975)),
      "positive_clusters":pos,"negative_clusters":neg,"cluster_sign_p":p}

def run(input_csv,output_dir,n_boot=BOOTSTRAPS):
    d=pd.read_csv(input_csv,parse_dates=["pivot_date","feature_date","control_date"])
    d=d.loc[d.threshold_id.astype(str)=="p0.020_atr1.0"].copy()
    metadata={"event_no","threshold_id","pivot_type","pivot_date","pivot_index","pivot_price",
      "reversal_pct","confirmation_lag","lead_sessions","feature_date","control_date",
      "control_match","control_year","control_trend_stack"}
    features=[c for c in d.columns if c not in metadata and not c.startswith("control_")]
    features=[c for c in features if pd.api.types.is_numeric_dtype(d[c])]
    features=[c for c in features if not c.startswith(("dealer_","foreign_dealer_","broker_","institutional_"))]
    rng=np.random.default_rng(SEED);rows=[];year_rows=[]
    for side in ("HIGH","LOW"):
      for lead in LEADS:
        part=d.loc[(d.pivot_type.astype(str).str.upper()==side)&(d.lead_sessions==lead)].copy()
        for feature in features:
          z=part
          if feature.startswith(("foreign_","trust_")):
            start=pd.Timestamp("2012-05-02")
            z=z.loc[(z.feature_date>=start)&(z.control_date>=start)&(z.pivot_date>=start)]
          stats=paired_feature_stats(z,feature,rng,n_boot)
          if stats is None:continue
          rows.append({"pivot_type":side,"lead_sessions":lead,"feature":feature,**stats})
          av=pd.to_numeric(z[feature],errors="coerce")
          bv=pd.to_numeric(z["control_"+feature],errors="coerce")
          ok=av.notna()&bv.notna()
          tmp=pd.DataFrame({"year":z.loc[ok,"pivot_date"].dt.year.to_numpy(),
            "diff":(av.loc[ok]-bv.loc[ok]).to_numpy()})
          for year,g in tmp.groupby("year"):
            year_rows.append({"pivot_type":side,"lead_sessions":lead,"feature":feature,
              "year":int(year),"n_pairs":len(g),"mean_difference":float(g["diff"].mean())})
    summary=pd.DataFrame(rows)
    if summary.empty:raise ValueError("No complete feature pairs to audit")
    summary["q_bh_all_feature_side_lead_tests"]=bh_adjust(summary.cluster_sign_p.to_numpy())
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    summary.to_csv(out/"tsmc_2330_all_feature_cluster_audit.csv",index=False)
    pd.DataFrame(year_rows).to_csv(out/"tsmc_2330_all_feature_year_stability.csv",index=False)
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument("--snapshots",required=True)
    p.add_argument("--output",required=True);p.add_argument("--bootstraps",type=int,default=BOOTSTRAPS)
    a=p.parse_args();s=run(a.snapshots,a.output,a.bootstraps)
    print(s.sort_values("q_bh_all_feature_side_lead_tests").head(20).to_string(index=False))
if __name__=="__main__":main()
