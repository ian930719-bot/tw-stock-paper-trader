"""Year-cluster robustness audit for the 500-pivot TSMC event study.
Exploratory only: no trading returns or predictive-performance claims.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

META={"event_no","threshold_id","pivot_type","pivot_date","pivot_index","pivot_price",
      "reversal_pct","confirmation_lag","lead_sessions","feature_date","control_date",
      "control_match","control_year","control_trend_stack"}

def _bh(pvalues):
    p=np.asarray(pvalues,float); n=len(p); order=np.argsort(p); out=np.full(n,np.nan)
    ranked=p[order]*n/(np.arange(n)+1)
    ranked=np.minimum.accumulate(ranked[::-1])[::-1]
    out[order]=np.minimum(ranked,1.0)
    return out

def analyze(input_csv, output_dir, bootstrap=2000, seed=23302027):
    d=pd.read_csv(input_csv,parse_dates=["pivot_date"])
    if not {"pivot_type","lead_sessions","pivot_date","event_no"}.issubset(d):
        raise ValueError("Input is not a 500-pivot snapshot file")
    features=[c for c in d.columns if c not in META and c != "trend_stack"
              and not c.startswith("control_") and "control_"+c in d.columns]
    if not features: raise ValueError("No paired event/control feature columns found")
    d["pivot_year"]=d.pivot_date.dt.year
    rng=np.random.default_rng(seed)
    rows=[]
    for (kind,lead),g in d.groupby(["pivot_type","lead_sessions"]):
        for f in features:
            paired=g[[f,"control_"+f,"pivot_year"]].copy()
            paired[f]=pd.to_numeric(paired[f],errors="coerce")
            paired["control_"+f]=pd.to_numeric(paired["control_"+f],errors="coerce")
            paired=paired.replace([np.inf,-np.inf],np.nan).dropna()
            # Institutional flow history is unavailable before 2012-05-02 in this dataset.
            # Exclude event, feature, and control observations before that date.
            if f.startswith(("foreign_", "trust_")):
                event_dates=pd.to_datetime(g.pivot_date)
                feature_dates=pd.to_datetime(g.feature_date)
                control_dates=pd.to_datetime(g.control_date)
                eligible=(event_dates >= pd.Timestamp("2012-05-02")) & (feature_dates >= pd.Timestamp("2012-05-02")) & (control_dates >= pd.Timestamp("2012-05-02"))
                paired=paired.loc[eligible.loc[paired.index]]
            paired["diff"]=paired[f]-paired["control_"+f]
            annual=paired.groupby("pivot_year")["diff"].agg(["mean","count"])
            annual=annual[annual["count"]>=5]
            if len(annual)<8: continue
            vals=annual["mean"].to_numpy(float)
            draws=rng.choice(vals,size=(bootstrap,len(vals)),replace=True).mean(axis=1)
            lo,hi=np.quantile(draws,[.025,.975])
            rows.append({"pivot_type":kind,"lead_sessions":int(lead),"feature":f,
              "paired_event_n":int(len(paired)),"years_n":int(len(vals)),
              "annual_positive_share":float(np.mean(vals>0)),
              "annual_negative_share":float(np.mean(vals<0)),
              "median_annual_mean_difference":float(np.median(vals)),
              "mean_annual_mean_difference":float(np.mean(vals)),
              "cluster_bootstrap_ci_low":float(lo),"cluster_bootstrap_ci_high":float(hi),
              "cluster_bootstrap_excludes_zero":bool(lo>0 or hi<0),
              "median_absolute_annual_difference":float(np.median(np.abs(vals)))})
    detail=pd.DataFrame(rows)
    if detail.empty: raise ValueError("No feature had enough year-clustered observations")
    summary=[]
    for (kind,f),g in detail.groupby(["pivot_type","feature"]):
        g=g.sort_values("lead_sessions")
        signs=np.sign(g.median_annual_mean_difference.to_numpy())
        dominant=max(np.mean(signs>0),np.mean(signs<0)) if len(signs) else 0
        ci_hits=int(g.cluster_bootstrap_excludes_zero.sum())
        summary.append({"pivot_type":kind,"feature":f,"lead_count":int(len(g)),
          "dominant_direction_share":float(dominant),
          "median_absolute_annual_difference":float(g.median_absolute_annual_difference.median()),
          "median_annual_mean_difference":float(g.median_annual_mean_difference.median()),
          "cluster_ci_excludes_zero_leads":ci_hits,
          "stable_candidate":bool(len(g)>=4 and dominant>=.75 and ci_hits>=2)})
    ranking=pd.DataFrame(summary).sort_values(
        ["stable_candidate","cluster_ci_excludes_zero_leads","dominant_direction_share",
         "median_absolute_annual_difference"],ascending=[False,False,False,False])
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    detail.to_csv(out/"tsmc_2330_500_year_cluster_feature_audit.csv",index=False)
    ranking.to_csv(out/"tsmc_2330_500_stable_feature_candidates.csv",index=False)
    result={"input_rows":int(len(d)),"features_audited":int(len(features)),
      "pivot_types":sorted(d.pivot_type.unique().tolist()),
      "lead_sessions":sorted(map(int,d.lead_sessions.unique())),
      "bootstrap_replicates":int(bootstrap),"minimum_events_per_year":5,"institutional_feature_start_date":"2012-05-02","matching_feature_excluded":["trend_stack"],
      "candidate_rule":"At least four lead horizons, same dominant direction in >=75% of horizons, and year-cluster bootstrap CI excludes zero in >=2 horizons.",
      "warning":"Exploratory feature stability screen only. Trend_stack is excluded because it was used for control matching. Foreign/trust features are restricted to event, feature, and control dates on/after 2012-05-02 because earlier institutional coverage is unavailable. Year-cluster bootstrap uses annual mean paired differences; overlapping events, repeated market regimes, feature multiplicity, and retrospective pivot labels remain. Candidate rules require untouched chronological validation and executable cost-aware backtests.",
      "stable_candidates":{k:int(v) for k,v in ranking[ranking.stable_candidate].groupby("pivot_type").size().items()},
      "outputs":["tsmc_2330_500_year_cluster_feature_audit.csv","tsmc_2330_500_stable_feature_candidates.csv"]}
    (out/"tsmc_2330_500_year_cluster_audit_summary.json").write_text(
      json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True); p.add_argument("--output",required=True)
    p.add_argument("--bootstrap",type=int,default=2000)
    a=p.parse_args()
    print(json.dumps(analyze(a.input,a.output,a.bootstrap),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
