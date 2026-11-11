"""Rebuild 500-pivot controls excluding corporate actions and event windows."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

THRESHOLD_ID="p0.020_atr1.0"
LEADS=(1,3,5,10,20)
SEED=23302026
FEATURES=("return_1","return_3","return_5","return_10","return_20",
 "distance_sma_5","distance_sma_20","distance_sma_50","distance_sma_200",
 "sma20_slope_5","sma50_slope_10","atr_pct14","rsi14","rsi_change_3",
 "macd_pct","macd_hist_pct","macd_hist_change_3","bb_width20","bb_position20",
 "relative_volume20","turnover_ratio20","realized_vol20","realized_vol60",
 "volatility_compression","overnight_gap","intraday_range_pct","close_location",
 "upper_wick_pct","lower_wick_pct","distance_prior_high20","distance_prior_low20",
 "bearish_rsi_divergence20","bullish_rsi_divergence20","bearish_macd_divergence20",
 "bullish_macd_divergence20","bearish_volume_divergence20","bullish_volume_divergence20",
 "stoch_k14","stoch_d3","cci20","williams_r14","adx14","plus_di14","minus_di14",
 "mfi14","chaikin_money_flow20","volume_zscore20","volume_sma5_ratio","volume_sma20_ratio",
 "ppo_hist","donchian_position20","obv_flow_ratio_5","obv_flow_ratio_20",
 "foreign_net_to_volume","foreign_net_5d_lots","foreign_net_20d_lots",
 "foreign_net_positive_streak","foreign_net_negative_streak",
 "trust_net_to_volume","trust_net_5d_lots","trust_net_20d_lots",
 "trust_net_positive_streak","trust_net_negative_streak")
TECHNICAL=("relative_volume20","distance_prior_low20","obv_flow_ratio_5","return_1","close_location",
 "bearish_rsi_divergence20","bearish_macd_divergence20","bearish_volume_divergence20",
 "bullish_rsi_divergence20","bullish_macd_divergence20","bullish_volume_divergence20",
 "rsi14","stoch_k14","distance_sma_20","distance_sma_50","mfi14")

def select_evenly(events,n):
    g=events.sort_values("pivot_index").reset_index(drop=True)
    if len(g)<n: raise ValueError(f"Need {n} pivots, found {len(g)}")
    return g.iloc[np.linspace(0,len(g)-1,n).round().astype(int)].copy()

def build_snapshots(features,events,sample_per_side=250):
    x=features.sort_values("date").reset_index(drop=True).copy()
    x["date"]=pd.to_datetime(x.date).dt.normalize()
    e=events.loc[events.threshold_id.astype(str)==THRESHOLD_ID].copy()
    if e.empty: raise ValueError("No events for frozen threshold")
    e["pivot_index"]=pd.to_numeric(e.pivot_index,errors="raise").astype(int)
    selected=pd.concat([select_evenly(e[e.type.astype(str).str.upper()==k],sample_per_side)
        for k in ("HIGH","LOW")],ignore_index=True).sort_values(["pivot_index","type"]).reset_index(drop=True)
    missing=[c for c in FEATURES if c not in x.columns]
    if missing: raise ValueError(f"Missing features: {missing}")
    dates=x.date; years=dates.dt.year.to_numpy()
    trend=x.trend_stack.fillna(0).astype(int).to_numpy() if "trend_stack" in x else np.zeros(len(x),int)
    action=x.corporate_action_window.fillna(True).astype(bool).to_numpy() if "corporate_action_window" in x else np.zeros(len(x),bool)
    valid=x[list(TECHNICAL)].notna().all(axis=1).to_numpy() & ~action
    pivots=e.pivot_index.to_numpy(dtype=int)
    excluded=np.zeros(len(x),bool)
    for pi in pivots: excluded[max(0,pi-3):min(len(x),pi+4)]=True
    records=[]
    for event_no,r in enumerate(selected.itertuples(index=False),1):
        pi=int(r.pivot_index)
        for lead in LEADS:
            si=pi-lead
            if si>=0 and valid[si]:
                records.append((event_no,r,pi,lead,si,int(years[si]),int(trend[si])))
                excluded[si]=True
    rng=np.random.default_rng(SEED); snapshots=[]
    for event_no,r,pi,lead,si,year,regime in records:
        pools=[
          ((years==year)&(trend==regime),"year_and_feature_date_trend"),
          ((years==year),"year_only"),
          ((np.abs(years-year)<=1)&(trend==regime),"within_1y_trend"),
          ((np.abs(years-year)<=1),"within_1y"),
          ((np.abs(years-year)<=2)&(trend==regime),"within_2y_trend"),
          ((np.abs(years-year)<=2),"within_2y"),
          ((trend==regime),"all_years_trend"),
          (np.ones(len(x),dtype=bool),"all_years")]
        chosen=None; match=None
        for mask,label in pools:
            cand=np.flatnonzero(mask&~excluded&valid&(np.arange(len(x))!=si))
            if len(cand)>=5: chosen=cand;match=label;break
        if chosen is None: raise ValueError(f"No eligible control pool for event {event_no}, lead {lead}")
        ci=int(rng.choice(chosen))
        row={"event_no":event_no,"threshold_id":THRESHOLD_ID,"pivot_type":str(r.type).upper(),
          "pivot_date":pd.Timestamp(r.pivot_date).date().isoformat(),"pivot_index":pi,
          "pivot_price":float(r.pivot_price),"reversal_pct":float(r.reversal_pct),
          "confirmation_lag":int(r.confirmation_lag) if hasattr(r,"confirmation_lag") else None,
          "lead_sessions":lead,"feature_date":dates.iloc[si].date().isoformat(),
          "control_date":dates.iloc[ci].date().isoformat(),"control_match":match,
          "control_year":int(years[ci]),"control_trend_stack":int(trend[ci])}
        for f in FEATURES:
            a=x.at[si,f];b=x.at[ci,f]
            row[f]=float(a) if pd.notna(a) else np.nan
            row["control_"+f]=float(b) if pd.notna(b) else np.nan
        snapshots.append(row)
    snap=pd.DataFrame(snapshots)
    lookup=dict(zip(dates,action))
    if any(lookup.get(pd.Timestamp(v),True) for v in snap.feature_date):
        raise ValueError("Corporate-action event feature date in sample")
    if any(lookup.get(pd.Timestamp(v),True) for v in snap.control_date):
        raise ValueError("Corporate-action control date in sample")
    date_to_index={v:i for i,v in enumerate(dates)}
    cidx=[date_to_index[pd.Timestamp(v)] for v in snap.control_date]
    if any(excluded[i] for i in cidx):
        raise ValueError("Control overlaps pivot neighborhood or sampled event feature date")
    return selected,snap

def run(features_path,events_path,output_dir,sample_per_side=250):
    x=pd.read_csv(features_path,parse_dates=["date"]);e=pd.read_csv(events_path)
    selected,snap=build_snapshots(x,e,sample_per_side)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    selected.to_csv(out/"tsmc_2330_500_selected_pivots.csv",index=False)
    snap.to_csv(out/"tsmc_2330_500_pivot_snapshots.csv",index=False)
    result={"symbol":"2330","threshold_id":THRESHOLD_ID,"selected_pivots":len(selected),
      "selected_high":int((selected.type.astype(str).str.upper()=="HIGH").sum()),
      "selected_low":int((selected.type.astype(str).str.upper()=="LOW").sum()),
      "snapshot_rows":len(snap),"lead_sessions":list(LEADS),
      "control_match_distribution":snap.control_match.value_counts().to_dict(),
      "unique_control_dates":int(snap.control_date.nunique()),
      "max_control_date_reuse":int(snap.control_date.value_counts().max()),
      "corporate_action_rows_excluded":True,"pivot_exclusion_sessions":3,
      "event_feature_dates_excluded_from_controls":True,
      "institutional_data_start":"2012-05-02; downstream flow-rule audit uses complete pairs only",
      "inference":"Cluster bootstrap and cluster sign test by control_date; BH correction across all rule x lead tests",
      "interpretation":"Exploratory event/control study only; no P&L or CAGR inference."}
    (out/"tsmc_2330_500_pivot_study_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--events",required=True)
    p.add_argument("--output",required=True);p.add_argument("--sample-per-side",type=int,default=250)
    a=p.parse_args();print(json.dumps(run(a.features,a.events,a.output,a.sample_per_side),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
