"""Fast, reproducible 500-pivot feature study for TSMC 2330.
Exploratory event study only; not a trading backtest or an OOS claim.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

THRESHOLD_ID = "p0.020_atr1.0"
LEADS = (1, 3, 5, 10, 20)
SEED = 23302026
FEATURES = (
    "return_1","return_3","return_5","return_10","return_20",
    "distance_sma_5","distance_sma_20","distance_sma_50","distance_sma_200",
    "sma20_slope_5","sma50_slope_10","trend_stack","atr_pct14","rsi14",
    "rsi_change_3","macd_pct","macd_hist_pct","macd_hist_change_3",
    "bb_width20","bb_position20","relative_volume20","turnover_ratio20",
    "realized_vol20","realized_vol60","volatility_compression","overnight_gap",
    "intraday_range_pct","close_location","upper_wick_pct","lower_wick_pct",
    "distance_prior_high20","distance_prior_low20","bearish_rsi_divergence20",
    "bullish_rsi_divergence20","bearish_macd_divergence20","bullish_macd_divergence20",
    "bearish_volume_divergence20","bullish_volume_divergence20","stoch_k14","stoch_d3",
    "cci20","williams_r14","adx14","plus_di14","minus_di14","mfi14",
    "chaikin_money_flow20","volume_zscore20","volume_sma5_ratio","volume_sma20_ratio",
    "ppo_hist","donchian_position20","obv_flow_ratio_5","obv_flow_ratio_20",
    "foreign_net_to_volume","foreign_net_5d_lots","foreign_net_20d_lots",
    "foreign_net_positive_streak","foreign_net_negative_streak",
    "trust_net_to_volume","trust_net_5d_lots","trust_net_20d_lots",
    "trust_net_positive_streak","trust_net_negative_streak",
)

def _stratified_select(g: pd.DataFrame, n: int) -> pd.DataFrame:
    g=g.sort_values("pivot_index").reset_index(drop=True)
    if len(g)<n: raise ValueError(f"Need {n} {g.type.iloc[0]} pivots, only have {len(g)}")
    return g.iloc[np.linspace(0,len(g)-1,n).round().astype(int)].copy()

def run(features_path, events_path, output_dir, sample_per_side=250):
    x=pd.read_csv(features_path,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    e=pd.read_csv(events_path)
    required={"threshold_id","type","pivot_index","confirmation_index","pivot_date","pivot_price","reversal_pct"}
    if not required.issubset(e.columns): raise ValueError(f"Event columns missing: {sorted(required-set(e.columns))}")
    e=e.loc[e.threshold_id.astype(str)==THRESHOLD_ID].copy()
    if e.empty: raise ValueError(f"Frozen threshold {THRESHOLD_ID} has no events")
    e["pivot_index"]=pd.to_numeric(e.pivot_index,errors="raise").astype(int)
    selected=pd.concat([_stratified_select(e[e.type.astype(str).str.upper()==k],sample_per_side) for k in ("HIGH","LOW")],ignore_index=True)
    selected=selected.sort_values(["pivot_index","type"]).reset_index(drop=True)
    if len(selected)!=sample_per_side*2 or selected.pivot_index.duplicated().any():
        raise ValueError("Selected pivots are not exactly the requested unique sample")
    cols=[c for c in FEATURES if c in x.columns and pd.api.types.is_numeric_dtype(x[c])]
    missing=[c for c in FEATURES if c not in cols]
    if len(cols)<30: raise ValueError(f"Only {len(cols)} expected numeric features available")
    dates=pd.to_datetime(x.date); years=dates.dt.year.to_numpy()
    trend=x.trend_stack.fillna(0).astype(int).to_numpy() if "trend_stack" in x else np.zeros(len(x),int)
    excluded=np.zeros(len(x),dtype=bool)
    for pi in e.pivot_index.to_numpy():
        pi=int(pi); excluded[max(0,pi-10):min(len(x),pi+11)]=True
    valid=x[cols].notna().any(axis=1).to_numpy()
    rng=np.random.default_rng(SEED); snapshots=[]
    for event_no,r in enumerate(selected.itertuples(index=False),1):
        pi=int(r.pivot_index); kind=str(r.type).upper(); year=int(years[pi]); regime=int(trend[pi])
        for lead in LEADS:
            si=pi-lead
            if si<0: continue
            candidates=np.flatnonzero((years==year)&(trend==regime)&(~excluded)&valid); match="year_and_trend"
            if not len(candidates):
                candidates=np.flatnonzero((years==year)&(~excluded)&valid); match="year_only"
            if not len(candidates):
                candidates=np.flatnonzero((~excluded)&valid); match="all_years_fallback"
            if not len(candidates): raise ValueError("No eligible control dates")
            ci=int(rng.choice(candidates))
            z={"event_no":event_no,"threshold_id":THRESHOLD_ID,"pivot_type":kind,
               "pivot_date":pd.Timestamp(r.pivot_date).date().isoformat(),"pivot_index":pi,
               "pivot_price":float(r.pivot_price),"reversal_pct":float(r.reversal_pct),
               "confirmation_lag":int(r.confirmation_lag) if hasattr(r,"confirmation_lag") else None,
               "lead_sessions":lead,"feature_date":pd.Timestamp(dates.iloc[si]).date().isoformat(),
               "control_date":pd.Timestamp(dates.iloc[ci]).date().isoformat(),"control_match":match,
               "control_year":int(years[ci]),"control_trend_stack":int(trend[ci])}
            for c in cols:
                a=x.at[si,c]; b=x.at[ci,c]
                z[c]=float(a) if pd.notna(a) else np.nan
                z["control_"+c]=float(b) if pd.notna(b) else np.nan
            snapshots.append(z)
    snap=pd.DataFrame(snapshots); stats=[]
    for (kind,lead),g in snap.groupby(["pivot_type","lead_sessions"]):
        for c in cols:
            a=pd.to_numeric(g[c],errors="coerce").dropna().to_numpy(float)
            b=pd.to_numeric(g["control_"+c],errors="coerce").dropna().to_numpy(float)
            if len(a)<20 or len(b)<20: continue
            am,bm=float(np.mean(a)),float(np.mean(b)); av,bv=float(np.std(a,ddof=1)),float(np.std(b,ddof=1))
            pooled=np.sqrt(((len(a)-1)*av**2+(len(b)-1)*bv**2)/max(1,len(a)+len(b)-2))
            stats.append({"pivot_type":kind,"lead_sessions":int(lead),"feature":c,"event_n":len(a),"control_n":len(b),
              "event_median":float(np.median(a)),"control_median":float(np.median(b)),"event_mean":am,"control_mean":bm,
              "mean_difference":am-bm,"standardized_mean_difference":(am-bm)/pooled if pooled>0 else np.nan,
              "event_missing_pct":100*(1-len(a)/len(g)),"control_missing_pct":100*(1-len(b)/len(g)),
              "control_match_year_trend_pct":100*(g.control_match=="year_and_trend").mean()})
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    selected.to_csv(out/"tsmc_2330_500_selected_pivots.csv",index=False)
    snap.to_csv(out/"tsmc_2330_500_pivot_snapshots.csv",index=False)
    pd.DataFrame(stats).to_csv(out/"tsmc_2330_500_pivot_feature_comparison.csv",index=False)
    counts=selected.groupby("type").size().to_dict()
    result={"symbol":"2330","threshold_id":THRESHOLD_ID,"selected_pivots":len(selected),
      "selected_high":int(counts.get("HIGH",0)),"selected_low":int(counts.get("LOW",0)),
      "available_events_high":int((e.type.astype(str).str.upper()=="HIGH").sum()),
      "available_events_low":int((e.type.astype(str).str.upper()=="LOW").sum()),
      "feature_count":len(cols),"feature_columns_omitted":missing,"lead_sessions":list(LEADS),
      "snapshot_rows":len(snap),"control_match_distribution":snap.control_match.value_counts().to_dict(),
      "selection":"Evenly spaced chronological selection within each pivot direction from one frozen threshold.",
      "control":"One deterministic random non-pivot control per event/lead, matched by year and trend_stack where possible; controls excluded within 10 sessions of any pivot under the frozen threshold.",
      "interpretation":"Exploratory event study only. Pivots are retrospective labels; features are sampled 1/3/5/10/20 sessions before pivot. Descriptive effect sizes do not establish predictive value or tradable performance. Freeze any discovered rules and test on untouched later dates.",
      "outputs":["tsmc_2330_500_selected_pivots.csv","tsmc_2330_500_pivot_snapshots.csv","tsmc_2330_500_pivot_feature_comparison.csv"]}
    (out/"tsmc_2330_500_pivot_study_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--features",required=True); p.add_argument("--events",required=True)
    p.add_argument("--output",required=True); p.add_argument("--sample-per-side",type=int,default=250)
    a=p.parse_args(); print(json.dumps(run(a.features,a.events,a.output,a.sample_per_side),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
