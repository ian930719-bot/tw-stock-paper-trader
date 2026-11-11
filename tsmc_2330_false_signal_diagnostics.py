"""Post-hoc diagnostics for frozen CPCV out-of-fold signals; never used for model fitting."""
from __future__ import annotations
import argparse, json
from bisect import bisect_right
from pathlib import Path
import numpy as np
import pandas as pd

FORWARD=(1,3,5,10,20,40)
FEATURE_PREFIXES=("rsi","bb_","distance_sma","distance_ema","return_","relative_volume",
 "turnover_ratio","close_location","atr_pct","adx","macd","stoch","cci","williams",
 "volume_zscore","donchian","obv","mfi","foreign_","trust_")

def run(features_path,events_path,predictions_path,output_dir):
    x=pd.read_csv(features_path,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    e=pd.read_csv(events_path)
    e["pivot_date"]=pd.to_datetime(e["pivot_date"],errors="coerce")
    p=pd.read_csv(predictions_path,parse_dates=["date"])
    if not {"date","open","high","low","close"}.issubset(x):
        raise ValueError("Feature data must contain date/open/high/low/close")
    if not {"threshold_id","type","pivot_date"}.issubset(e):
        raise ValueError("Pivot event schema incomplete")
    if not {"feature_set","date","threshold_id","horizon_sessions","pivot_type","probability","signal","event"}.issubset(p):
        raise ValueError("CPCV prediction schema incomplete")
    if x.date.duplicated().any() or not x.date.is_monotonic_increasing:
        raise ValueError("Feature dates must be unique ascending")
    date_to_i={pd.Timestamp(d).normalize():i for i,d in enumerate(x.date)}
    # Keep only frozen model-positive signals. Outcomes below are strictly post-hoc.
    signals=p.loc[p.signal.astype(bool)].copy()
    event_map={}
    for (tid,kind),g in e.dropna(subset=["pivot_date"]).groupby(["threshold_id","type"]):
        event_map[(str(tid),str(kind).upper())]=sorted(
            date_to_i[pd.Timestamp(d).normalize()] for d in g.pivot_date
            if pd.Timestamp(d).normalize() in date_to_i)
    rows=[]
    for r in signals.itertuples(index=False):
        i=date_to_i.get(pd.Timestamp(r.date).normalize())
        if i is None: continue
        kind=str(r.pivot_type).upper(); h=int(r.horizon_sessions)
        candidates=event_map.get((str(r.threshold_id),kind),[])
        pos=bisect_right(candidates,i)
        lag=(candidates[pos]-i) if pos<len(candidates) and candidates[pos]-i<=40 else None
        pivot=(candidates[pos] if lag is not None else None)
        end20=min(i+20,len(x)-1); end40=min(i+40,len(x)-1)
        r20=float(x.close.iloc[end20]/x.close.iloc[i]-1) if end20>i else np.nan
        r40=float(x.close.iloc[end40]/x.close.iloc[i]-1) if end40>i else np.nan
        if lag is None:
            continued=(r20>=.02 if kind=="HIGH" else r20<=-.02) if np.isfinite(r20) else False
            label="trend_continuation" if continued else "complete_failure"
        elif lag<h: label="late_signal_pivot_already_formed"
        elif lag==h: label="exact_hit"
        elif lag<=h+5: label="early_warning_1_to_5_sessions"
        else: label="delayed_pivot_6_to_40_sessions"
        rec={"feature_set":r.feature_set,"threshold_id":r.threshold_id,
             "horizon_sessions":h,"pivot_type":kind,"signal_date":pd.Timestamp(r.date).date().isoformat(),
             "probability":float(r.probability),"exact_horizon_event":bool(r.event),
             "next_pivot_lag":lag,
             "next_pivot_date":pd.Timestamp(x.date.iloc[pivot]).date().isoformat() if pivot is not None else None,
             "outcome_class":label,"forward_close_return_20":r20,"forward_close_return_40":r40}
        for n in FORWARD:
            j=min(i+n,len(x)-1)
            rec[f"close_return_{n}"]=float(x.close.iloc[j]/x.close.iloc[i]-1) if j>i else np.nan
            rec[f"max_upside_{n}"]=float(x.high.iloc[i+1:j+1].max()/x.close.iloc[i]-1) if j>i else np.nan
            rec[f"max_downside_{n}"]=float(x.low.iloc[i+1:j+1].min()/x.close.iloc[i]-1) if j>i else np.nan
        for c in x.columns:
            if c.startswith(FEATURE_PREFIXES):
                v=x.at[i,c]
                rec[c]=float(v) if pd.notna(v) and isinstance(v,(int,float,np.integer,np.floating)) else np.nan
        rows.append(rec)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    detail=pd.DataFrame(rows)
    detail.to_csv(out/"tsmc_2330_cpcv_signal_outcomes.csv",index=False)
    keys=["feature_set","threshold_id","horizon_sessions","pivot_type","outcome_class"]
    summary=(detail.groupby(keys,dropna=False).size().rename("signals").reset_index()
             if not detail.empty else pd.DataFrame(columns=keys+["signals"]))
    summary.to_csv(out/"tsmc_2330_cpcv_signal_outcome_summary.csv",index=False)
    broad=["feature_set","horizon_sessions","pivot_type","outcome_class"]
    comparative=(detail.groupby(broad,dropna=False).size().rename("signals").reset_index()
                 if not detail.empty else pd.DataFrame(columns=broad+["signals"]))
    comparative.to_csv(out/"tsmc_2330_cpcv_signal_comparison.csv",index=False)
    fcols=[c for c in detail if c.startswith(FEATURE_PREFIXES)]
    medians=(detail.groupby(keys)[fcols].median(numeric_only=True).reset_index()
             if not detail.empty and fcols else pd.DataFrame())
    medians.to_csv(out/"tsmc_2330_cpcv_signal_feature_medians.csv",index=False)
    classes=detail.outcome_class.value_counts().to_dict() if not detail.empty else {}
    return {"signals_analysed":len(detail),"feature_sets":sorted(detail.feature_set.unique().tolist()) if not detail.empty else [],
      "outcome_classes":classes,
      "note":"Retrospective CPCV signal outcome diagnostics only. Future pivots and returns are not used to fit, tune, or calibrate the signal model."}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",required=True);ap.add_argument("--events",required=True)
    ap.add_argument("--predictions",required=True);ap.add_argument("--output",required=True)
    a=ap.parse_args()
    print(json.dumps(run(a.features,a.events,a.predictions,a.output),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
