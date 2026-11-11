"""Post-hoc outcomes for frozen out-of-fold TSMC signals; never used for model fitting."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

FORWARD=(1,3,5,10,20,40)
FEATURE_PREFIXES=("rsi","bb_","distance_sma","distance_ema","return_","relative_volume",
 "turnover_ratio","close_location","atr_pct","adx","macd","stoch","cci","williams",
 "volume_zscore","donchian","obv","mfi")

def run(features_path,events_path,predictions_path,output_dir):
    x=pd.read_csv(features_path,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    e=pd.read_csv(events_path)
    e["pivot_date"]=pd.to_datetime(e["pivot_date"],errors="coerce")
    p=pd.read_csv(predictions_path,parse_dates=["date"])
    required={"date","open","high","low","close"}
    if not required.issubset(x): raise ValueError(f"Missing feature fields: {sorted(required-set(x))}")
    if not {"threshold_id","type","pivot_date"}.issubset(e): raise ValueError("Incomplete pivot event schema")
    if not {"date","threshold_id","horizon_sessions","pivot_type","probability","threshold","signal"}.issubset(p):
        raise ValueError("Incomplete OOF prediction schema")
    if x.date.duplicated().any(): raise ValueError("Duplicate feature dates")
    date_index={pd.Timestamp(d).normalize():i for i,d in enumerate(x.date)}
    signals=p.loc[p.signal.astype(bool)].copy()
    rows=[]
    for r in signals.itertuples(index=False):
        i=date_index.get(pd.Timestamp(r.date).normalize())
        if i is None: continue
        kind=str(r.pivot_type).upper(); h=int(r.horizon_sessions)
        candidates=e.loc[(e.threshold_id.astype(str)==str(r.threshold_id))&
                         (e.type.astype(str).str.upper()==kind),"pivot_date"].dropna()
        future=[]
        for d in candidates:
            j=date_index.get(pd.Timestamp(d).normalize())
            if j is not None and i<j<=i+40: future.append((j-i,j))
        future.sort()
        lag,pivot=future[0] if future else (None,None)
        end20=min(i+20,len(x)-1); end40=min(i+40,len(x)-1)
        r20=float(x.close.iloc[end20]/x.close.iloc[i]-1) if end20>i else np.nan
        r40=float(x.close.iloc[end40]/x.close.iloc[i]-1) if end40>i else np.nan
        if lag is None:
            continued=(r20>=.02 if kind=="HIGH" else r20<=-.02) if np.isfinite(r20) else False
            label="trend_continuation" if continued else "complete_failure"
        elif lag==h: label="exact_hit"
        elif lag<h: label="late_signal_pivot_already_formed"
        elif lag<=h+5: label="early_warning_1_to_5_sessions"
        else: label="delayed_pivot_6_to_40_sessions"
        rec={"threshold_id":r.threshold_id,"analysis_year":int(r.analysis_year),
             "validation_year":int(r.validation_year),"horizon_sessions":h,"pivot_type":kind,
             "signal_date":pd.Timestamp(r.date).date().isoformat(),"probability":float(r.probability),
             "frozen_threshold":float(r.threshold),"next_pivot_lag":lag,
             "next_pivot_date":pd.Timestamp(pivot).date().isoformat() if pivot is not None else None,
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
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    detail=pd.DataFrame(rows)
    detail.to_csv(out/"tsmc_2330_oof_signal_outcomes.csv",index=False)
    if detail.empty:
        summary=pd.DataFrame(columns=["threshold_id","horizon_sessions","pivot_type","outcome_class","signals"])
        medians=pd.DataFrame()
    else:
        summary=detail.groupby(["threshold_id","horizon_sessions","pivot_type","outcome_class"],dropna=False).size().rename("signals").reset_index()
        fcols=[c for c in detail if c.startswith(FEATURE_PREFIXES)]
        medians=detail.groupby(["threshold_id","horizon_sessions","pivot_type","outcome_class"])[fcols].median(numeric_only=True).reset_index()
    summary.to_csv(out/"tsmc_2330_oof_signal_outcome_summary.csv",index=False)
    medians.to_csv(out/"tsmc_2330_oof_signal_feature_medians.csv",index=False)
    return {"signals_analysed":len(detail),"outcome_classes":detail.outcome_class.value_counts().to_dict() if not detail.empty else {},
            "note":"Retrospective outcome diagnostics only. Future outcomes are never used to fit or calibrate the signal model."}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",required=True); ap.add_argument("--events",required=True)
    ap.add_argument("--predictions",required=True); ap.add_argument("--output",required=True)
    a=ap.parse_args()
    print(json.dumps(run(a.features,a.events,a.predictions,a.output),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
