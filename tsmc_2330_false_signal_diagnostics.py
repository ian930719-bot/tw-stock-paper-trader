"""Post-hoc CPCV signal outcomes; future outcomes are diagnostic only, never model inputs."""
from __future__ import annotations
import argparse,json
from bisect import bisect_right
from pathlib import Path
import numpy as np
import pandas as pd

PREFIXES=("rsi","bb_","distance_sma","distance_ema","return_","relative_volume","turnover_ratio","close_location","atr_pct","adx","macd","stoch","cci","williams","volume_zscore","donchian","obv","mfi","foreign_","trust_")
def run(features_path,events_path,predictions_path,output_dir):
 x=pd.read_csv(features_path,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
 e=pd.read_csv(events_path);e["pivot_date"]=pd.to_datetime(e["pivot_date"],errors="coerce")
 p=pd.read_csv(predictions_path,parse_dates=["date"])
 if x.date.duplicated().any() or not x.date.is_monotonic_increasing: raise ValueError("Feature dates must be unique ascending")
 if not {"threshold_id","type","pivot_date"}.issubset(e): raise ValueError("Pivot event schema incomplete")
 req={"feature_set","date","threshold_id","horizon_sessions","pivot_type","probability","signal","event"}
 if not req.issubset(p): raise ValueError(f"CPCV prediction schema missing {sorted(req-set(p))}")
 di={pd.Timestamp(d).normalize():i for i,d in enumerate(x.date)}
 em={}
 for (tid,kind),g in e.dropna(subset=["pivot_date"]).groupby(["threshold_id","type"]):
  em[(str(tid),str(kind).upper())]=sorted(di[pd.Timestamp(d).normalize()] for d in g.pivot_date if pd.Timestamp(d).normalize() in di)
 rows=[]
 for r in p.loc[p.signal.astype(bool)].itertuples(index=False):
  i=di.get(pd.Timestamp(r.date).normalize())
  if i is None: continue
  kind=str(r.pivot_type).upper();h=int(r.horizon_sessions);cand=em.get((str(r.threshold_id),kind),[])
  k=bisect_right(cand,i);lag=cand[k]-i if k<len(cand) and cand[k]-i<=40 else None
  pivot=cand[k] if lag is not None else None
  j20=min(i+20,len(x)-1);ret20=float(x.close.iloc[j20]/x.close.iloc[i]-1) if j20>i else np.nan
  if lag is None:
   cont=(ret20>=.02 if kind=="HIGH" else ret20<=-.02) if np.isfinite(ret20) else False
   label="trend_continuation" if cont else "complete_failure"
  elif lag<h: label="late_signal_pivot_already_formed"
  elif lag==h: label="exact_hit"
  elif lag<=h+5: label="early_warning_1_to_5_sessions"
  else: label="delayed_pivot_6_to_40_sessions"
  z={"feature_set":r.feature_set,"threshold_id":r.threshold_id,"horizon_sessions":h,"pivot_type":kind,
   "signal_date":pd.Timestamp(r.date).date().isoformat(),"probability":float(r.probability),
   "exact_horizon_event":bool(r.event),"next_pivot_lag":lag,
   "next_pivot_date":pd.Timestamp(x.date.iloc[pivot]).date().isoformat() if pivot is not None else None,
   "outcome_class":label,"forward_close_return_20":ret20}
  for n in (1,3,5,10,20,40):
   j=min(i+n,len(x)-1)
   z[f"close_return_{n}"]=float(x.close.iloc[j]/x.close.iloc[i]-1) if j>i else np.nan
   z[f"max_upside_{n}"]=float(x.high.iloc[i+1:j+1].max()/x.close.iloc[i]-1) if j>i else np.nan
   z[f"max_downside_{n}"]=float(x.low.iloc[i+1:j+1].min()/x.close.iloc[i]-1) if j>i else np.nan
  for c in x.columns:
   if c.startswith(PREFIXES):
    v=x.at[i,c];z[c]=float(v) if pd.notna(v) and isinstance(v,(int,float,np.integer,np.floating)) else np.nan
  rows.append(z)
 out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);d=pd.DataFrame(rows)
 d.to_csv(out/"tsmc_2330_cpcv_signal_outcomes.csv",index=False)
 keys=["feature_set","threshold_id","horizon_sessions","pivot_type","outcome_class"]
 sm=d.groupby(keys,dropna=False).size().rename("signals").reset_index() if not d.empty else pd.DataFrame(columns=keys+["signals"])
 sm.to_csv(out/"tsmc_2330_cpcv_signal_outcome_summary.csv",index=False)
 broad=["feature_set","horizon_sessions","pivot_type","outcome_class"]
 cp=d.groupby(broad,dropna=False).size().rename("signals").reset_index() if not d.empty else pd.DataFrame(columns=broad+["signals"])
 cp.to_csv(out/"tsmc_2330_cpcv_signal_comparison.csv",index=False)
 fc=[c for c in d if c.startswith(PREFIXES)]
 md=d.groupby(keys)[fc].median(numeric_only=True).reset_index() if not d.empty and fc else pd.DataFrame()
 md.to_csv(out/"tsmc_2330_cpcv_signal_feature_medians.csv",index=False)
 return {"signals_analysed":len(d),"outcome_classes":d.outcome_class.value_counts().to_dict() if not d.empty else {},"note":"Retrospective CPCV diagnostics only; future outcomes never fit, tune, or calibrate the model."}
def main():
 a=argparse.ArgumentParser();a.add_argument("--features",required=True);a.add_argument("--events",required=True);a.add_argument("--predictions",required=True);a.add_argument("--output",required=True)
 z=a.parse_args();print(json.dumps(run(z.features,z.events,z.predictions,z.output),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
