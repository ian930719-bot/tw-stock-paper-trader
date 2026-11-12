"""Expanding annual OOF model for filtering the frozen low-entry signal.

Events are de-duplicated by a fixed 10-session gap. The target is whether the
20-session close return from next-session open is positive. Each annual model
uses only events whose full 20-session label horizon ended before that year.
This is not pristine blind OOS because the event family was designed after
historical inspection; no future labels enter validation predictions.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,brier_score_loss
from tsmc_2330_4019_replication import INITIAL_CAPITAL,SLIPPAGE_BPS,inputs,build_candidate_signals,simulate,annual_metrics

FEATURES=["rsi14","distance_sma_200","bb_position20","mfi14","volatility_compression","distance_ema_20","atr_pct14","bullish_rsi_divergence20"]
MIN_TRAIN_EVENTS=15
PROBABILITY_THRESHOLD=.50
LABEL_HORIZON=20

def dedup_indices(signal,min_gap=10):
 hits=np.flatnonzero(signal);selected=[];last=-10**9
 for i in hits:
  if int(i)-last<min_gap:continue
  selected.append(int(i));last=int(i)
 return selected

def build_events(d,low):
 rows=[]
 for idx in dedup_indices(low,10):
  if idx+1>=len(d):continue
  end_idx=idx+LABEL_HORIZON
  label=None
  if end_idx<len(d):
   entry=float(d.iloc[idx+1].raw_open)
   close=float(d.iloc[end_idx].raw_close)
   label=int(close/entry-1>0)
  rows.append({"idx":idx,"label_end_idx":end_idx if label is not None else None,
   "signal_date":d.iloc[idx].date.date().isoformat(),"year":int(d.iloc[idx].date.year),
   "label":label,"features":{f:(float(d.iloc[idx][f]) if pd.notna(d.iloc[idx][f]) else np.nan) for f in FEATURES}})
 return rows

def make_model():
 return Pipeline([
  ("imputer",SimpleImputer(strategy="median")),
  ("scale",StandardScaler()),
  ("model",LogisticRegression(C=.1,class_weight="balanced",max_iter=1000,random_state=23302026))
 ])

def run(features,limits_path,dividends_path,output):
 d,limits,dividends=inputs(features,limits_path,dividends_path);d=d.sort_values("date").reset_index(drop=True)
 missing=sorted(set(FEATURES)-set(d.columns))
 if missing:raise ValueError(f"Missing model features: {missing}")
 low,high,risk=build_candidate_signals(d);events=build_events(d,low)
 years=sorted(int(y) for y in d.date.dt.year.unique())
 filtered_low=np.zeros(len(d),dtype=bool);folds=[];predictions=[]
 for year in [y for y in years if y>=2015]:
  year_idx=np.flatnonzero((d.date.dt.year==year).to_numpy())
  if not len(year_idx):continue
  start_idx=int(year_idx[0])
  train=[e for e in events if e["label"] is not None and e["year"]<year and e["label_end_idx"]<start_idx]
  valid=[e for e in events if e["year"]==year]
  use_model=(len(train)>=MIN_TRAIN_EVENTS and len({e["label"] for e in train})==2)
  model=make_model() if use_model else None
  if use_model:
   X=pd.DataFrame([{f:e["features"][f] for f in FEATURES} for e in train],columns=FEATURES)
   y=np.array([e["label"] for e in train],dtype=int)
   model.fit(X,y)
  accepted=0
  for e in valid:
   if use_model:
    Xv=pd.DataFrame([{f:e["features"][f] for f in FEATURES}],columns=FEATURES)
    prob=float(model.predict_proba(Xv)[0,1])
    allow=prob>=PROBABILITY_THRESHOLD
    method="model"
   else:
    prob=None;allow=True;method="baseline_fallback_insufficient_training"
   if allow:filtered_low[e["idx"]]=True;accepted+=1
   predictions.append({"signal_date":e["signal_date"],"year":year,"training_events":len(train),
    "model_used":use_model,"predicted_positive_probability":prob,"allow_entry":bool(allow),
    "actual_20s_positive_label":e["label"],"label_end_idx":e["label_end_idx"],"decision_method":method})
  folds.append({"validation_year":year,"training_event_count":len(train),
   "training_positive_fraction":float(np.mean([e["label"] for e in train])) if train else None,
   "validation_event_count":len(valid),"accepted_event_count":accepted,
   "model_used":use_model,"feature_names":FEATURES if use_model else [],
   "label_horizon_sessions":LABEL_HORIZON,"training_labels_matured_before_validation":True})
 dedup_low=np.zeros(len(d),dtype=bool)
 for e in events:dedup_low[e["idx"]]=True
 # OOF classification diagnostics only on events with realized labels.
 realized=[p for p in predictions if p["actual_20s_positive_label"] is not None and p["predicted_positive_probability"] is not None]
 cls={}
 if realized:
  yy=np.array([p["actual_20s_positive_label"] for p in realized]);pp=np.array([p["predicted_positive_probability"] for p in realized]);pred=(pp>=PROBABILITY_THRESHOLD).astype(int)
  cls={"n":len(yy),"positive_fraction":float(yy.mean()),"accuracy":float(accuracy_score(yy,pred)),
   "balanced_accuracy":float(balanced_accuracy_score(yy,pred)),
   "brier_score":float(brier_score_loss(yy,pp)),
   "roc_auc":float(roc_auc_score(yy,pp)) if len(np.unique(yy))==2 else None,
   "threshold":PROBABILITY_THRESHOLD}
 idx=np.flatnonzero((d.date.dt.year>=2015).to_numpy());start=int(idx[0]);end=int(idx[-1])
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 metrics={};annual={}
 low_streams={"original_full_low_signal":low,"deduplicated_low_signal":dedup_low,"model_filtered_low_signal":filtered_low}
 for stream_name,low_stream in low_streams.items():
  metrics[stream_name]={};annual[stream_name]={}
  for slip in SLIPPAGE_BPS:
   sim=simulate(d,low_stream,high,risk,limits,dividends,slip,start,end)
   metrics[stream_name][f"{slip}bps"]={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
   annual[stream_name][f"{slip}bps"]=annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL)
   pd.DataFrame(sim["trades"]).to_csv(out/f"{stream_name}_trades_{slip}bps.csv",index=False)
   pd.DataFrame(sim["equity_curve"]).to_csv(out/f"{stream_name}_equity_{slip}bps.csv",index=False)
   pd.DataFrame(annual[stream_name][f"{slip}bps"]).to_csv(out/f"{stream_name}_annual_{slip}bps.csv",index=False)
 pd.DataFrame(predictions).to_csv(out/"annual_oof_low_signal_predictions.csv",index=False)
 summary={"symbol":"2330","classification":"EXPANDING_ANNUAL_OOF_LOW_SIGNAL_LOGISTIC_FILTER_NOT_PRISTINE_BLIND_OOS",
  "data_period":[d.date.min().date().isoformat(),d.date.max().date().isoformat()],
  "validation_period":[d.date.iloc[start].date().isoformat(),d.date.iloc[end].date().isoformat()],
  "protocol":"10-session deduplicated low events; 20-session forward close-return label; annual expanding training; each training label horizon must mature before validation year; one continuous 2015+ account.",
  "features":FEATURES,"minimum_training_events":MIN_TRAIN_EVENTS,"probability_threshold":PROBABILITY_THRESHOLD,
  "oof_event_classifier_metrics":cls,"annual_folds":folds,
  "continuous_oof_metrics_by_cost":metrics,"annual_oof_metrics_by_cost":annual,
  "target_status":"UNMET unless CAGR>=50% and win rate>=86.9565% at every cost; MDD<25% separately reported.",
  "limitations":["The entry-signal family and event definition were developed on previously inspected history; this is not pristine blind OOS.","The 20-session positive-close label is a proxy, not identical to full-trade profitability.","The 10-session deduplication changes the baseline event stream; results must be compared with the frozen baseline." ]}
 (out/"tsmc_2330_low_signal_expanding_oof.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--limits",required=True);p.add_argument("--dividends",required=True);p.add_argument("--output",default="artifacts/tsmc_2330_low_signal_expanding_oof");a=p.parse_args()
 x=run(a.features,a.limits,a.dividends,a.output)
 print(json.dumps({k:v for k,v in x.items() if k not in {"annual_folds"}},ensure_ascii=False,indent=2,default=float))
if __name__=="__main__":main()

# Frozen annual OOF model specification; no full-period tuning.
