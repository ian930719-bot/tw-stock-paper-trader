"""All-event high-signal continuation diagnostic with non-overlapping events.

The 10-session de-duplication rule is fixed before inspection. Forward outcomes
are descriptive labels only, never trading inputs. Not OOF.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from tsmc_2330_4019_replication import inputs,build_candidate_signals
FEATURES=["return_1","close_location","intraday_range_pct","distance_sma_20","distance_sma_50","distance_sma_200","rsi14","relative_volume20","stoch_k14","obv_flow_ratio_5","macd_hist_change_3","macd_hist_pct","sma50_slope_10","distance_ema_20","ema20_slope_5","adx14","plus_di14","minus_di14","distance_prior_high20"]
def val(x):
 try:
  z=float(x);return z if np.isfinite(z) else None
 except Exception:return None
def run(features,limits,dividends,output):
 d,_,_=inputs(features,limits,dividends);d=d.sort_values("date").reset_index(drop=True)
 _,high,_=build_candidate_signals(d)
 hits=np.flatnonzero(high);selected=[];last=-999
 for i in hits:
  if i-last<10:continue
  selected.append(int(i));last=int(i)
 rows=[]
 for n,i in enumerate(selected):
  if i+1>=len(d):continue
  ex=i+1;openp=float(d.iloc[ex].raw_open);row=d.iloc[i]
  r={"event_id":n+1,"signal_date":row.date.date().isoformat(),"hypothetical_execution_date":d.iloc[ex].date.date().isoformat(),"signal_close":val(row.raw_close),"execution_open":openp,
   "features":{k:val(row[k]) for k in FEATURES if k in d.columns}}
  for h in (5,10,20,40,60):
   j=min(len(d)-1,ex+h-1);path=d.iloc[ex:j+1]
   r[f"fwd_{h}s_close_return"]=val(float(d.iloc[j].raw_close)/openp-1)
   r[f"fwd_{h}s_mfe"]=val(float(path.raw_high.max())/openp-1)
   r[f"fwd_{h}s_mae"]=val(float(path.raw_low.min())/openp-1)
  rows.append(r)
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 flat=[]
 for r in rows:
  x={k:v for k,v in r.items() if k!="features"};x.update(r["features"]);flat.append(x)
 pd.DataFrame(flat).to_csv(out/"all_high_signal_events.csv",index=False)
 summary={}
 for h in (5,10,20,40,60):
  c=[x[f"fwd_{h}s_close_return"] for x in rows if x.get(f"fwd_{h}s_close_return") is not None]
  m=[x[f"fwd_{h}s_mfe"] for x in rows if x.get(f"fwd_{h}s_mfe") is not None]
  a=[x[f"fwd_{h}s_mae"] for x in rows if x.get(f"fwd_{h}s_mae") is not None]
  summary[str(h)]={"n":len(c),"median_close":float(np.median(c)) if c else None,"positive_close_fraction":float(np.mean(np.array(c)>0)) if c else None,"median_mfe":float(np.median(m)) if m else None,"median_mae":float(np.median(a)) if a else None,"close_return_ge_5pct_fraction":float(np.mean(np.array(c)>=.05)) if c else None}
 result={"classification":"EXPLORATORY_ALL_HIGH_SIGNAL_EVENT_DIAGNOSTIC_NOT_OOF","raw_high_signal_count":int(len(hits)),"deduplicated_event_count":len(rows),"deduplication_sessions":10,"date_range":[d.date.min().date().isoformat(),d.date.max().date().isoformat()],"forward_summary":summary,"warning":"Future returns are descriptive outcomes only. All historical data has been inspected; do not treat as independent blind validation."}
 (out/"all_high_signal_summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--limits",required=True);p.add_argument("--dividends",required=True);p.add_argument("--output",default="artifacts/tsmc_2330_all_high_signal_diagnostic");a=p.parse_args()
 print(json.dumps(run(a.features,a.limits,a.dividends,a.output),indent=2,ensure_ascii=False))
if __name__=="__main__":main()
