"""Event-level diagnostic of the frozen high-point exit signal.

For each completed trade exited by the high-confirmation rule, records the first
high signal, its close-known features, actual next-session exit, and post-exit
forward path. Forward outcomes are diagnostic labels only and never fed into
signal selection or trading decisions. This is not OOF.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from tsmc_2330_4019_replication import inputs,build_candidate_signals,simulate,SLIPPAGE_BPS

FEATURES=[
 "return_1","close_location","intraday_range_pct","distance_sma_20","distance_sma_50",
 "distance_sma_200","rsi14","relative_volume20","stoch_k14","obv_flow_ratio_5",
 "macd_hist_change_3","macd_hist_pct","sma50_slope_10","distance_ema_20",
 "ema20_slope_5","adx14","plus_di14","minus_di14","distance_prior_high20"
]
def clean(v):
 try:
  x=float(v)
  return x if np.isfinite(x) else None
 except Exception:return None

def run(features,limits_path,dividends_path,output):
 d,limits,dividends=inputs(features,limits_path,dividends_path)
 d=d.sort_values("date").reset_index(drop=True)
 low,high,risk=build_candidate_signals(d)
 start=int(np.flatnonzero((d.date.dt.year>=2015).to_numpy())[0]);end=len(d)-1
 sim=simulate(d,low,high,risk,limits,dividends,10,start,end)
 ix={pd.Timestamp(x).date().isoformat():i for i,x in enumerate(d.date)}
 events=[]
 for tr in sim["trades"]:
  if not str(tr.get("exit_reason","")).startswith("high_confirm"):continue
  ei=ix.get(pd.Timestamp(tr["entry_date"]).date().isoformat())
  xi=ix.get(pd.Timestamp(tr["exit_date"]).date().isoformat())
  if ei is None or xi is None or xi<=ei:continue
  hits=np.flatnonzero(high[ei:xi+1])
  if not len(hits):continue
  si=ei+int(hits[0]); row=d.iloc[si]
  exit_open=float(d.iloc[xi].raw_open)
  event={
   "entry_date":tr["entry_date"],"high_signal_date":row.date.date().isoformat(),
   "exit_execution_date":tr["exit_date"],"exit_reason":tr["exit_reason"],
   "signal_to_execution_sessions":xi-si,"signal_close":clean(row.raw_close),
   "next_session_exit_open":exit_open,"trade_net_pnl":clean(tr.get("net_pnl")),
   "trade_net_return":clean(tr.get("net_return")),
   "features_at_signal":{k:clean(row[k]) for k in FEATURES if k in d.columns}
  }
  for h in (5,10,20,40,60):
   end_i=min(len(d)-1,xi+h-1)
   if end_i<xi:continue
   path=d.iloc[xi:end_i+1]
   event[f"fwd_{h}s_close_return_from_exit_open"]=clean(float(d.iloc[end_i].raw_close)/exit_open-1)
   event[f"fwd_{h}s_max_upside_from_exit_open"]=clean(float(path.raw_high.max())/exit_open-1)
   event[f"fwd_{h}s_max_drawdown_from_exit_open"]=clean(float(path.raw_low.min())/exit_open-1)
  events.append(event)
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(events).to_csv(out/"tsmc_2330_high_exit_forward_events.csv",index=False)
 # Aggregate descriptive outcomes only; small sample caveat is explicit.
 agg={}
 for h in (5,10,20,40,60):
  ups=[x[f"fwd_{h}s_max_upside_from_exit_open"] for x in events if x.get(f"fwd_{h}s_max_upside_from_exit_open") is not None]
  dd=[x[f"fwd_{h}s_max_drawdown_from_exit_open"] for x in events if x.get(f"fwd_{h}s_max_drawdown_from_exit_open") is not None]
  closes=[x[f"fwd_{h}s_close_return_from_exit_open"] for x in events if x.get(f"fwd_{h}s_close_return_from_exit_open") is not None]
  agg[str(h)]={"n":len(closes),"median_close_return":float(np.median(closes)) if closes else None,
    "positive_close_fraction":float(np.mean(np.array(closes)>0)) if closes else None,
    "median_max_upside":float(np.median(ups)) if ups else None,
    "median_max_drawdown":float(np.median(dd)) if dd else None,
    "upside_ge_5pct_fraction":float(np.mean(np.array(ups)>=.05)) if ups else None}
 summary={"classification":"DESCRIPTIVE_HIGH_EXIT_EVENT_DIAGNOSTIC_NOT_OOF",
  "period":[d.date.min().date().isoformat(),d.date.max().date().isoformat()],
  "account_period":[d.date.iloc[start].date().isoformat(),d.date.iloc[end].date().isoformat()],
  "high_confirmed_exit_events":len(events),"forward_path_summary":agg,
  "events":events,
  "warning":"Forward returns are outcomes used only to describe historical exit events. They are not inputs to any trading signal. Sample is small and history was previously inspected; do not claim independent validation."}
 (out/"tsmc_2330_high_exit_forward_diagnostic.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
 return summary
def main():
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--limits",required=True);p.add_argument("--dividends",required=True);p.add_argument("--output",default="artifacts/tsmc_2330_high_exit_forward_diagnostic")
 a=p.parse_args();print(json.dumps(run(a.features,a.limits,a.dividends,a.output),ensure_ascii=False,indent=2,default=float))
if __name__=="__main__":main()

# Frozen diagnostic specification: do not optimize on forward outcomes.
