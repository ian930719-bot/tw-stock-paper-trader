"""EMA20 reclaim after high-point exit: fixed trend-pullback re-entry hypothesis."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from tsmc_2330_4019_replication import INITIAL_CAPITAL,SLIPPAGE_BPS,inputs,build_candidate_signals,simulate,annual_metrics

COOLDOWN=5
def make_reentry_signal(d):
 required={"distance_ema_20","rsi14","distance_sma_50","sma50_slope_10","adx14","plus_di14","minus_di14","corporate_action_window"}
 missing=sorted(required-set(d.columns))
 if missing:raise ValueError(f"Missing causal re-entry features: {missing}")
 cross=(d.distance_ema_20.shift(1)<=0)&(d.distance_ema_20>0)
 signal=(cross&d.rsi14.between(40,55)&(d.distance_sma_50>0)&(d.sma50_slope_10>0)&(d.adx14>=18)&(d.plus_di14>d.minus_di14)&~d.corporate_action_window.fillna(True).astype(bool))
 return signal.fillna(False).to_numpy(bool)

def run(features,limits_path,dividends_path,output):
 d,limits,dividends=inputs(features,limits_path,dividends_path);d=d.sort_values("date").reset_index(drop=True)
 low,high,risk=build_candidate_signals(d);reentry=make_reentry_signal(d)
 start=int((d.date.dt.year>=2015).to_numpy().argmax());end=len(d)-1
 out=Path(output);out.mkdir(parents=True,exist_ok=True);results={}
 for name,sig in {"baseline_none":np.zeros(len(d),dtype=bool),"ema20_reclaim_pullback":reentry}.items():
  results[name]={}
  for slip in SLIPPAGE_BPS:
   sim=simulate(d,low,high,risk,limits,dividends,slip,start,end,reentry_signal=sig,reentry_cooldown=COOLDOWN)
   results[name][f"{slip}bps"]={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
   pd.DataFrame(sim["trades"]).to_csv(out/f"{name}_trades_{slip}bps.csv",index=False)
   pd.DataFrame(sim["equity_curve"]).to_csv(out/f"{name}_equity_{slip}bps.csv",index=False)
   pd.DataFrame(annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL)).to_csv(out/f"{name}_annual_{slip}bps.csv",index=False)
 events=d.loc[reentry,["date","distance_ema_20","rsi14","distance_sma_50","sma50_slope_10","adx14","plus_di14","minus_di14"]].copy()
 events.to_csv(out/"ema20_reclaim_candidate_events.csv",index=False)
 summary={"symbol":"2330","classification":"EXPLORATORY_FIXED_EMA20_RECLAIM_REENTRY_NOT_OOF",
  "account_period":[d.date.iloc[start].date().isoformat(),d.date.iloc[end].date().isoformat()],
  "signal":"At close, EMA20 distance crosses from <=0 to >0; RSI14 40-55; price above SMA50; SMA50 slope positive; ADX>=18; +DI>-DI; outside corporate-action window.",
  "execution":"Buy next trading session open, only after a high-point exit, with fixed 5-session cooldown; original low signals retain priority.",
  "candidate_signal_count":int(reentry.sum()),"metrics_by_policy_and_cost":results,
  "warning":"This is a new fixed rule family, tested on previously inspected history. No promotion without expanding annual OOF and win-rate floor."}
 (out/"tsmc_2330_ema20_reclaim_reentry.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
 return summary
def main():
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--limits",required=True);p.add_argument("--dividends",required=True);p.add_argument("--output",default="artifacts/tsmc_2330_ema20_reclaim_reentry");a=p.parse_args()
 print(json.dumps(run(a.features,a.limits,a.dividends,a.output),ensure_ascii=False,indent=2,default=float))
if __name__=="__main__":main()

# Fixed trend-pullback re-entry hypothesis; no parameter grid.
