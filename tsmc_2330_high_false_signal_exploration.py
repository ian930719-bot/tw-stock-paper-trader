"""Fixed high-point false-signal / trend-continuation gate experiment.

Signals are known at the close; orders execute next session open. This run is
exploratory on previously inspected history and is not independent OOF.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from tsmc_2330_4019_replication import INITIAL_CAPITAL,SLIPPAGE_BPS,inputs,build_candidate_signals,simulate,annual_metrics

def hold_candidates(d):
 required={"distance_ema_20","ema20_slope_5","rsi14","macd_hist_pct","adx14","plus_di14","minus_di14","distance_prior_high20","corporate_action_window"}
 missing=sorted(required-set(d.columns))
 if missing: raise ValueError(f"Missing hold-filter features: {missing}")
 action=d.corporate_action_window.fillna(True).astype(bool)
 trend=(d.distance_ema_20>0)&(d.ema20_slope_5>0)&(d.rsi14>=58)&(d.macd_hist_pct>0)
 rules={
  "hold_ema20_rsi58_macd_positive":trend,
  "hold_plus_adx_directional_strength":trend&(d.adx14>=20)&(d.plus_di14>d.minus_di14),
  "hold_plus_near_20d_high":trend&(d.distance_prior_high20>=-0.02),
 }
 return {k:(v.fillna(False).to_numpy(bool)&~action.to_numpy(bool)) for k,v in rules.items()}

def run(features,limits_path,dividends_path,output):
 d,limits,dividends=inputs(features,limits_path,dividends_path)
 d=d.sort_values("date").reset_index(drop=True)
 low,high,risk=build_candidate_signals(d)
 start=int(np.flatnonzero((d.date.dt.year>=2015).to_numpy())[0]);end=len(d)-1
 filters={"baseline_no_suppression":np.zeros(len(d),dtype=bool),**hold_candidates(d)}
 out=Path(output);out.mkdir(parents=True,exist_ok=True);result={}
 for name,hold in filters.items():
  result[name]={}
  for slip in SLIPPAGE_BPS:
   sim=simulate(d,low,high,risk,limits,dividends,slip,start,end,high_hold_signal=hold)
   result[name][f"{slip}bps"]={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
   pd.DataFrame(sim["trades"]).to_csv(out/f"{name}_trades_{slip}bps.csv",index=False)
   pd.DataFrame(sim["equity_curve"]).to_csv(out/f"{name}_equity_{slip}bps.csv",index=False)
   pd.DataFrame(annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL)).to_csv(out/f"{name}_annual_{slip}bps.csv",index=False)
 summary={"symbol":"2330","classification":"EXPLORATORY_FIXED_HIGH_FALSE_SIGNAL_FILTER_NOT_OOF",
  "period":[d.date.min().date().isoformat(),d.date.max().date().isoformat()],
  "account_period":[d.date.iloc[start].date().isoformat(),d.date.iloc[end].date().isoformat()],
  "method":"At a high-point signal, suppress the exit only while the predeclared trend-continuation filter is true; if continuation later weakens, the original high confirmation logic resumes. Risk exit remains higher priority.",
  "filters":list(filters),"metrics_by_filter_and_cost":result,
  "warning":"Previously inspected history; no filter can be promoted without annual expanding OOF and win-rate floor >=86.9565%."}
 (out/"tsmc_2330_high_false_signal_exploration.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--limits",required=True);p.add_argument("--dividends",required=True);p.add_argument("--output",default="artifacts/tsmc_2330_high_false_signal_exploration")
 a=p.parse_args();print(json.dumps(run(a.features,a.limits,a.dividends,a.output),ensure_ascii=False,indent=2,default=float))
if __name__=="__main__":main()
