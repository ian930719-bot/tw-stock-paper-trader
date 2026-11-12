"""Fixed low-signal confirmation comparison on the frozen base entry rule."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from tsmc_2330_4019_replication import INITIAL_CAPITAL,SLIPPAGE_BPS,inputs,build_candidate_signals,simulate,annual_metrics

POLICIES=["immediate","bullish_day_3","two_higher_closes_5","signal_high_breakout_5"]
def run(features,limits_path,dividends_path,output):
 d,limits,dividends=inputs(features,limits_path,dividends_path);d=d.sort_values("date").reset_index(drop=True)
 low,high,risk=build_candidate_signals(d);start=int((d.date.dt.year>=2015).to_numpy().argmax());end=len(d)-1
 out=Path(output);out.mkdir(parents=True,exist_ok=True);results={}
 for policy in POLICIES:
  results[policy]={}
  for slip in SLIPPAGE_BPS:
   sim=simulate(d,low,high,risk,limits,dividends,slip,start,end,low_entry_policy=policy)
   results[policy][f"{slip}bps"]={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
   pd.DataFrame(sim["trades"]).to_csv(out/f"{policy}_trades_{slip}bps.csv",index=False)
   pd.DataFrame(sim["equity_curve"]).to_csv(out/f"{policy}_equity_{slip}bps.csv",index=False)
   pd.DataFrame(annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL)).to_csv(out/f"{policy}_annual_{slip}bps.csv",index=False)
 summary={"symbol":"2330","classification":"EXPLORATORY_FIXED_LOW_ENTRY_CONFIRMATION_COMPARISON_NOT_OOF",
  "account_period":[d.date.iloc[start].date().isoformat(),d.date.iloc[end].date().isoformat()],
  "base_low_signal":"unchanged; relative volume>=1.2, return_1<=-0.6%, stoch<=30, close_location<=0.25, OBV flow<=0, SMA50 distance<=4%",
  "policies":{
   "immediate":"original next-session-open entry",
   "bullish_day_3":"after low signal, wait up to 3 sessions for close>open and close>prior close; buy following open",
   "two_higher_closes_5":"after low signal, wait up to 5 sessions for two consecutive higher closes; buy following open",
   "signal_high_breakout_5":"after low signal, wait up to 5 sessions for close above the signal-day high; buy following open"
  },
  "metrics_by_policy_and_cost":results,
  "warning":"Exploratory on previously inspected history. Confirmation policies are fixed, not optimized; only expanding OOF can establish generalization."}
 (out/"tsmc_2330_low_entry_confirmation_comparison.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
 return summary
def main():
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--limits",required=True);p.add_argument("--dividends",required=True);p.add_argument("--output",default="artifacts/tsmc_2330_low_entry_confirmation");a=p.parse_args()
 print(json.dumps(run(a.features,a.limits,a.dividends,a.output),ensure_ascii=False,indent=2,default=float))
if __name__=="__main__":main()

# Frozen entry confirmation hypotheses; no parameter grid.
