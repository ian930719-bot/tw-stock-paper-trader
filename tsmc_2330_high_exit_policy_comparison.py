"""Fixed high-exit confirmation policy comparison.

Four predeclared policies (handoff baseline plus three structural alternatives)
are run against the same frozen low-entry and SMA200 risk rules. No parameter
grid; all results are exploratory on previously inspected history, not OOF.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from tsmc_2330_4019_replication import INITIAL_CAPITAL,SLIPPAGE_BPS,inputs,build_candidate_signals,simulate,annual_metrics

POLICIES=["handoff_or","and_sma20_rsi55","ema20_rsi55","prior5low"]
def run(features,limits_path,dividends_path,output):
 d,limits,dividends=inputs(features,limits_path,dividends_path)
 d=d.sort_values("date").reset_index(drop=True)
 low,high,risk=build_candidate_signals(d)
 start=int((d.date.dt.year>=2015).to_numpy().argmax());end=len(d)-1
 out=Path(output);out.mkdir(parents=True,exist_ok=True);results={}
 for policy in POLICIES:
  results[policy]={}
  for slip in SLIPPAGE_BPS:
   sim=simulate(d,low,high,risk,limits,dividends,slip,start,end,high_exit_policy=policy)
   results[policy][f"{slip}bps"]={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
   pd.DataFrame(sim["trades"]).to_csv(out/f"{policy}_trades_{slip}bps.csv",index=False)
   pd.DataFrame(sim["equity_curve"]).to_csv(out/f"{policy}_equity_{slip}bps.csv",index=False)
   pd.DataFrame(annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL)).to_csv(out/f"{policy}_annual_{slip}bps.csv",index=False)
 summary={"symbol":"2330","classification":"EXPLORATORY_FIXED_HIGH_EXIT_POLICY_COMPARISON_NOT_OOF",
  "account_period":[d.date.iloc[start].date().isoformat(),d.date.iloc[end].date().isoformat()],
  "fixed_low_entry_rule":"unchanged 40.19 handoff transcription",
  "fixed_risk_rule":"SMA200 60-of-80 sessions and RSI14<40; 70-session cooldown",
  "policies":{
   "handoff_or":"immediate if distance SMA20<0 OR RSI14<58; otherwise exit if RSI14<48 or 14-session timeout",
   "and_sma20_rsi55":"immediate only if distance SMA20<0 AND RSI14<55; otherwise same RSI48/14-session confirmation",
   "ema20_rsi55":"immediate only if distance EMA20<0 AND RSI14<55; otherwise exit if EMA20 breaks with RSI<55, RSI<48, or 14-session timeout",
   "prior5low":"no immediate exit; after high signal exit on close below prior five-session low, RSI<48, or 14-session timeout"
  },
  "metrics_by_policy_and_cost":results,
  "warning":"Exploratory on previously inspected history. The policy definitions are frozen; no result is independent OOF. Any promising policy must be evaluated with annual expanding OOF before promotion."}
 (out/"tsmc_2330_high_exit_policy_comparison.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
 return summary
def main():
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--limits",required=True);p.add_argument("--dividends",required=True);p.add_argument("--output",default="artifacts/tsmc_2330_high_exit_policy_comparison");a=p.parse_args()
 print(json.dumps(run(a.features,a.limits,a.dividends,a.output),ensure_ascii=False,indent=2,default=float))
if __name__=="__main__":main()
