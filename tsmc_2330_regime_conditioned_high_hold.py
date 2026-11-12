"""Causal regime-conditioned high-signal hold hypothesis.

Compares baseline with two fixed, predeclared continuation gates:
(1) a strong long-term uptrend regime and (2) the same short-term pullback
condition without the long-term regime gate. All features are close-known.
Exploratory only; the rule is not promoted without expanding OOF.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from tsmc_2330_4019_replication import INITIAL_CAPITAL,SLIPPAGE_BPS,inputs,build_candidate_signals,simulate,annual_metrics

def gates(d):
 action=d.corporate_action_window.fillna(True).astype(bool)
 pullback=(d.distance_ema_20>0)&d.rsi14.between(50,65)&(d.plus_di14>d.minus_di14)
 strong=pullback&(d.distance_sma_200>=.25)
 return {
  "baseline":np.zeros(len(d),dtype=bool),
  "strong_long_trend_pullback":(strong&~action).fillna(False).to_numpy(bool),
  "general_ema20_directional_pullback":(pullback&~action).fillna(False).to_numpy(bool)
 }
def run(features,limits_path,dividends_path,output):
 d,limits,dividends=inputs(features,limits_path,dividends_path);d=d.sort_values("date").reset_index(drop=True)
 low,high,risk=build_candidate_signals(d);start=int((d.date.dt.year>=2015).to_numpy().argmax());end=len(d)-1
 out=Path(output);out.mkdir(parents=True,exist_ok=True);result={}
 for name,gate in gates(d).items():
  result[name]={}
  for slip in SLIPPAGE_BPS:
   sim=simulate(d,low,high,risk,limits,dividends,slip,start,end,high_hold_signal=gate)
   result[name][f"{slip}bps"]={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
   pd.DataFrame(sim["trades"]).to_csv(out/f"{name}_trades_{slip}bps.csv",index=False)
   pd.DataFrame(sim["equity_curve"]).to_csv(out/f"{name}_equity_{slip}bps.csv",index=False)
   pd.DataFrame(annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL)).to_csv(out/f"{name}_annual_{slip}bps.csv",index=False)
 summary={"symbol":"2330","classification":"EXPLORATORY_REGIME_CONDITIONED_HIGH_SIGNAL_HOLD_NOT_OOF",
  "account_period":[d.date.iloc[start].date().isoformat(),d.date.iloc[end].date().isoformat()],
  "fixed_rules":{
   "strong_long_trend_pullback":"distance SMA200>=25%, close above EMA20, RSI14 in [50,65], +DI>-DI",
   "general_ema20_directional_pullback":"close above EMA20, RSI14 in [50,65], +DI>-DI"
  },
  "timing":"all conditions known at signal-day close; any suppressed high exit is reconsidered only on a subsequent high signal or risk exit",
  "metrics_by_gate_and_cost":result,
  "warning":"The hypothesis was formulated after descriptive event analysis on inspected history. It is exploratory and potentially selection-biased; only expanding OOF can establish whether it generalizes."}
 (out/"tsmc_2330_regime_conditioned_high_hold.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
 return summary
def main():
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--limits",required=True);p.add_argument("--dividends",required=True);p.add_argument("--output",default="artifacts/tsmc_2330_regime_conditioned_high_hold");a=p.parse_args()
 print(json.dumps(run(a.features,a.limits,a.dividends,a.output),ensure_ascii=False,indent=2,default=float))
if __name__=="__main__":main()
