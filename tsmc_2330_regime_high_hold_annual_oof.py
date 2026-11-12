"""Annual expanding OOF for regime-conditioned high-signal hold policy.

Each validation year chooses between the frozen baseline and one frozen strong
long-term trend pullback gate using only the prefix through the prior year.
The trading account is continuous from 2015; no annual reset or forced sale.
Because the gate family was formulated after descriptive inspection of known
history, this is honest expanding OOF but not pristine independent blind OOS.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from tsmc_2330_4019_replication import INITIAL_CAPITAL,SLIPPAGE_BPS,inputs,build_candidate_signals,simulate,annual_metrics

WIN_FLOOR=0.869565
MIN_TRAIN_TRADES=5
POLICIES=("baseline","strong_long_trend_pullback")

def gate_signal(d):
 action=d.corporate_action_window.fillna(True).astype(bool)
 g=(d.distance_sma_200>=.25)&(d.distance_ema_20>0)&d.rsi14.between(50,65)&(d.plus_di14>d.minus_di14)&~action
 return g.fillna(False).to_numpy(bool)

def eligible(m):
 return (m["completed_trades"]>=MIN_TRAIN_TRADES and m["win_rate_completed_only"] is not None
  and m["win_rate_completed_only"]>=WIN_FLOOR and m["cagr_mark_to_market"] is not None)

def select_policy(metrics):
 choices=[]
 for name,m in metrics.items():
  if eligible(m):
   choices.append((name,(m["cagr_mark_to_market"],m["win_rate_completed_only"],-abs(m["max_drawdown"] or -1))))
 if not choices:return "baseline"
 return max(choices,key=lambda x:x[1])[0]

def run(features,limits_path,dividends_path,output):
 d,limits,dividends=inputs(features,limits_path,dividends_path);d=d.sort_values("date").reset_index(drop=True)
 low,high,risk=build_candidate_signals(d);gate=gate_signal(d)
 years=sorted(int(y) for y in d.date.dt.year.unique())
 if 2015 not in years:raise RuntimeError("Validation period must begin in 2015")
 selections={};folds=[]
 for year in [y for y in years if y>=2015]:
  prior=np.flatnonzero((d.date.dt.year<year).to_numpy())
  if not len(prior):
   selections[year]="baseline";continue
  end_i=int(prior[-1]);training={}
  for name,sig in {"baseline":np.zeros(len(d),dtype=bool),"strong_long_trend_pullback":gate}.items():
   m=simulate(d,low,high,risk,limits,dividends,10,0,end_i,high_hold_signal=sig)
   training[name]={k:v for k,v in m.items() if k not in {"trades","equity_curve"}}
  chosen=select_policy(training);selections[year]=chosen
  folds.append({"validation_year":year,"training_start":d.date.iloc[0].date().isoformat(),
   "training_end":d.date.iloc[end_i].date().isoformat(),"selected_policy":chosen,
   "win_rate_floor":WIN_FLOOR,"minimum_training_trades":MIN_TRAIN_TRADES,
   "frozen_before_validation_year":True,"training_metrics":training})
 # Stitch yearly frozen policies into one signal vector. Only validation years
 # use their selected policy; pre-2015 is training only.
 validation_gate=np.zeros(len(d),dtype=bool)
 for i,row in d.iterrows():
  year=int(row.date.year)
  if year>=2015 and selections.get(year)=="strong_long_trend_pullback":
   validation_gate[i]=gate[i]
 idx=np.flatnonzero((d.date.dt.year>=2015).to_numpy());start=int(idx[0]);end=int(idx[-1])
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 metrics_by_cost={};trades_by_cost={};equity_by_cost={};annual_by_cost={}
 for slip in SLIPPAGE_BPS:
  sim=simulate(d,low,high,risk,limits,dividends,slip,start,end,high_hold_signal=validation_gate)
  metrics_by_cost[f"{slip}bps"]={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
  trades_by_cost[f"{slip}bps"]=sim["trades"];equity_by_cost[f"{slip}bps"]=sim["equity_curve"]
  annual_by_cost[f"{slip}bps"]=annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL)
  pd.DataFrame(sim["trades"]).to_csv(out/f"trades_{slip}bps.csv",index=False)
  pd.DataFrame(sim["equity_curve"]).to_csv(out/f"equity_{slip}bps.csv",index=False)
  pd.DataFrame(annual_by_cost[f"{slip}bps"]).to_csv(out/f"annual_{slip}bps.csv",index=False)
 summary={"symbol":"2330","classification":"EXPANDING_ANNUAL_OOF_REGIME_HIGH_HOLD_FAMILY_NOT_PRISTINE_BLIND_OOS",
  "data_period":[d.date.min().date().isoformat(),d.date.max().date().isoformat()],
  "validation_period":[d.date.iloc[start].date().isoformat(),d.date.iloc[end].date().isoformat()],
  "protocol":"Each year selects between baseline and one fixed regime gate using only 2010 through the previous year; one continuous 2015+ account; no year-end liquidation; final position mark-to-market.",
  "selection_objective":"Maximize training-prefix 10bps CAGR subject to >=5 completed trades and >=86.9565% win rate; if neither qualifies, use baseline.",
  "win_rate_floor":WIN_FLOOR,"minimum_training_trades":MIN_TRAIN_TRADES,
  "fixed_regime_gate":"distance SMA200>=25%, close above EMA20, RSI14 50-65, +DI>-DI",
  "annual_frozen_selections":folds,"selected_policy_by_year":{str(k):v for k,v in selections.items()},
  "continuous_oof_metrics_by_cost":metrics_by_cost,"annual_oof_metrics_by_cost":annual_by_cost,
  "target_status":"UNMET unless CAGR>=50% and win rate>=86.9565% at every cost; MDD<25% tracked separately.",
  "limitations":["The regime gate was formulated after inspecting historical high-signal outcomes; OOF controls temporal leakage but not research-family selection bias.","2010-2026 market history was previously inspected; do not call this pristine blind validation.","Historical daily OHLC cannot reproduce queue priority or true spread." ]}
 (out/"tsmc_2330_regime_high_hold_annual_oof.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
 pd.DataFrame([{"year":k,"policy":v} for k,v in selections.items()]).to_csv(out/"selected_policy_by_year.csv",index=False)
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--limits",required=True);p.add_argument("--dividends",required=True);p.add_argument("--output",default="artifacts/tsmc_2330_regime_high_hold_annual_oof");a=p.parse_args()
 x=run(a.features,a.limits,a.dividends,a.output)
 print(json.dumps({k:v for k,v in x.items() if k not in {"annual_frozen_selections"}},ensure_ascii=False,indent=2,default=float))
if __name__=="__main__":main()
