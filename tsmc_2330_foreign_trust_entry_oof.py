"""Full execution backtest of predeclared foreign/trust flow entry gates.

The candidate universe is fixed in advance. Each validation year chooses a
policy using only prior-year history, subject to the existing win-rate floor.
Institutional data is used only as an after-close filter for next-session entry.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from tsmc_2330_4019_replication import (
 INITIAL_CAPITAL,SLIPPAGE_BPS,annual_metrics,build_candidate_signals,inputs,simulate
)
WIN_FLOOR=0.8695652173913043
MIN_TRAIN_TRADES=5
VALIDATION_START_YEAR=2015
FLOW_COLS=["foreign_net_lots","foreign_net_5d_lots","foreign_net_20d_lots",
 "trust_net_lots","trust_net_5d_lots","trust_net_20d_lots",
 "institutional_data_available"]

def eligible(m):
 w,c=m.get("win_rate_completed_only"),m.get("cagr_mark_to_market")
 return int(m.get("completed_trades",0))>=MIN_TRAIN_TRADES and w is not None and np.isfinite(w) and w>=WIN_FLOOR and c is not None and np.isfinite(c)

def choose(metrics):
 pool=[(k,v) for k,v in metrics.items() if eligible(v)]
 if not pool:return "baseline"
 return max(pool,key=lambda kv:(kv[1]["cagr_mark_to_market"],kv[1]["win_rate_completed_only"],-abs(kv[1].get("max_drawdown") or -1)))[0]

def build_policies(d,base_low,base_high,hold):
 available=d.institutional_data_available.fillna(0).eq(1).to_numpy(bool)
 f5=d.foreign_net_5d_lots.gt(0).fillna(False).to_numpy(bool)
 f20=d.foreign_net_20d_lots.gt(0).fillna(False).to_numpy(bool)
 t5=d.trust_net_5d_lots.gt(0).fillna(False).to_numpy(bool)
 t20=d.trust_net_20d_lots.gt(0).fillna(False).to_numpy(bool)
 positive_gates={
  "baseline":np.ones(len(d),dtype=bool),
  "foreign_5d_positive":(~available)|f5,
  "foreign_20d_positive":(~available)|f20,
  "trust_5d_positive":(~available)|t5,
  "trust_20d_positive":(~available)|t20,
  "foreign_and_trust_5d_positive":(~available)|(f5&t5),
  "foreign_or_trust_5d_positive":(~available)|(f5|t5),
 }
 negative_gates={
  "foreign_5d_negative":(~available)|d.foreign_net_5d_lots.lt(0).fillna(False).to_numpy(bool),
  "foreign_20d_negative":(~available)|d.foreign_net_20d_lots.lt(0).fillna(False).to_numpy(bool),
  "trust_5d_negative":(~available)|d.trust_net_5d_lots.lt(0).fillna(False).to_numpy(bool),
  "trust_20d_negative":(~available)|d.trust_net_20d_lots.lt(0).fillna(False).to_numpy(bool),
  "foreign_and_trust_5d_negative":(~available)|(d.foreign_net_5d_lots.lt(0).fillna(False).to_numpy(bool)&d.trust_net_5d_lots.lt(0).fillna(False).to_numpy(bool)),
  "foreign_or_trust_5d_negative":(~available)|(d.foreign_net_5d_lots.lt(0).fillna(False).to_numpy(bool)|d.trust_net_5d_lots.lt(0).fillna(False).to_numpy(bool)),
 }
 policies={}
 zero=np.zeros(len(d),dtype=bool)
 for name,gate in positive_gates.items():
  policies[name]=(base_low&gate,base_high,zero)
  policies[name+"_high_hold"]=(base_low&gate,base_high,hold)
  if name!="baseline":
   # Flow gates high-exit suppression without changing the entry signal.
   policies["high_hold_"+name]=(base_low,base_high,hold&gate)
 for name,gate in negative_gates.items():
  # Require confirmed net selling before acting on the original high-exit signal.
  # Missing institutional data falls back to the original high-exit signal.
  policies["exit_"+name]=(base_low,base_high&gate,zero)
 return policies

def run(features_path,limits_path,dividends_path,output_dir):
 d,limits,dividends=inputs(features_path,limits_path,dividends_path)
 d=d.sort_values("date").reset_index(drop=True)
 missing=[c for c in FLOW_COLS if c not in d.columns]
 if missing:raise ValueError(f"Feature dataset missing required foreign/trust fields: {missing}")
 # The preserved causal feature artifact already contains same-day flow features.
 # Do not merge a second copy: this would create ambiguous *_x/*_y columns.
 base_low,base_high,risk=build_candidate_signals(d)
 action=d.corporate_action_window.fillna(True).astype(bool)
 hold=((d.distance_sma_200>=.25)&(d.distance_ema_20>0)&d.rsi14.between(50,65)&
       (d.plus_di14>d.minus_di14)&~action).fillna(False).to_numpy(bool)
 policies=build_policies(d,base_low,base_high,hold)
 years=sorted(int(y) for y in d.date.dt.year.unique())
 selections={};folds=[]
 for year in [y for y in years if y>=VALIDATION_START_YEAR]:
  prior=np.flatnonzero((d.date.dt.year<year).to_numpy())
  if not len(prior):selections[year]="baseline";continue
  train_end=int(prior[-1]);train_metrics={}
  for name,(low,high_signal,hold_signal) in policies.items():
   sim=simulate(d,low,high_signal,risk,limits,dividends,10,0,train_end,high_hold_signal=hold_signal)
   train_metrics[name]={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
  selected=choose(train_metrics);selections[year]=selected
  folds.append({"validation_year":year,"training_end":d.date.iloc[train_end].date().isoformat(),
   "selected_policy":selected,"frozen_before_validation_year":True,
   "training_metrics":train_metrics})
 zero=np.zeros(len(d),dtype=bool);oof_low=zero.copy();oof_high=zero.copy();oof_hold=zero.copy()
 for i,row in d.iterrows():
  low,high_signal,hold_signal=policies[selections.get(int(row.date.year),"baseline")]
  oof_low[i]=low[i];oof_high[i]=high_signal[i];oof_hold[i]=hold_signal[i]
 start_idx=int(np.flatnonzero((d.date.dt.year>=VALIDATION_START_YEAR).to_numpy())[0]);end_idx=len(d)-1
 out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
 metrics_by_cost={};annual_by_cost={}
 for slip in SLIPPAGE_BPS:
  sim=simulate(d,oof_low,oof_high,risk,limits,dividends,slip,start_idx,end_idx,high_hold_signal=oof_hold)
  metrics_by_cost[f"{slip}bps"]={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
  annual_by_cost[f"{slip}bps"]=annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL)
  pd.DataFrame(sim["trades"]).to_csv(out/f"trades_{slip}bps.csv",index=False)
  pd.DataFrame(sim["equity_curve"]).to_csv(out/f"equity_{slip}bps.csv",index=False)
  pd.DataFrame(annual_by_cost[f"{slip}bps"]).to_csv(out/f"annual_{slip}bps.csv",index=False)
 result={"symbol":"2330","classification":"EXECUTABLE_CAUSAL_FLOW_FILTER_ANNUAL_OOF_BACKTEST",
  "data_period":[d.date.min().date().isoformat(),d.date.max().date().isoformat()],
  "validation_period":[d.date.iloc[start_idx].date().isoformat(),d.date.iloc[end_idx].date().isoformat()],
  "signal_timing":"institutional flow known after confirmation-day close; filtered entry executes no earlier than next session open",
  "data_source":"FinMind TaiwanStockInstitutionalInvestorsBuySellWide; data availability explicitly tracked",
  "candidate_gates":{"foreign_5d_positive":"foreign cumulative net lots over 5 sessions >0",
   "foreign_20d_positive":"foreign cumulative net lots over 20 sessions >0",
   "trust_5d_positive":"investment trust cumulative net lots over 5 sessions >0",
   "trust_20d_positive":"investment trust cumulative net lots over 20 sessions >0",
   "foreign_and_trust_5d_positive":"both foreign and trust 5-session net lots >0",
   "foreign_or_trust_5d_positive":"either foreign or trust 5-session net lots >0",
   "exit_confirmation":"Require negative foreign/trust 5-session or 20-session flow to confirm an original high-exit signal; missing flow falls back to the original exit.",
   "high_hold":"existing predeclared regime-conditioned high-hold rule"},
  "excluded":["dealers","foreign dealer proprietary","broker branches","aggregate three-institution flow"],
  "annual_selection":"For each validation year, select among fixed gate policies with/without high-hold using only prior years; maximize 10bps training CAGR subject to >=5 completed trades and >=86.9565% completed-trade win rate; otherwise baseline.",
  "minimum_training_trades":MIN_TRAIN_TRADES,"win_rate_floor":WIN_FLOOR,
  "selected_policy_by_year":{str(k):v for k,v in selections.items()},
  "annual_frozen_selections":folds,"continuous_oof_metrics_by_cost":metrics_by_cost,
  "annual_oof_metrics_by_cost":annual_by_cost,
  "target_status":"UNMET unless 10bps CAGR>=50% and completed-trade win rate>=86.9565%; MDD<=25% tracked separately.",
  "limitations":["Event-flow study is not evidence of a profitable trading strategy; this execution backtest is the required test.",
   "FinMind institutional history starts 2012-05-02; missing flow is not imputed as zero, and every flow-gated candidate falls back to an unfiltered entry when flow is unavailable.",
   "Annual OOF controls temporal selection but research-family selection bias remains; 2010-2026 was previously inspected.",
   "2026 is partial year; daily OHLC cannot reproduce queue priority or intraday fills."]}
 (out/"tsmc_2330_foreign_trust_entry_oof.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
 pd.DataFrame([{"year":y,"selected_policy":p} for y,p in selections.items()]).to_csv(out/"selected_policy_by_year.csv",index=False)
 return result

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--limits",required=True);p.add_argument("--dividends",required=True);p.add_argument("--output",default="artifacts/tsmc_2330_foreign_trust_entry_oof")
 a=p.parse_args();print(json.dumps(run(a.features,a.limits,a.dividends,a.output),ensure_ascii=False,indent=2,default=float))
