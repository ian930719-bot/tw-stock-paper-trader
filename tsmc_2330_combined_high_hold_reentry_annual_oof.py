"""Combined expanding annual OOF for fixed high-hold and EMA20-reclaim re-entry.

Four predeclared configurations: baseline, regime high-hold only, EMA20 reclaim
re-entry only, and both. Each validation year selects using only the prefix
through the prior year. Single continuous account from 2015; no year reset.
Research-family design used previously inspected history, so this is not
pristine independent blind OOS.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from tsmc_2330_4019_replication import INITIAL_CAPITAL,SLIPPAGE_BPS,inputs,build_candidate_signals,simulate,annual_metrics

WIN_FLOOR=.869565
MIN_TRAIN_TRADES=5
COOLDOWN=5
CONFIGS={
 "baseline":(False,False),
 "strong_trend_high_hold":(True,False),
 "ema20_reclaim_reentry":(False,True),
 "combined_high_hold_and_reentry":(True,True),
}
def gate_signal(d):
 action=d.corporate_action_window.fillna(True).astype(bool)
 x=(d.distance_sma_200>=.25)&(d.distance_ema_20>0)&d.rsi14.between(50,65)&(d.plus_di14>d.minus_di14)&~action
 return x.fillna(False).to_numpy(bool)
def reentry_signal(d):
 cross=(d.distance_ema_20.shift(1)<=0)&(d.distance_ema_20>0)
 x=(cross&d.rsi14.between(40,55)&(d.distance_sma_50>0)&(d.sma50_slope_10>0)&(d.adx14>=18)&(d.plus_di14>d.minus_di14)&~d.corporate_action_window.fillna(True).astype(bool))
 return x.fillna(False).to_numpy(bool)
def eligible(m):
 return m["completed_trades"]>=MIN_TRAIN_TRADES and m["win_rate_completed_only"] is not None and m["win_rate_completed_only"]>=WIN_FLOOR and m["cagr_mark_to_market"] is not None
def select_policy(metrics):
 choices=[]
 for name,m in metrics.items():
  if eligible(m):choices.append((name,(m["cagr_mark_to_market"],m["win_rate_completed_only"],-abs(m["max_drawdown"] or -1))))
 return max(choices,key=lambda z:z[1])[0] if choices else "baseline"
def run(features,limits_path,dividends_path,output):
 d,limits,dividends=inputs(features,limits_path,dividends_path);d=d.sort_values("date").reset_index(drop=True)
 low,high,risk=build_candidate_signals(d);hold=gate_signal(d);reentry=reentry_signal(d)
 years=sorted(int(y) for y in d.date.dt.year.unique());selections={};folds=[]
 for year in [y for y in years if y>=2015]:
  prior=np.flatnonzero((d.date.dt.year<year).to_numpy())
  if not len(prior):selections[year]="baseline";continue
  end_i=int(prior[-1]);training={}
  for name,(use_hold,use_reentry) in CONFIGS.items():
   sim=simulate(d,low,high,risk,limits,dividends,10,0,end_i,
    high_hold_signal=hold if use_hold else np.zeros(len(d),dtype=bool),
    reentry_signal=reentry if use_reentry else np.zeros(len(d),dtype=bool),
    reentry_cooldown=COOLDOWN)
   training[name]={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
  chosen=select_policy(training);selections[year]=chosen
  folds.append({"validation_year":year,"training_end":d.date.iloc[end_i].date().isoformat(),
   "selected_policy":chosen,"training_metrics":training,
   "frozen_before_validation_year":True,"win_rate_floor":WIN_FLOOR,
   "minimum_training_trades":MIN_TRAIN_TRADES})
 val_hold=np.zeros(len(d),dtype=bool);val_reentry=np.zeros(len(d),dtype=bool)
 for i,row in d.iterrows():
  name=selections.get(int(row.date.year),"baseline")
  use_hold,use_re=CONFIGS[name]
  if int(row.date.year)>=2015:
   if use_hold:val_hold[i]=hold[i]
   if use_re:val_reentry[i]=reentry[i]
 idx=np.flatnonzero((d.date.dt.year>=2015).to_numpy());start=int(idx[0]);end=int(idx[-1])
 out=Path(output);out.mkdir(parents=True,exist_ok=True);metrics={};annual={}
 for slip in SLIPPAGE_BPS:
  sim=simulate(d,low,high,risk,limits,dividends,slip,start,end,
   high_hold_signal=val_hold,reentry_signal=val_reentry,reentry_cooldown=COOLDOWN)
  metrics[f"{slip}bps"]={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
  annual[f"{slip}bps"]=annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL)
  pd.DataFrame(sim["trades"]).to_csv(out/f"trades_{slip}bps.csv",index=False)
  pd.DataFrame(sim["equity_curve"]).to_csv(out/f"equity_{slip}bps.csv",index=False)
  pd.DataFrame(annual[f"{slip}bps"]).to_csv(out/f"annual_{slip}bps.csv",index=False)
 summary={"symbol":"2330","classification":"EXPANDING_ANNUAL_OOF_COMBINED_REGIME_HOLD_AND_EMA20_REENTRY_NOT_PRISTINE_BLIND_OOS",
  "data_period":[d.date.min().date().isoformat(),d.date.max().date().isoformat()],
  "validation_period":[d.date.iloc[start].date().isoformat(),d.date.iloc[end].date().isoformat()],
  "protocol":"Each year selects among four fixed configurations using only 2010 through prior year; continuous 2015+ account; no annual reset; mark-to-market final position.",
  "selection_objective":"Maximize training-prefix 10bps CAGR subject to at least five closed trades and win rate >=86.9565%; otherwise baseline.",
  "fixed_configurations":CONFIGS,"annual_frozen_selections":folds,
  "selected_policy_by_year":{str(k):v for k,v in selections.items()},
  "continuous_oof_metrics_by_cost":metrics,"annual_oof_metrics_by_cost":annual,
  "target_status":"UNMET unless CAGR>=50% and win rate>=86.9565% at all cost scenarios; MDD<25% separately tracked.",
  "limitations":["Both rule families were formulated after inspecting historical data; expanding OOF prevents direct future leakage but not research-family selection bias.","2010-2026 data was previously inspected; this is not pristine independent blind OOS.","Historical daily OHLC cannot model queue priority or true spread."]}
 (out/"tsmc_2330_combined_high_hold_reentry_annual_oof.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
 pd.DataFrame([{"year":k,"policy":v} for k,v in selections.items()]).to_csv(out/"selected_policy_by_year.csv",index=False)
 return summary
def main():
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--limits",required=True);p.add_argument("--dividends",required=True);p.add_argument("--output",default="artifacts/tsmc_2330_combined_high_hold_reentry_oof");a=p.parse_args()
 x=run(a.features,a.limits,a.dividends,a.output);print(json.dumps({k:v for k,v in x.items() if k!="annual_frozen_selections"},ensure_ascii=False,indent=2,default=float))
if __name__=="__main__":main()
