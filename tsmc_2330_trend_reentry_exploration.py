"""Exploratory fixed-rule trend re-entry study on the frozen TSMC 2330 dataset.

All re-entry conditions are close-known at t and execute no earlier than t+1 open.
This is a diagnostic experiment on historically inspected data, not OOF.
No candidate is promoted based on this run; annual expanding OOF is required next.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from tsmc_2330_4019_replication import (
    INITIAL_CAPITAL, SLIPPAGE_BPS, inputs, build_candidate_signals, simulate, annual_metrics
)

def reentry_candidates(d: pd.DataFrame) -> dict[str, tuple[np.ndarray,int]]:
    required={"raw_high","raw_close","distance_ema_20","distance_ema_50","ema20_slope_5",
              "rsi14","macd_hist_pct","macd_hist_change_3","adx14","corporate_action_window"}
    missing=sorted(required-set(d.columns))
    if missing: raise ValueError(f"Missing re-entry feature columns: {missing}")
    action=d.corporate_action_window.fillna(True).astype(bool)
    close=d.raw_close
    prev_high=d.raw_high.shift(1)
    ema20=(d.distance_ema_20>0)
    ema50=(d.distance_ema_50>0)
    rsi50=(d.rsi14>=50)
    macdpos=(d.macd_hist_pct>0)
    slope=(d.ema20_slope_5>0)
    candidates={
      "ema20_rsi50_macdpositive_cd10": (ema20 & rsi50 & macdpos & slope,10),
      "prevhigh_breakout_ema20_rsi50_macd_cd10": (ema20 & rsi50 & macdpos & slope & (close>prev_high),10),
      "ema20_rsi50_macd_acceleration_ema50_cd10": (ema20 & ema50 & rsi50 & (d.macd_hist_change_3>0) & slope & (d.adx14>=18),10),
      "rsi50_cross_ema20_macd_cd10": (ema20 & (d.rsi14.shift(1)<50) & rsi50 & macdpos & slope,10),
      "ema20_rsi50_macdpositive_cd20": (ema20 & rsi50 & macdpos & slope,20),
    }
    return {k:(v.fillna(False).to_numpy(bool)&~action.to_numpy(bool),cd) for k,(v,cd) in candidates.items()}

def run(features_path,limits_path,dividends_path,output):
    d,limits,dividends=inputs(features_path,limits_path,dividends_path)
    d=d.sort_values("date").reset_index(drop=True)
    low,high,risk=build_candidate_signals(d)
    start_i=int(np.flatnonzero((d.date.dt.year>=2015).to_numpy())[0]); end_i=len(d)-1
    signals=reentry_candidates(d)
    signals={"baseline_no_trend_reentry":(np.zeros(len(d),dtype=bool),0),**signals}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    results={}
    for name,(sig,cooldown) in signals.items():
      results[name]={}
      for slip in SLIPPAGE_BPS:
        sim=simulate(d,low,high,risk,limits,dividends,slip,start_i,end_i,
                     reentry_signal=sig,reentry_cooldown=cooldown)
        metrics={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
        results[name][f"{slip}bps"]=metrics
        pd.DataFrame(sim["trades"]).to_csv(out/f"{name}_trades_{slip}bps.csv",index=False)
        pd.DataFrame(sim["equity_curve"]).to_csv(out/f"{name}_equity_{slip}bps.csv",index=False)
        pd.DataFrame(annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL)).to_csv(
          out/f"{name}_annual_{slip}bps.csv",index=False)
    summary={
      "symbol":"2330","classification":"EXPLORATORY_FIXED_RULE_REENTRY_DIAGNOSTIC_NOT_OOF",
      "data_period":[d.date.min().date().isoformat(),d.date.max().date().isoformat()],
      "account_period":[d.date.iloc[start_i].date().isoformat(),d.date.iloc[end_i].date().isoformat()],
      "signal_execution":"all indicators known at close t; next trading session open execution",
      "base_rule":"frozen 40.19 handoff rule transcription currently reproduces 35.82% CAGR at 10bps; unresolved discrepancy remains",
      "reentry_rule_families":{k:{"cooldown_sessions":v[1]} for k,v in signals.items()},
      "metrics_by_candidate_and_cost":results,
      "warning":"All history has been inspected. This is exploratory only; no result is independent OOF or a deployable strategy. Only candidates preserving win-rate floor may proceed to expanding OOF."
    }
    (out/"tsmc_2330_trend_reentry_exploration.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    return summary

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--features",required=True);p.add_argument("--limits",required=True)
 p.add_argument("--dividends",required=True);p.add_argument("--output",default="artifacts/tsmc_2330_trend_reentry_exploration")
 a=p.parse_args();print(json.dumps(run(a.features,a.limits,a.dividends,a.output),ensure_ascii=False,indent=2,default=float))
if __name__=="__main__":main()
