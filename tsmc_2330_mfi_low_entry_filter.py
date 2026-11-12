"""Fixed MFI confirmation filter for the frozen low-point entry signal."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from tsmc_2330_4019_replication import INITIAL_CAPITAL,SLIPPAGE_BPS,inputs,build_candidate_signals,simulate,annual_metrics

def run(features,limits_path,dividends_path,output):
 d,limits,dividends=inputs(features,limits_path,dividends_path);d=d.sort_values("date").reset_index(drop=True)
 low,high,risk=build_candidate_signals(d)
 mfi_ok=(d.mfi14>=40).fillna(False).to_numpy(bool)
 streams={"baseline_original_low_signal":low,"mfi14_at_least_40":low&mfi_ok}
 start=int((d.date.dt.year>=2015).to_numpy().argmax());end=len(d)-1
 out=Path(output);out.mkdir(parents=True,exist_ok=True);metrics={};annual={}
 for name,signal in streams.items():
  metrics[name]={};annual[name]={}
  for slip in SLIPPAGE_BPS:
   sim=simulate(d,signal,high,risk,limits,dividends,slip,start,end)
   metrics[name][f"{slip}bps"]={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
   annual[name][f"{slip}bps"]=annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL)
   pd.DataFrame(sim["trades"]).to_csv(out/f"{name}_trades_{slip}bps.csv",index=False)
   pd.DataFrame(sim["equity_curve"]).to_csv(out/f"{name}_equity_{slip}bps.csv",index=False)
   pd.DataFrame(annual[name][f"{slip}bps"]).to_csv(out/f"{name}_annual_{slip}bps.csv",index=False)
 summary={"symbol":"2330","classification":"EXPLORATORY_FIXED_MFI_LOW_ENTRY_FILTER_NOT_OOF",
  "account_period":[d.date.iloc[start].date().isoformat(),d.date.iloc[end].date().isoformat()],
  "filter":"Require MFI14>=40 on the low-signal close; missing MFI fails closed. All other entry/exit rules unchanged.",
  "metrics_by_stream_and_cost":metrics,"annual_metrics_by_stream_and_cost":annual,
  "warning":"Exploratory on previously inspected history; threshold is fixed at the conventional MFI neutral boundary and is not an independent OOF result."}
 (out/"tsmc_2330_mfi_low_entry_filter.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
 return summary
def main():
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--limits",required=True);p.add_argument("--dividends",required=True);p.add_argument("--output",default="artifacts/tsmc_2330_mfi_low_entry_filter");a=p.parse_args()
 print(json.dumps(run(a.features,a.limits,a.dividends,a.output),ensure_ascii=False,indent=2,default=float))
if __name__=="__main__":main()

# Fixed conventional MFI threshold; no grid search.
