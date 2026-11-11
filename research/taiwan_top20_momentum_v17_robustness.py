"""V17: transaction-cost and rebalance-frequency sensitivity for V16 breadth-filtered momentum."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import taiwan_top20_momentum_trend_breadth_v16 as v16

def run(bars_dir,universe_json,events_csv,outdir):
    d=v16.load_panel(bars_dir,universe_json,events_csv)
    rows=[]
    base=(v16.BUY_FEE,v16.SELL_FEE_TAX,v16.SLIPPAGE,v16.REBALANCE_SESSIONS)
    try:
        for frequency in (10,20,40):
            v16.REBALANCE_SESSIONS=frequency
            for multiplier in (1,2,3):
                v16.BUY_FEE=base[0]*multiplier
                v16.SELL_FEE_TAX=base[1]*multiplier
                v16.SLIPPAGE=base[2]*multiplier
                for period,(start,end) in v16.SPLITS.items():
                    metrics,_=v16.simulate(d,start,end,"mom20_breadth60")
                    rows.append({"frequency_sessions":frequency,"cost_multiplier":multiplier,"period":period,**metrics})
    finally:
        v16.BUY_FEE,v16.SELL_FEE_TAX,v16.SLIPPAGE,v16.REBALANCE_SESSIONS=base
    table=pd.DataFrame(rows)
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/"v17_cost_frequency_sensitivity.csv",index=False)
    summary={"version":"V17","tested_rule":"V16 20-session momentum Top 5, only when at least 60% of eligible Top-20 stocks close above MA200.",
      "frequency_sessions":[10,20,40],"cost_multipliers":[1,2,3],
      "base_costs":{"buy_fee":base[0],"sell_fee_and_tax":base[1],"slippage_each_side":base[2]},
      "periods":{k:{"start":v[0],"end":v[1]} for k,v in v16.SPLITS.items()},
      "results":rows,
      "limitations":["Same historical data and universe limitations as V16.","2022-2026 was previously inspected and is not a pristine holdout.","Cost multipliers are stress tests, not estimates of actual future broker charges.","No limit-up/down queues, partial fills, market impact or full corporate-action accounting."]}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--bars-dir",default="data/taiwan_history/bars")
    p.add_argument("--universe-json",default="artifacts/top20/extended_oos_report.json")
    p.add_argument("--events-csv",default="data/taiwan_history/corporate_actions/events.csv")
    p.add_argument("--output",default="artifacts/top20_v17_robustness")
    a=p.parse_args();run(a.bars_dir,a.universe_json,a.events_csv,a.output)
