"""Out-of-time causal entry-filter replay in the real shared NT$100,000 top-ten account.

Thresholds were selected on the fixed-basket 2015-2021 training trade set and
are frozen here. The point-in-time daily top-ten universe is observed at close;
all orders execute no earlier than the next session open.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from top10_frozen_high_hold_transfer_audit import (
    read_events, build_limits, make_features, hold_gate,
    run_shared_capital_portfolio, annual_metrics,
)
from tsmc_2330_4019_replication import build_candidate_signals, INITIAL_CAPITAL, SLIPPAGE_BPS

FILTERS = {
    "baseline": None,
    "sma50_slope_ge_0.006062": ("sma50_slope_10", 0.006062),
    "distance_sma200_ge_0.073873": ("distance_sma_200", 0.073873),
    "sma50_and_sma200": (("sma50_slope_10", 0.006062), ("distance_sma_200", 0.073873)),
}

def entry_mask(d: pd.DataFrame, spec):
    if spec is None:
        return np.ones(len(d), dtype=bool)
    if isinstance(spec[0], tuple):
        mask=np.ones(len(d),dtype=bool)
        for col,threshold in spec:
            if col not in d:
                return np.zeros(len(d),dtype=bool)
            mask &= pd.to_numeric(d[col],errors="coerce").ge(threshold).fillna(False).to_numpy(bool)
        return mask
    col,threshold=spec
    if col not in d:
        return np.zeros(len(d),dtype=bool)
    return pd.to_numeric(d[col],errors="coerce").ge(threshold).fillna(False).to_numpy(bool)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--raw-bars",required=True)
    p.add_argument("--universe-file",required=True)
    p.add_argument("--actions",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    uni_full=pd.read_csv(a.universe_file,dtype={"symbol":str,"stock_id":str,"market":str})
    uni_full["date"]=pd.to_datetime(uni_full.date,errors="coerce").dt.strftime("%Y-%m-%d")
    uni_full["symbol"]=uni_full.symbol.astype(str).str.strip()
    # Strict OOT portfolio window; indicator history remains full-length for warmup.
    uni=uni_full[uni_full.date>="2022-01-01"].copy()
    if uni.empty: raise RuntimeError("No point-in-time top-ten dates in the 2022+ out-of-time test window")
    symbols=sorted(uni.symbol.unique())
    stock_map=uni.drop_duplicates("symbol").set_index("symbol").stock_id.astype(str).to_dict()
    market_map=uni.drop_duplicates("symbol").set_index("symbol").market.astype(str).to_dict()
    prepared={};blocked=[]
    for n,sym in enumerate(symbols,1):
        try:
            raw_full=pd.read_csv(Path(a.raw_bars)/f"{sym}.csv",parse_dates=["date"])
            raw_full=raw_full.sort_values("date").drop_duplicates("date").reset_index(drop=True)
            if raw_full.empty: raise RuntimeError("empty official raw history")
            ev,cashdiv,refs,event_limits,share_adjustments=read_events(
                a.actions,sym,stock_map[sym],market_map[sym])
            limits=build_limits(raw_full,refs,event_limits,sym)
            raw=raw_full.dropna(subset=["open","high","low","close"]).copy()
            raw=raw[raw.volume>0].reset_index(drop=True)
            if raw.empty: raise RuntimeError("no traded raw history")
            d=make_features(raw,ev)
            low,high,risk=build_candidate_signals(d)
            prepared[sym]={"d":d,"low":low,"high":high,"risk":risk,
                "hold":hold_gate(d),"limits":limits,"dividends":cashdiv,
                "share_adjustments":share_adjustments}
            if n%50==0: print(f"[PREPARED] {n}/{len(symbols)}",flush=True)
        except Exception as e:
            blocked.append({"symbol":sym,"reason":f"{type(e).__name__}: {e}"})
            print(f"[BLOCKED] {sym}: {e}",flush=True)
    if not prepared:
        raise RuntimeError("No constituent passed official-data validation")
    rows=[];report_filters={}
    for filter_name,spec in FILTERS.items():
        variant={}
        for sym,x in prepared.items():
            y=dict(x)
            y["low"]=x["low"] & entry_mask(x["d"],spec)
            variant[sym]=y
        report_filters[filter_name]={}
        for slip in SLIPPAGE_BPS:
            result=run_shared_capital_portfolio(
                variant,uni,slippage_bps=slip,initial_capital=INITIAL_CAPITAL)
            metrics={k:v for k,v in result.items() if k not in {"trades","equity_curve"}}
            rows.append({"filter":filter_name,"slippage_bps":slip,**metrics})
            report_filters[filter_name][f"{slip}bps"]=metrics
            if slip==10:
                pd.DataFrame(result.get("trades",[])).to_csv(out/f"trades_{filter_name}_10bps.csv",index=False)
                pd.DataFrame(result.get("equity_curve",[])).to_csv(out/f"equity_{filter_name}_10bps.csv",index=False)
                if result.get("equity_curve"):
                    pd.DataFrame(annual_metrics(result["equity_curve"],result.get("trades",[]),INITIAL_CAPITAL)).to_csv(
                        out/f"annual_{filter_name}_10bps.csv",index=False)
        print(f"[PORTFOLIO] {filter_name} complete",flush=True)
    pd.DataFrame(rows).to_csv(out/"shared_capital_entry_filter_metrics.csv",index=False)
    manifest_path=Path(a.universe_file).parent/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    report={
        "status":"COMPLETE",
        "classification":"FROZEN_TRAINING_THRESHOLD_OUT_OF_TIME_SHARED_CAPITAL_ENTRY_FILTER_REPLAY",
        "initial_capital":INITIAL_CAPITAL,
        "signal_timing":"signal at close t; fill at next session open",
        "test_universe":"daily official top ten by traded shares, point-in-time",
        "training_period":"2015-01-01 through 2021-12-31",
        "test_period":[str(uni.date.min()),str(uni.date.max())],
        "thresholds_frozen_from_training_only":{
            "sma50_slope_10_min":0.006062,
            "distance_sma_200_min":0.073873,
        },
        "costs":{"commission_per_side":0.001425,"minimum_commission_ntd":20,
                 "sell_tax":0.003,"slippage_bps":list(SLIPPAGE_BPS)},
        "universe_manifest":manifest,
        "symbols_expected":len(symbols),"symbols_prepared":len(prepared),
        "blocked_symbols":blocked,
        "filters":report_filters,
        "interpretation":"This is the actual one-position shared NT$100,000 account. A filter is not approved unless its out-of-time portfolio results improve risk without unacceptable return/trade-frequency loss. No threshold is reselected on 2022+ data.",
    }
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    print(json.dumps({"symbols_expected":len(symbols),"symbols_prepared":len(prepared),
        "blocked":len(blocked),"baseline_10bps":report_filters["baseline"].get("10bps"),
        "filters_10bps":{k:v.get("10bps") for k,v in report_filters.items() if k!="baseline"}},
        ensure_ascii=False,indent=2,default=float))

if __name__=="__main__": main()
