"""Predeclared TSMC 2330 rule-family search with chronological selection and cost stress.

Signals use adjusted causal features at close t and execute on next session open.
Candidate parameters are selected only on 2010-2015 research data; 2016-2020
validation and 2021+ historical test are report-only. No guarantee is implied.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
from tsmc_2330_execution_backtest import _simulate_period, SLIPPAGE_BPS, INITIAL_CAPITAL

BUY_FEE = 0.001425
SELL_TAX = 0.003
MIN_FEE = 20
RESEARCH = ("2010-01-04", "2015-12-31")
VALIDATION = ("2016-01-04", "2020-12-31")
TEST = ("2021-01-01", None)
WIN_FLOOR = 0.8695652173913043

def inputs(features_path, limits_path, dividends_path):
    d = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    required = {"date","raw_open","raw_high","raw_low","raw_close","return_1","relative_volume20",
                "stoch_k14","close_location","obv_flow_ratio_5","intraday_range_pct",
                "distance_sma_200","macd_hist_change_3","sma50_slope_10","corporate_action_window"}
    missing = sorted(required-set(d.columns))
    if missing: raise ValueError(f"Missing feature columns: {missing}")
    lim = pd.read_csv(limits_path, dtype={"stock_id":str})
    lim = lim[lim.stock_id.astype(str).str.strip()=="2330"].copy()
    lim["date"] = pd.to_datetime(lim.date, errors="coerce").dt.normalize()
    lookup = {r.date.strftime("%Y-%m-%d"):(float(r.limit_up),float(r.limit_down))
              for r in lim.itertuples(index=False)
              if pd.notna(r.limit_up) and pd.notna(r.limit_down)}
    if len(lookup)/len(d) < .98: raise RuntimeError("Price-limit coverage below 98%")
    div = pd.read_csv(dividends_path)
    div = div[div.stock_or_cache_dividend.astype(str).str.strip()=="息"].copy()
    div["date"] = pd.to_datetime(div.date, errors="coerce").dt.normalize()
    div["stock_and_cache_dividend"] = pd.to_numeric(div.stock_and_cache_dividend, errors="coerce")
    div = div.dropna(subset=["date","stock_and_cache_dividend"])
    dividends = {r.date.strftime("%Y-%m-%d"):float(r.stock_and_cache_dividend) for r in div.itertuples()}
    return d, lookup, dividends

def make_signals(d, low_ret, low_rv, high_loc, high_dist, high_ret=.01, high_range=.016):
    action = d.corporate_action_window.fillna(True).astype(bool)
    low = ((d.relative_volume20 >= low_rv) & (d.return_1 <= low_ret) &
           (d.stoch_k14 <= 30) & (d.close_location <= .25) &
           (d.obv_flow_ratio_5 <= 0))
    high = ((d.return_1 >= high_ret) & (d.close_location >= high_loc) &
            (d.intraday_range_pct >= high_range) & (d.distance_sma_200 >= high_dist) &
            (d.macd_hist_change_3 <= 0) & (d.sma50_slope_10 > 0))
    return (low.fillna(False).to_numpy(bool) & ~action.to_numpy(bool),
            high.fillna(False).to_numpy(bool) & ~action.to_numpy(bool))

def compact(m):
    return {k:m.get(k) for k in ("period_start","period_end","final_equity","total_return","cagr",
        "max_drawdown","trade_count","win_rate","exposure_ratio","total_commission","total_sell_tax",
        "estimated_slippage_cost","blocked_limit_orders","missing_price_limit_orders")}

def run(features_path, limits_path, dividends_path, outdir):
    d, limits, dividends = inputs(features_path, limits_path, dividends_path)
    test_end = d.date.max().date().isoformat()
    grid = list(itertools.product(
        (-.006,-.008,-.010), (1.1,1.2), (.70,.75,.80), (.08,.10,.12)
    ))
    candidates=[]
    for low_ret,low_rv,high_loc,high_dist in grid:
        lo,hi=make_signals(d,low_ret,low_rv,high_loc,high_dist)
        m=_simulate_period(d,lo,hi,limits,dividends,*RESEARCH,10,INITIAL_CAPITAL,True)
        candidates.append({"params":{"low_return":low_ret,"low_relative_volume":low_rv,
            "high_close_location":high_loc,"high_distance_sma200":high_dist},
            "research":compact(m),"research_trades":m["trade_count"],"research_win_rate":m["win_rate"]})
    eligible=[x for x in candidates if (x["research_trades"]>=5 and x["research_win_rate"] is not None
        and x["research_win_rate"]>=WIN_FLOOR)]
    # Select only on the predeclared research period. If no candidate preserves the win floor,
    # mark the search as failed rather than relaxing the floor.
    pool=eligible if eligible else []
    selected=max(pool,key=lambda x:(x["research"]["cagr"] if x["research"]["cagr"] is not None else -1,
        x["research_win_rate"],-abs(x["research"]["max_drawdown"] or -1))) if pool else None
    rows=[]
    for x in candidates:
        p=x["params"]; lo,hi=make_signals(d,p["low_return"],p["low_relative_volume"],
            p["high_close_location"],p["high_distance_sma200"])
        row={"params":p,"research":x["research"],"research_win_rate":x["research_win_rate"],
             "research_eligible":x in eligible}
        for label,(a,b) in (("validation",VALIDATION),("test",(TEST[0],test_end))):
            base=_simulate_period(d,lo,hi,limits,dividends,a,b,10,INITIAL_CAPITAL,True)
            row[label+"_10bps"]=compact(base)
        rows.append(row)
    stress={}
    if selected:
        p=selected["params"]; lo,hi=make_signals(d,p["low_return"],p["low_relative_volume"],
            p["high_close_location"],p["high_distance_sma200"])
        for period,(a,b) in (("research",RESEARCH),("validation",VALIDATION),("test",(TEST[0],test_end))):
            stress[period]={}
            for slip in SLIPPAGE_BPS:
                m=_simulate_period(d,lo,hi,limits,dividends,a,b,slip,INITIAL_CAPITAL,True)
                stress[period][f"{slip}bps"]=compact(m)
    # Export trade-level diagnostics for the candidate selected on research data only.
    # Validation and historical-test ledgers are descriptive; they never affect selection.
    selected_trade_diagnostics = {}
    selected_trade_rows = []
    selected_equity_rows = []
    if selected:
        p = selected["params"]
        selected_low, selected_high = make_signals(
            d, p["low_return"], p["low_relative_volume"],
            p["high_close_location"], p["high_distance_sma200"],
        )
        for period_name, (period_start, period_end) in (
            ("research", RESEARCH),
            ("validation", VALIDATION),
            ("test", (TEST[0], test_end)),
        ):
            metrics = _simulate_period(
                d, selected_low, selected_high, limits, dividends,
                period_start, period_end, 10, INITIAL_CAPITAL, True,
            )
            selected_trade_diagnostics[period_name] = compact(metrics)
            for trade_number, trade in enumerate(metrics["trades"], start=1):
                selected_trade_rows.append({
                    "period": period_name, "slippage_bps_per_side": 10,
                    "trade_number": trade_number, **trade,
                })
            for point in metrics["equity_curve"]:
                selected_equity_rows.append({"period": period_name, **point})
    result={"symbol":"2330","candidate_count":len(candidates),"selection_period":list(RESEARCH),
        "validation_period":list(VALIDATION),"historical_test_period":[TEST[0],test_end],
        "selection_rule":"maximize research-period CAGR subject to >=5 trades and win rate >=86.9565%; no validation/test selection",
        "win_rate_floor":WIN_FLOOR,"selected_candidate":selected,"selected_cost_stress":stress,\n        "selected_trade_diagnostics":selected_trade_diagnostics,
        "all_candidates":rows,"selection_passed":selected is not None,
        "target_status":"NOT_ESTABLISHED; inspect validation/test metrics and cost stress",
        "execution":"close signal t; next-session open; long-only all-in/all-out; integer shares; price-limit locked orders; fees/tax/slippage; cash dividends",
        "costs":{"commission_each_side":BUY_FEE,"sell_tax":SELL_TAX,"slippage_bps_each_side":list(SLIPPAGE_BPS)},
        "warning":"This is a finite predeclared rule-family search, not a guarantee. Historical test data has been examined in prior research and is not pristine OOS. Forward paper validation remains necessary."}
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    (out/"tsmc_2330_40pct_rule_search.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    flat=[]
    for r in rows:
        flat.append({"low_return":r["params"]["low_return"],"low_relative_volume":r["params"]["low_relative_volume"],
          "high_close_location":r["params"]["high_close_location"],"high_distance_sma200":r["params"]["high_distance_sma200"],
          "research_cagr":r["research"].get("cagr"),"research_win_rate":r["research_win_rate"],
          "research_trades":r["research"].get("trade_count"),"research_eligible":r["research_eligible"],
          "validation_cagr":r["validation_10bps"].get("cagr"),"validation_win_rate":r["validation_10bps"].get("win_rate"),
          "test_cagr":r["test_10bps"].get("cagr"),"test_win_rate":r["test_10bps"].get("win_rate")})
    pd.DataFrame(flat).to_csv(out/"tsmc_2330_40pct_rule_search.csv",index=False)\n    pd.DataFrame(selected_trade_rows).to_csv(out/"tsmc_2330_40pct_selected_trade_ledger.csv",index=False)\n    pd.DataFrame(selected_equity_rows).to_csv(out/"tsmc_2330_40pct_selected_equity.csv",index=False)\n    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",required=True); ap.add_argument("--limits",required=True)
    ap.add_argument("--dividends",required=True); ap.add_argument("--output",default="artifacts/tsmc_2330_swing")
    a=ap.parse_args(); print(json.dumps(run(a.features,a.limits,a.dividends,a.output),ensure_ascii=False,indent=2,default=float))

# Re-run after extending the workflow timeout for the full TSMC research pipeline.
