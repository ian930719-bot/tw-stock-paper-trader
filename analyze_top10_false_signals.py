"""Pooled, causal false-signal diagnostics for frozen high-hold trade outcomes.

Thresholds are selected on 2015-2021 trades only and evaluated on 2022-2026.
This is a diagnostic screen, not a deployable strategy or a substitute for a
full event-driven portfolio backtest.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

FEATURES = [
    "return_1", "relative_volume20", "stoch_k14", "close_location",
    "obv_flow_ratio_5", "distance_sma_200", "distance_sma_50",
    "distance_sma_20", "distance_ema_20", "rsi14", "macd_hist_change_3",
    "sma50_slope_10", "intraday_range_pct", "atr_pct", "adx14",
    "plus_di", "minus_di", "bb_width", "bb_percent_b", "mfi14",
]
QUANTILES = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
SPLIT_DATE = "2022-01-01"

def parse_features(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    if d.empty:
        raise RuntimeError("No executed trade diagnostics were supplied")
    if "entry_features_json" not in d or "entry_signal_date" not in d:
        raise RuntimeError("Trade file is missing causal entry signal snapshots")
    rows=[]
    for raw in d.entry_features_json.fillna("{}"):
        try: rows.append(json.loads(raw))
        except Exception: rows.append({})
    f=pd.DataFrame(rows)
    keep=[x for x in FEATURES if x in f.columns]
    if not keep: raise RuntimeError("No predeclared technical features found in trade snapshots")
    for col in keep: f[col]=pd.to_numeric(f[col],errors="coerce")
    out=pd.concat([d.reset_index(drop=True),f[keep].reset_index(drop=True)],axis=1)
    out["entry_signal_date"]=pd.to_datetime(out.entry_signal_date,errors="coerce")
    out["net_return"]=pd.to_numeric(out.net_return,errors="coerce")
    out=out.dropna(subset=["entry_signal_date","net_return"]).copy()
    out["win"]=(out.net_return>0).astype(int)
    out["period"]=np.where(out.entry_signal_date<pd.Timestamp(SPLIT_DATE),"train_2015_2021","test_2022_2026")
    return out

def score_mask(d, mask):
    x=d.loc[mask]
    if x.empty:return {"n":0,"win_rate":None,"mean_net_return":None,"median_net_return":None,"compounded_trade_return":None}
    r=x.net_return.astype(float)
    compounded=float(np.prod(1+r)-1) if np.all(1+r>0) else None
    return {"n":int(len(x)),"win_rate":float(x.win.mean()),"mean_net_return":float(r.mean()),
            "median_net_return":float(r.median()),"compounded_trade_return":compounded}

def choose_filter(train, feature):
    x=pd.to_numeric(train[feature],errors="coerce")
    valid=x.notna()
    base=train.loc[valid]
    if len(base)<20:return None
    best=None
    for q in QUANTILES:
        threshold=float(x[valid].quantile(q))
        for direction in ("ge","le"):
            mask=(x>=threshold) if direction=="ge" else (x<=threshold)
            mask=mask & valid
            n=int(mask.sum())
            if n<max(10,int(math.ceil(len(base)*0.40))):continue
            m=score_mask(train,mask)
            candidate={"feature":feature,"direction":direction,"threshold":threshold,
                       "train_n":n,"train_retention":n/len(base),
                       "train_win_rate":m["win_rate"],"train_mean_net_return":m["mean_net_return"]}
            key=(candidate["train_win_rate"],candidate["train_mean_net_return"],candidate["train_n"])
            if best is None or key>best[0]:best=(key,candidate)
    return best[1] if best else None

def evaluate_filter(data, rule):
    if rule["direction"]=="ge":mask=data[rule["feature"]]>=rule["threshold"]
    else:mask=data[rule["feature"]]<=rule["threshold"]
    mask=mask & data[rule["feature"]].notna()
    return score_mask(data,mask),int(mask.sum())

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--trades",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    d=parse_features(a.trades)
    train=d[d.period=="train_2015_2021"].copy()
    test=d[d.period=="test_2022_2026"].copy()
    base={"total_trades":len(d),"symbols":sorted(d.symbol.astype(str).unique().tolist()),
          "train_trades":len(train),"test_trades":len(test),
          "train_win_rate":float(train.win.mean()) if len(train) else None,
          "test_win_rate":float(test.win.mean()) if len(test) else None,
          "split_date":SPLIT_DATE,"features_available":[x for x in FEATURES if x in d.columns]}
    (out/"summary.json").write_text(json.dumps(base,ensure_ascii=False,indent=2),encoding="utf-8")
    sep=[]
    for feature in base["features_available"]:
        w=pd.to_numeric(d.loc[d.win==1,feature],errors="coerce").dropna()
        l=pd.to_numeric(d.loc[d.win==0,feature],errors="coerce").dropna()
        if len(w)<3 or len(l)<3:continue
        try:pval=float(mannwhitneyu(w,l,alternative="two-sided").pvalue)
        except Exception:pval=None
        pooled=np.sqrt((w.var(ddof=1)+l.var(ddof=1))/2)
        effect=float((w.mean()-l.mean())/pooled) if pooled and np.isfinite(pooled) else None
        sep.append({"feature":feature,"wins_n":len(w),"losses_n":len(l),
                    "win_mean":float(w.mean()),"loss_mean":float(l.mean()),
                    "win_median":float(w.median()),"loss_median":float(l.median()),
                    "standardized_mean_difference":effect,"mann_whitney_p":pval})
    pd.DataFrame(sep).sort_values("mann_whitney_p",na_position="last").to_csv(out/"feature_separation.csv",index=False)
    baseline_test=score_mask(test,pd.Series(True,index=test.index))
    rules=[]
    for feature in base["features_available"]:
        rule=choose_filter(train,feature)
        if not rule:continue
        test_metrics,_=evaluate_filter(test,rule)
        rules.append({**rule,"test_n":test_metrics["n"],
                      "test_retention":test_metrics["n"]/len(test) if len(test) else None,
                      "test_win_rate":test_metrics["win_rate"],
                      "test_mean_net_return":test_metrics["mean_net_return"],
                      "test_median_net_return":test_metrics["median_net_return"],
                      "test_compounded_trade_return":test_metrics["compounded_trade_return"],
                      "test_win_rate_delta_pp":(test_metrics["win_rate"]-baseline_test["win_rate"])*100
                        if test_metrics["win_rate"] is not None and baseline_test["win_rate"] is not None else None})
    pd.DataFrame(rules).sort_values(["test_win_rate_delta_pp","test_n"],ascending=[False,False]).to_csv(out/"single_feature_filter_validation.csv",index=False)
    loso=[]
    for heldout in sorted(d.symbol.astype(str).unique()):
        tr=d[(d.symbol.astype(str)!=heldout)&(d.period=="train_2015_2021")]
        te=d[(d.symbol.astype(str)==heldout)&(d.period=="test_2022_2026")]
        if len(tr)<20 or len(te)<3:continue
        for feature in base["features_available"]:
            rule=choose_filter(tr,feature)
            if not rule:continue
            metrics,_=evaluate_filter(te,rule)
            loso.append({"heldout_symbol":heldout,**rule,"test_n":metrics["n"],
                         "test_win_rate":metrics["win_rate"],
                         "test_mean_net_return":metrics["mean_net_return"]})
    pd.DataFrame(loso).to_csv(out/"leave_one_symbol_out.csv",index=False)
    notes=[
      "Signals are sampled at the close that created the order; fills occur on the next available session open.",
      "Thresholds are selected only on 2015-2021 executed trades and evaluated on 2022-2026; no test-period threshold tuning.",
      "Each row is an independent single-symbol trade. Pooled compounded_trade_return is descriptive only and is not a portfolio CAGR.",
      "Single-feature screens involve multiple comparisons and are exploratory; no filter is approved for deployment from this report.",
      "The fixed TSMC strategy is not retrained per stock. This analysis asks whether a simple causal entry filter improves out-of-time win rate without pretending to establish a new portfolio result.",
    ]
    (out/"methodology.txt").write_text("\n".join(notes)+"\n",encoding="utf-8")
    print(json.dumps({**base,"test_baseline":baseline_test,
      "candidate_filters":len(rules),"loso_rows":len(loso)},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
