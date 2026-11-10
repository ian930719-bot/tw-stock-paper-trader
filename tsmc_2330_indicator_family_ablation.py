"""Incremental indicator-family ablation for TSMC 2330 swing turning points.

Uses the training-only selected pivot threshold; fits on 2010-2016,
compares families on 2017-2021 validation, refits through validation, and
reports 2022+ as descriptive only. No execution or CAGR is calculated.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tsmc_swing_research import detect_confirmed_pivots
from tsmc_2330_selected_threshold_signal_diagnostics import period_events, build_target

TRAIN_START, TRAIN_END = "2010-01-01", "2016-12-31"
VAL_START, VAL_END = "2017-01-01", "2021-12-31"
TEST_START = "2022-01-01"
HORIZONS = (1, 3, 5)
CHIP = ("foreign_", "foreign_dealer_", "trust_", "dealer_", "institutional_", "broker_")

def family_for(col: str) -> str:
    if col.startswith(CHIP): return "institutional"
    if col.startswith(("ema_", "distance_ema_", "ema12_", "ema20_", "ema50_", "ema200_", "ppo", "roc_")): return "trend_ema_momentum"
    if col.startswith(("stoch_", "williams_", "plus_di", "minus_di", "dx", "adx")) or col == "cci20": return "oscillator_trend_strength"
    if col.startswith(("obv_", "mfi", "chaikin_", "volume_sma", "volume_zscore", "rolling_vwap", "distance_vwap")): return "volume_money_flow"
    if col.startswith(("donchian_", "distance_donchian_")): return "price_channel"
    return "core_technical"

def make_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=0.2, random_state=0))

def metrics(model, X, mask, y):
    if not mask.any() or len(np.unique(y[mask])) < 2:
        return {"error":"segment lacks rows or both classes","rows":int(mask.sum()),"event_days":int(y[mask].sum())}
    p=model.predict_proba(X.loc[mask])[:,1]; truth=y[mask]; prevalence=float(truth.mean())
    n=max(1,int(np.ceil(len(p)*.10))); top=np.argsort(p)[-n:]
    return {"rows":int(mask.sum()),"event_days":int(truth.sum()),"prevalence":prevalence,
      "roc_auc":float(roc_auc_score(truth,p)),"pr_auc":float(average_precision_score(truth,p)),
      "top_decile_precision":float(truth[top].mean()),
      "top_decile_lift":float(truth[top].mean()/prevalence) if prevalence else None,
      "top_decile_recall":float(truth[top].sum()/truth.sum()) if truth.sum() else None}

def run(features_path, oracle_path, output_dir):
    x=pd.read_csv(features_path,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    oracle=json.loads(Path(oracle_path).read_text(encoding="utf-8"))
    chosen=oracle.get("threshold_selected_using_training_period_only",{})
    rev,atr=chosen.get("min_reversal_pct"),chosen.get("atr_multiple")
    if rev is None or atr is None: raise ValueError("Missing training-only selected pivot threshold")
    events=detect_confirmed_pivots(x,float(rev),float(atr))
    if events.empty: raise ValueError("No pivots for selected threshold")
    dates=x.date; idx=np.arange(len(x))
    val_start_idx=int(np.flatnonzero((dates>=pd.Timestamp(VAL_START)).to_numpy())[0])
    test_start_idx=int(np.flatnonzero((dates>=pd.Timestamp(TEST_START)).to_numpy())[0])
    groups={}
    for col in x.columns:
        if col in {"symbol","date","open","high","low","close","volume","turnover","corporate_action_window","atr14","obv"} or col.startswith("raw_"): continue
        if pd.api.types.is_numeric_dtype(x[col]): groups.setdefault(family_for(col),[]).append(col)
    core=groups.get("core_technical",[])
    variants={"core_technical":core}
    for family,cols in groups.items():
        if family!="core_technical": variants[f"core_plus_{family}"]=core+cols
    variants["all_technical"]=sum([v for k,v in groups.items() if k!="institutional"],[])
    variants["all_technical_plus_institutional"]=sum(groups.values(),[])
    results=[]
    for horizon in HORIZONS:
      train_events=period_events(events,dates,TRAIN_START,TRAIN_END,horizon)
      val_events=period_events(events,dates,VAL_START,VAL_END,horizon)
      test_events=period_events(events,dates,TEST_START,dates.max().date().isoformat(),horizon)
      for kind in ("LOW","HIGH"):
        ytr=build_target(dates,train_events,horizon,kind); yva=build_target(dates,val_events,horizon,kind); yte=build_target(dates,test_events,horizon,kind)
        tr=(dates>=TRAIN_START).to_numpy()&(dates<=TRAIN_END).to_numpy()&(idx>=horizon)&(idx<=val_start_idx-horizon-1)
        va=(dates>=VAL_START).to_numpy()&(dates<=VAL_END).to_numpy()&(idx>=val_start_idx+horizon)&(idx<=test_start_idx-horizon-1)
        te=(dates>=TEST_START).to_numpy()&(idx>=test_start_idx+horizon)
        if "corporate_action_window" in x:
            safe=~x.corporate_action_window.fillna(True).astype(bool).to_numpy(); tr&=safe; va&=safe; te&=safe
        for name,cols in variants.items():
            usable=[c for c in cols if x.loc[tr,c].notna().any()]
            if not usable or len(np.unique(ytr[tr]))<2 or len(np.unique(yva[va]))<2:
                results.append({"family_set":name,"pivot_type":kind,"horizon":horizon,"status":"skipped"}); continue
            X=x[usable].replace([np.inf,-np.inf],np.nan)
            model=make_model().fit(X.loc[tr],ytr[tr])
            val=metrics(model,X,va,yva)
            trva=(tr|va)
            final=make_model().fit(X.loc[trva],np.maximum(ytr,yva)[trva])
            test=metrics(final,X,te,yte)
            results.append({"family_set":name,"pivot_type":kind,"horizon":horizon,"feature_count":len(usable),
                "validation":val,"test_2022_plus_descriptive":test})
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(out/"tsmc_2330_indicator_family_ablation.csv",index=False)
    summary={"symbol":"2330","threshold":chosen,"train":[TRAIN_START,TRAIN_END],"validation":[VAL_START,VAL_END],
      "descriptive_test":[TEST_START,dates.max().date().isoformat()],"families":{k:len(v) for k,v in groups.items()},
      "results":results,"test_period_used_for_selection":False,"test_period_is_pristine":False,
      "strategy_backtest":False,"note":"Incremental feature-family classification ablation only; not trading returns."}
    (out/"tsmc_2330_indicator_family_ablation.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    return summary

def main():
    p=argparse.ArgumentParser(); p.add_argument("--features",required=True); p.add_argument("--oracle",required=True); p.add_argument("--output",default="artifacts/tsmc_2330_swing")
    a=p.parse_args(); print(json.dumps(run(a.features,a.oracle,a.output),ensure_ascii=False,indent=2,default=float))
if __name__=="__main__": main()
