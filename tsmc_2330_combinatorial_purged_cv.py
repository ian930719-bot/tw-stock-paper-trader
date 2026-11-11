"""Combinatorial purged CV for TSMC 2330 pivot-signal robustness.
Not a chronological deployment test: some training folds contain future dates.
"""
from __future__ import annotations
import argparse,itertools,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GROUPS=6
TEST_GROUPS=2
HORIZONS=(1,3,5,10,20)
THRESHOLDS=("p0.030_atr2.0","p0.050_atr2.0","p0.080_atr2.0","p0.100_atr2.0",
"p0.010_atr1.0","p0.020_atr1.0","p0.030_atr1.0","p0.040_atr1.0","p0.020_atr1.5")
FEATURES=("return_1","return_3","return_5","return_10","return_20","return_40",
"distance_sma_20","distance_sma_50","distance_sma_200","sma20_slope_5","trend_stack",
"atr_pct14","rsi14","macd_pct","bb_width20","bb_position20","relative_volume20",
"turnover_ratio20","realized_vol20","overnight_gap","intraday_range_pct","close_location",
"upper_wick_pct","lower_wick_pct","distance_prior_high20","distance_prior_low20",
"bearish_rsi_divergence20","bullish_rsi_divergence20")

def make_groups(n):
    if n<GROUPS*30: raise ValueError(f"Need at least {GROUPS*30} rows; got {n}")
    return [a for a in np.array_split(np.arange(n),GROUPS)]

def labels_for(dates,events,horizon,kind):
    lookup={pd.Timestamp(v).normalize():i for i,v in enumerate(dates)}
    y=np.zeros(len(dates),dtype=np.int8)
    for v in pd.to_datetime(events.loc[events.type.astype(str).str.upper()==kind,"pivot_date"],errors="coerce").dropna():
        i=lookup.get(pd.Timestamp(v).normalize())
        if i is not None and i>=horizon: y[i-horizon]=1
    return y

def purged_train_mask(n,groups,test_ids,purge):
    mask=np.ones(n,dtype=bool)
    for gid in test_ids:
        b=groups[gid]; lo,hi=int(b[0]),int(b[-1])
        mask[max(0,lo-purge):min(n,hi+purge+1)]=False
    return mask

def summarize_coefficients(coef):
    if coef.empty: return pd.DataFrame()
    return (coef.groupby(["feature_set","threshold_id","horizon_sessions","pivot_type","feature"],as_index=False)
        .agg(folds=("fold","nunique"),
             median_coefficient=("coefficient","median"),
             mean_coefficient=("coefficient","mean"),
             positive_fold_share=("coefficient",lambda s:float((s>0).mean())),
             median_abs_coefficient=("coefficient",lambda s:float(s.abs().median()))))

def metrics(y,p,cut=.5):
    y=np.asarray(y,dtype=np.int8);p=np.asarray(p,dtype=float);s=p>=cut
    n=int(s.sum());e=int(y.sum());tp=int((s&(y==1)).sum())
    pr=tp/n if n else None;rc=tp/e if e else None
    return {"rows":len(y),"event_days":e,"signal_days":n,"true_positive_days":tp,
      "precision":pr,"recall":rc,"f1":2*pr*rc/(pr+rc) if pr is not None and rc is not None and pr+rc else 0.,
      "prevalence":float(y.mean()) if len(y) else None,
      "lift":pr/float(y.mean()) if pr is not None and y.mean() else None,
      "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
      "pr_auc":float(average_precision_score(y,p)) if e else 0.}

CHIP_PREFIXES=("foreign_","foreign_dealer_","trust_","dealer_","institutional_")

def build_feature_sets(frame):
    chip_cols=[c for c in frame.columns
        if c.startswith(CHIP_PREFIXES) and c != "institutional_data_available"
        and pd.api.types.is_numeric_dtype(frame[c])]
    if "institutional_data_available" in frame:
        chip_available=frame["institutional_data_available"].fillna(0).astype(bool).to_numpy()
    elif chip_cols:
        chip_available=frame[chip_cols].notna().any(axis=1).to_numpy()
    else:
        chip_available=np.zeros(len(frame),dtype=bool)
    min_chip_rows=min(300,max(30,int(len(frame)*0.10)))
    feature_sets={"technical_only":list(FEATURES)}
    if chip_cols and int(chip_available.sum())>=min_chip_rows:
        feature_sets["technical_plus_institutional"]=list(FEATURES)+chip_cols
    return feature_sets,chip_available,len(chip_cols),min_chip_rows


def run(features_path,events_path,output_dir,threshold_ids=None):
    x=pd.read_csv(features_path,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    ev=pd.read_csv(events_path)
    if not {"threshold_id","type","pivot_date"}.issubset(ev): raise ValueError("Event schema incomplete")
    missing=set(FEATURES)-set(x.columns)
    if missing: raise ValueError(f"Missing core features: {sorted(missing)}")
    if x.date.duplicated().any() or not x.date.is_monotonic_increasing: raise ValueError("Dates must be unique ascending")
    requested=threshold_ids or list(THRESHOLDS)
    absent=[t for t in requested if t not in set(ev.threshold_id.astype(str))]
    if absent:
        from tsmc_swing_research import detect_confirmed_pivots
        action=x.get("corporate_action_window",pd.Series(False,index=x.index)).fillna(True).astype(bool).to_numpy()
        extras=[]
        for tid in absent:
            m=re.fullmatch(r"p(\d+\.\d+)_atr(\d+\.\d+)",tid)
            if not m: continue
            e=detect_confirmed_pivots(x,float(m.group(1)),float(m.group(2)))
            if not e.empty:
                bad=action[e.pivot_index.astype(int).to_numpy()]|action[e.confirmation_index.astype(int).to_numpy()]
                e=e.loc[~bad].copy()
                if not e.empty: e.insert(0,"threshold_id",tid);extras.append(e)
        if extras: ev=pd.concat([ev,*extras],ignore_index=True)
    groups=make_groups(len(x)); combos=list(itertools.combinations(range(GROUPS),TEST_GROUPS))
    feature_sets,chip_available,chip_feature_count,min_chip_rows=build_feature_sets(x)
    base_X=x[list(FEATURES)].replace([np.inf,-np.inf],np.nan)
    base_valid=base_X.notna().any(axis=1).to_numpy(copy=True)
    chip_rows=int(chip_available.sum())
    compare_chip="technical_plus_institutional" in feature_sets
    # Both ablation arms use identical dates whenever institutional history is adequate.
    valid=base_valid & chip_available if compare_chip else base_valid
    fold_rows=[]; pred_rows=[]; coef_rows=[]
    for feature_set,feature_columns in feature_sets.items():
      X=x[feature_columns].replace([np.inf,-np.inf],np.nan)
      for tid in requested:
       tev=ev.loc[ev.threshold_id.astype(str)==tid]
       if tev.empty:
         fold_rows.append({"feature_set":feature_set,"threshold_id":tid,"status":"skipped_no_events"});continue
       for h in HORIZONS:
        for kind in ("HIGH","LOW"):
         y=labels_for(x.date,tev,h,kind); probs=[[] for _ in range(len(x))]
         for fid,test_ids in enumerate(combos,1):
           test=np.zeros(len(x),dtype=bool)
           for gid in test_ids:test[groups[gid]]=True
           train=purged_train_mask(len(x),groups,test_ids,h)&valid
           test &= valid
           tr=int(train.sum());te=int(test.sum());pos=int(y[train].sum())
           base={"feature_set":feature_set,"threshold_id":tid,"horizon_sessions":h,"pivot_type":kind,"fold":fid,
             "test_groups":";".join(map(str,test_ids)),"training_rows":tr,"training_event_days":pos,
             "test_rows":te,"test_event_days":int(y[test].sum()),"purge_sessions":h}
           if tr<100 or pos<3 or len(np.unique(y[train]))<2 or te==0:
             fold_rows.append({**base,"status":"skipped_insufficient_fold_samples"});continue
           cols=[col for col in feature_columns if X.loc[train,col].notna().any()]
           model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
             LogisticRegression(solver="liblinear",max_iter=1000,class_weight="balanced",C=.2,random_state=0))
           model.fit(X.loc[train,cols],y[train]);p=model.predict_proba(X.loc[test,cols])[:,1]
           weights=model.named_steps["logisticregression"].coef_[0]
           for feature,weight in zip(cols,weights):
             coef_rows.append({"feature_set":feature_set,"threshold_id":tid,"horizon_sessions":h,"pivot_type":kind,
               "fold":fid,"feature":feature,"coefficient":float(weight)})
           fold_rows.append({**base,"status":"evaluated","feature_count":len(cols),**metrics(y[test],p)})
           for i,pv in zip(np.flatnonzero(test),p):probs[i].append(float(pv))
         inds=[i for i,v in enumerate(probs) if v]
         for i in inds:
           pred_rows.append({"feature_set":feature_set,"threshold_id":tid,"horizon_sessions":h,"pivot_type":kind,
             "date":x.date.iloc[i].date().isoformat(),"probability":float(np.mean(probs[i])),
             "signal":bool(np.mean(probs[i])>=.5),"event":bool(y[i]),"fold_predictions":len(probs[i])})
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    folds=pd.DataFrame(fold_rows);pred=pd.DataFrame(pred_rows);coef=pd.DataFrame(coef_rows)
    coef_summary=summarize_coefficients(coef)
    folds.to_csv(out/"tsmc_2330_combinatorial_purged_cv_folds.csv",index=False)
    coef.to_csv(out/"tsmc_2330_combinatorial_purged_cv_coefficients.csv",index=False)
    coef_summary.to_csv(out/"tsmc_2330_combinatorial_purged_cv_coefficient_stability.csv",index=False)
    pred.to_csv(out/"tsmc_2330_combinatorial_purged_cv_predictions.csv",index=False)
    pooled=[]
    if not pred.empty:
      for k,g in pred.groupby(["feature_set","threshold_id","horizon_sessions","pivot_type"],sort=True):
        pooled.append({"feature_set":k[0],"threshold_id":k[1],"horizon_sessions":int(k[2]),"pivot_type":k[3],
          "folds":len(combos),**metrics(g.event.astype(int),g.probability)})
    pd.DataFrame(pooled).to_csv(out/"tsmc_2330_combinatorial_purged_cv_summary.csv",index=False)
    summary={"symbol":"2330","protocol":"6 contiguous chronological groups; 15 combinations of 2 held-out groups; other 4 groups train; purge training observations within target horizon of held-out blocks",
      "groups":GROUPS,"test_groups_per_fold":TEST_GROUPS,"combinations":len(combos),
      "threshold_ids":requested,"horizons_sessions":list(HORIZONS),"pivot_types":["HIGH","LOW"],
      "feature_count":len(FEATURES),"feature_sets":{k:len(v) for k,v in feature_sets.items()},
      "institutional_data_available_rows":chip_rows,"institutional_feature_count":chip_feature_count,
      "institutional_minimum_rows_required":min_chip_rows,
      "institutional_ablation_status":"evaluated_same_dates" if compare_chip else "skipped_insufficient_or_missing_institutional_history",
      "probability_cutoff":.5,
      "evaluated_folds":int((folds.status=="evaluated").sum()) if not folds.empty else 0,
      "skipped_folds":int(folds.status.astype(str).str.startswith("skipped").sum()) if not folds.empty else 0,
      "pooled_results":pooled,"uses_future_training_blocks":True,
      "interpretation":"Robustness across historical regimes only. Some training folds contain dates later than held-out folds, so this is not a deployability test. Institutional ablation arms use identical dates when chip coverage is sufficient. Freeze research choices, then run a final untouched chronological holdout before paper trading.",
      "trading_backtest":False}
    (out/"tsmc_2330_combinatorial_purged_cv.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary

def main():
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--events",required=True)
 p.add_argument("--output",default="artifacts/tsmc_2330_swing");p.add_argument("--threshold-ids",nargs="*",default=None)
 a=p.parse_args();print(json.dumps(run(a.features,a.events,a.output,a.threshold_ids),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
