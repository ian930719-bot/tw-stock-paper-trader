"""Paired-year blind validation: 70% train, 30% calibrate, next-year blind test."""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PAIR_YEARS=tuple(range(2010,2027,2))
HORIZONS=(1,3,5,10,20)
DEFAULT_THRESHOLDS=("p0.030_atr2.0","p0.050_atr2.0","p0.080_atr2.0","p0.100_atr2.0")
EXCLUDE={"symbol","date","open","high","low","close","volume","turnover",
         "corporate_action_window","atr14","raw_open","raw_high","raw_low","raw_close"}

def _lookup(dates):
    return {pd.Timestamp(v).normalize():i for i,v in enumerate(pd.to_datetime(dates))}

def event_target(dates, events, horizon, kind):
    """One positive row exactly horizon sessions before each confirmed pivot."""
    y=np.zeros(len(dates),dtype=np.int8); lookup=_lookup(dates)
    if not events.empty:
        for v in pd.to_datetime(events.loc[events.type.astype(str).str.upper()==kind,"pivot_date"],errors="coerce").dropna():
            i=lookup.get(pd.Timestamp(v).normalize())
            if i is not None and i>=horizon: y[i-horizon]=1
    return y

def ambiguous_rows(dates, events, horizon, cutoff):
    """Exclude exact-lead rows whose pivot was not confirmed by the cutoff."""
    out=np.zeros(len(dates),dtype=bool); lookup=_lookup(dates)
    if not events.empty:
        later=events.loc[pd.to_datetime(events.confirmation_date,errors="coerce")>cutoff]
        for v in pd.to_datetime(later.pivot_date,errors="coerce").dropna():
            i=lookup.get(pd.Timestamp(v).normalize())
            if i is not None and i>=horizon: out[i-horizon]=True
    return out

def calibrate(y,p):
    y=np.asarray(y,dtype=np.int8); p=np.asarray(p,dtype=float)
    if int(y.sum())<3: return None,{"status":"skipped_insufficient_calibration_events","events":int(y.sum())}
    pr,rc,ts=precision_recall_curve(y,p); best=None
    for i,t in enumerate(ts):
        n=int((p>=t).sum())
        if n<3: continue
        f=2*pr[i]*rc[i]/(pr[i]+rc[i]) if pr[i]+rc[i] else 0.
        cand=(float(f),float(pr[i]),-n,float(t),n)
        if best is None or cand[:3]>best[:3]: best=cand
    if best is None: return None,{"status":"skipped_insufficient_calibration_signals","events":int(y.sum())}
    return best[3],{"status":"calibrated","threshold":best[3],"f1":best[0],
                    "precision":best[1],"signal_days":best[4],"event_days":int(y.sum())}

def metrics(y,p,signal):
    y=np.asarray(y,dtype=np.int8); p=np.asarray(p,dtype=float); signal=np.asarray(signal,dtype=bool)
    n=int(signal.sum()); e=int(y.sum()); tp=int((signal&(y==1)).sum())
    prec=tp/n if n else None; rec=tp/e if e else None; prev=e/len(y) if len(y) else None
    return {"rows":len(y),"event_days":e,"signal_days":n,"true_positive_days":tp,
            "prevalence":prev,"precision":prec,"recall":rec,
            "f1":2*prec*rec/(prec+rec) if prec is not None and rec is not None and prec+rec else 0.,
            "lift":prec/prev if prec is not None and prev else None,
            "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            "pr_auc":float(average_precision_score(y,p)) if e else 0.}

def _add_missing_pivots(x,events,requested):
    missing=[t for t in requested if t not in set(events.threshold_id.astype(str))]
    if not missing: return events
    from tsmc_swing_research import detect_confirmed_pivots
    action=x.get("corporate_action_window",pd.Series(False,index=x.index)).fillna(True).astype(bool).to_numpy()
    extra=[]
    for tid in missing:
        m=re.fullmatch(r"p(\d+\.\d+)_atr(\d+\.\d+)",tid)
        if not m: continue
        ev=detect_confirmed_pivots(x,float(m.group(1)),float(m.group(2)))
        if ev.empty: continue
        bad=action[ev.pivot_index.astype(int).to_numpy()]|action[ev.confirmation_index.astype(int).to_numpy()]
        ev=ev.loc[~bad].copy()
        if not ev.empty: ev.insert(0,"threshold_id",tid); extra.append(ev)
    return pd.concat([events,*extra],ignore_index=True) if extra else events

def run(features_path,events_path,output_dir,threshold_ids=None,data_audit_path=None):
    x=pd.read_csv(features_path,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    ev=pd.read_csv(events_path)
    ev["pivot_date"]=pd.to_datetime(ev["pivot_date"],errors="coerce")
    ev["confirmation_date"]=pd.to_datetime(ev["confirmation_date"],errors="coerce")
    if not {"date","close"}.issubset(x): raise ValueError("Features need date and close")
    if not {"threshold_id","type","pivot_date","confirmation_date"}.issubset(ev): raise ValueError("Pivot events schema incomplete")
    if x.date.duplicated().any() or not x.date.is_monotonic_increasing: raise ValueError("Dates must be unique and ascending")
    requested=threshold_ids or list(DEFAULT_THRESHOLDS)
    ev=_add_missing_pivots(x,ev,requested)
    ev["pivot_date"]=pd.to_datetime(ev["pivot_date"],errors="coerce")
    ev["confirmation_date"]=pd.to_datetime(ev["confirmation_date"],errors="coerce")
    present=set(ev.threshold_id.astype(str)); requested=[t for t in requested if t in present]
    cols=[c for c in x if c not in EXCLUDE and not c.startswith("raw_") and pd.api.types.is_numeric_dtype(x[c])]
    if not cols: raise ValueError("No numeric causal features")
    X=x[cols].replace([np.inf,-np.inf],np.nan)
    valid=X.notna().any(axis=1).to_numpy(copy=True)
    if "corporate_action_window" in x: valid &= ~x.corporate_action_window.fillna(True).astype(bool).to_numpy()
    dates=x.date; years=dates.dt.year.to_numpy(); idx=np.arange(len(x))
    rows=[]; cals=[]; preds=[]
    for tid in requested:
        all_ev=ev.loc[ev.threshold_id.astype(str)==tid].copy()
        for ay in PAIR_YEARS:
            vy=ay+1; ai=np.flatnonzero(years==ay); vi=np.flatnonzero(years==vy)
            if not len(ai):
                rows.append({"threshold_id":tid,"analysis_year":ay,"validation_year":vy,"status":"skipped_analysis_year_missing"}); continue
            if not len(vi):
                rows.append({"threshold_id":tid,"analysis_year":ay,"validation_year":vy,"status":"pending_validation_year_missing"}); continue
            split=int(math.floor(.70*len(ai)))
            if split<1 or split>=len(ai):
                rows.append({"threshold_id":tid,"analysis_year":ay,"validation_year":vy,"status":"skipped_insufficient_analysis_rows"}); continue
            train_cut=dates.iloc[ai[split-1]]; end=dates.iloc[ai[-1]]
            tr_ev=all_ev.loc[(all_ev.pivot_date.dt.year==ay)&(all_ev.confirmation_date<=train_cut)]
            ca_ev=all_ev.loc[(all_ev.pivot_date.dt.year==ay)&(all_ev.confirmation_date<=end)]
            va_ev=all_ev.loc[all_ev.pivot_date.dt.year.isin([vy,vy+1])]
            for h in HORIZONS:
                base_tr=np.zeros(len(x),bool); base_tr[ai[:split]]=True
                base_tr &= valid & (idx<=ai[split-1]-h)
                base_ca=np.zeros(len(x),bool); base_ca[ai[split:]]=True
                base_ca &= valid & (idx<=ai[-1]-h)
                vm=np.zeros(len(x),bool); vm[vi]=True
                vm &= valid & (idx<=len(x)-h-1)
                for kind in ("HIGH","LOW"):
                    kev=all_ev.loc[all_ev.type.astype(str).str.upper()==kind]
                    tm=base_tr & ~ambiguous_rows(dates,kev,h,train_cut)
                    cm=base_ca & ~ambiguous_rows(dates,kev,h,end)
                    yt=event_target(dates,tr_ev,h,kind); yc=event_target(dates,ca_ev,h,kind); yv=event_target(dates,va_ev,h,kind)
                    nr,nc,nv=int(tm.sum()),int(cm.sum()),int(vm.sum())
                    nt,ncal=int(yt[tm].sum()),int(yc[cm].sum())
                    common={"threshold_id":tid,"analysis_year":ay,"validation_year":vy,
                            "horizon_sessions":h,"pivot_type":kind,"training_rows":nr,
                            "training_event_days":nt,"calibration_rows":nc,"calibration_event_days":ncal,
                            "validation_rows":nv}
                    if nr<50 or nt<3 or len(np.unique(yt[tm]))<2:
                        rows.append({**common,"status":"skipped_insufficient_training_data"}); continue
                    fcols=[c for c in cols if X.loc[tm,c].notna().any()]
                    model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                        LogisticRegression(solver="liblinear",max_iter=1000,class_weight="balanced",C=.2,random_state=0))
                    model.fit(X.loc[tm,fcols],yt[tm])
                    if nc==0:
                        rows.append({**common,"status":"skipped_insufficient_calibration_rows"}); continue
                    cp=model.predict_proba(X.loc[cm,fcols])[:,1]
                    threshold,ci=calibrate(yc[cm],cp)
                    cals.append({**common,"training_end":dates.loc[tm].max().date().isoformat(),
                        "training_cutoff":train_cut.date().isoformat(),"calibration_start":dates.loc[cm].min().date().isoformat() if nc else None,
                        "calibration_end":dates.loc[cm].max().date().isoformat() if nc else None,
                        "calibration_cutoff":end.date().isoformat(),**ci})
                    if threshold is None:
                        rows.append({**common,"status":ci["status"]}); continue
                    if nv==0:
                        rows.append({**common,"status":"skipped_insufficient_validation_rows"}); continue
                    vp=model.predict_proba(X.loc[vm,fcols])[:,1]; truth=yv[vm]; sig=vp>=threshold
                    rows.append({**common,"status":"evaluated","training_start":dates.loc[tm].min().date().isoformat(),
                        "training_end":dates.loc[tm].max().date().isoformat(),"training_cutoff":train_cut.date().isoformat(),
                        "calibration_start":dates.loc[cm].min().date().isoformat(),"calibration_end":dates.loc[cm].max().date().isoformat(),
                        "calibration_cutoff":end.date().isoformat(),"validation_start":dates.loc[vm].min().date().isoformat(),
                        "validation_end":dates.loc[vm].max().date().isoformat(),"partial_validation_year":vy==int(dates.max().year),
                        "validation_year_complete":vy<int(dates.max().year),"validation_pivot_events":int(((all_ev.pivot_date.dt.year==vy)&(all_ev.type.astype(str).str.upper()==kind)).sum()),
                        "validation_event_days":int(truth.sum()),"calibrated_threshold":threshold,"feature_count":len(fcols),
                        **metrics(truth,vp,sig)})
                    for d,p,y,s in zip(dates.loc[vm],vp,truth,sig):
                        preds.append({"threshold_id":tid,"analysis_year":ay,"validation_year":vy,"horizon_sessions":h,
                            "pivot_type":kind,"date":d.date().isoformat(),"probability":float(p),
                            "threshold":float(threshold),"signal":bool(s),"event":bool(y)})
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    annual=pd.DataFrame(rows); cal=pd.DataFrame(cals); pred=pd.DataFrame(preds)
    annual.to_csv(out/"tsmc_2330_paired_annual_blind_validation.csv",index=False)
    cal.to_csv(out/"tsmc_2330_paired_annual_calibration.csv",index=False)
    pred.to_csv(out/"tsmc_2330_paired_annual_predictions.csv",index=False)
    pooled=[]
    if not pred.empty:
        for keys,g in pred.groupby(["threshold_id","horizon_sessions","pivot_type"],sort=True):
            pooled.append({"threshold_id":keys[0],"horizon_sessions":int(keys[1]),"pivot_type":keys[2],
                           **metrics(g.event.astype(int).to_numpy(),g.probability.to_numpy(),g.signal.to_numpy())})
    pd.DataFrame(pooled).to_csv(out/"tsmc_2330_paired_annual_pooled_oof.csv",index=False)
    audit=json.loads(Path(data_audit_path).read_text(encoding="utf-8")) if data_audit_path and Path(data_audit_path).exists() else {}
    summary={"symbol":"2330","protocol":"even year A: chronological first 70% train, final 30% calibrate; freeze model and threshold; odd year A+1 blind test",
        "analysis_years":list(PAIR_YEARS),"horizons_sessions":list(HORIZONS),"threshold_ids":requested,
        "feature_count":len(cols),"data_rows":len(x),"data_start":dates.min().date().isoformat(),"data_end":dates.max().date().isoformat(),
        "data_research_grade_eligible":audit.get("research_grade_eligible"),"data_research_grade_reason":audit.get("research_grade_reason"),
        "evaluated_folds":int((annual.status=="evaluated").sum()),"skipped_folds":int(annual.status.astype(str).str.startswith("skipped").sum()),
        "pending_folds":int((annual.status=="pending_validation_year_missing").sum()),"pooled_out_of_fold_results":pooled,
        "model":"median imputation + scaling + logistic regression liblinear C=0.2 balanced; fitted on first 70% only",
        "calibration":"maximum F1 on final 30%; minimum 3 exact-lead event rows and 3 signals; otherwise skipped",
        "label":"exactly 1/3/5/10/20 sessions before a confirmed pivot; HIGH and LOW trained separately",
        "purging":"cutoff-horizon purge plus exclusion of exact-lead rows whose pivot confirmation was unavailable at cutoff",
        "validation_is_never_used_for_fit_or_threshold":True,"trading_backtest":False,
        "warning":"Event classification only, not executable returns or CAGR. Pivot outcomes are retrospective labels. Data provenance gates remain separate."}
    (out/"tsmc_2330_paired_annual_blind_validation.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",required=True); p.add_argument("--events",required=True)
    p.add_argument("--output",default="artifacts/tsmc_2330_swing")
    p.add_argument("--threshold-ids",nargs="*",default=None); p.add_argument("--data-audit",default=None)
    a=p.parse_args()
    print(json.dumps(run(a.features,a.events,a.output,a.threshold_ids,a.data_audit),ensure_ascii=False,indent=2))

if __name__=="__main__": main()
