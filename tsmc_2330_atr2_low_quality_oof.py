"""Expanding-year OOF classifier for ATR-2 causal TSMC low-pivot quality.

This predicts an event-quality label from confirmation-close features only.
It is NOT a trading strategy and does not report portfolio CAGR or win rate.
Training labels are embargoed by one full calendar year to ensure the 60-session
forward outcome was observable before each validation year began.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

FEATURES = [
    "confirm_return_1", "confirm_return_5", "confirm_return_20",
    "confirm_atr_pct14", "confirm_rsi14", "confirm_macd_hist_change_3",
    "confirm_distance_sma_20", "confirm_distance_ema_20",
    "confirm_distance_sma_50", "confirm_distance_sma_200",
    "confirm_relative_volume20", "confirm_close_location",
    "confirm_intraday_range_pct", "confirm_obv_flow_ratio_5",
    "confirm_plus_di14", "confirm_minus_di14", "confirm_mfi14",
    "confirm_stoch_k14", "confirmation_lag_sessions",
    "reversal_at_confirmation_pct",
]
TARGET = "quality_label"

def metrics(y, p):
    if len(y) == 0:
        return {"n":0}
    pred = p >= .65
    out = {
        "n":int(len(y)), "positive_rate":float(np.mean(y)),
        "threshold":.65, "coverage":float(np.mean(pred)),
        "accuracy":float(accuracy_score(y,pred)),
        "precision":float(precision_score(y,pred,zero_division=0)),
        "recall":float(recall_score(y,pred,zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp":confusion_matrix(y,pred,labels=[0,1]).ravel().tolist(),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"]=float(roc_auc_score(y,p))
    else:
        out["roc_auc"]=None
    return out

def run(events_path, output):
    d=pd.read_csv(events_path,parse_dates=["confirmation_date"])
    d=d[np.isclose(d.atr_multiplier,2.0)].copy()
    d=d[~d.corporate_action_window_at_confirmation.astype(bool)].copy()
    d=d.dropna(subset=["close_return_60d","mfe_60d"])
    # A high-quality confirmed low must deliver both positive 60-session close return
    # and at least 8% MFE. This is a research label, not a promised trade outcome.
    d[TARGET]=((d.close_return_60d>0)&(d.mfe_60d>=.08)).astype(int)
    d["validation_year"]=d.confirmation_date.dt.year
    d["confirmation_year"]=d.validation_year
    d=d.sort_values("confirmation_date").reset_index(drop=True)
    folds=[]; predictions=[]
    years=sorted(d.validation_year.unique())
    for year in years:
        # One-calendar-year embargo: train events must confirm no later than Dec 31
        # two years before validation year, so their 60-session labels predate the fold.
        train_cutoff=pd.Timestamp(year=int(year)-2,month=12,day=31)
        train=d[d.confirmation_date<=train_cutoff]
        test=d[d.validation_year==year]
        if len(train)<30 or train[TARGET].nunique()<2 or len(test)==0:
            folds.append({"validation_year":int(year),"train_n":int(len(train)),
                          "test_n":int(len(test)),"status":"insufficient_prior_labels"})
            continue
        model=Pipeline([
            ("impute",SimpleImputer(strategy="median")),
            ("scale",StandardScaler()),
            ("logit",LogisticRegression(C=.3,class_weight="balanced",max_iter=2000,random_state=2330))
        ])
        model.fit(train[FEATURES],train[TARGET])
        prob=model.predict_proba(test[FEATURES])[:,1]
        m=metrics(test[TARGET].to_numpy(),prob)
        folds.append({"validation_year":int(year),"training_cutoff":train_cutoff.date().isoformat(),
                      "train_n":int(len(train)),"test_n":int(len(test)),"status":"scored","metrics":m})
        for (_,row),pr in zip(test.iterrows(),prob):
            predictions.append({
                "confirmation_date":row.confirmation_date.date().isoformat(),
                "validation_year":int(year),"actual_quality_label":int(row[TARGET]),
                "predicted_probability":float(pr),"predicted_high_quality":bool(pr>=.65),
                "close_return_60d":float(row.close_return_60d),"mfe_60d":float(row.mfe_60d),
                "confirmation_lag_sessions":int(row.confirmation_lag_sessions)
            })
    pred=pd.DataFrame(predictions)
    overall=metrics(pred.actual_quality_label.to_numpy(),pred.predicted_probability.to_numpy()) if len(pred) else {"n":0}
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    pred.to_csv(out/"tsmc_2330_atr2_low_quality_oof_predictions.csv",index=False)
    result={
      "symbol":"2330","classification":"EXPANDING_YEAR_OOF_EVENT_CLASSIFIER_NOT_TRADING_STRATEGY",
      "source_atr_multiplier":2.0,"source_event_count":int(len(d)),
      "target_definition":"quality=1 iff post-confirmation 60-session close return >0 AND MFE >=8%; label only",
      "features":FEATURES,"feature_timing":"confirmation close only; no pivot-date retrospective feature columns",
      "training_protocol":"expanding chronological training; 1 calendar year embargo; minimum 30 prior labeled events; logistic regression C=0.3 class_weight=balanced",
      "probability_threshold":.65,"overall_oof_metrics":overall,"annual_folds":folds,
      "prediction_count":int(len(pred)),
      "limitations":["Events are temporally clustered; fold metrics are descriptive.",
        "ATR thresholds are overlapping in source family; this model uses ATR 2.0 only.",
        "Historical 2010-2026 data has been examined previously.",
        "This predicts event labels, not actual executable trades, CAGR, completed-trade win rate, or MDD."]
    }
    (out/"tsmc_2330_atr2_low_quality_oof.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--events",required=True)
    p.add_argument("--output",default="artifacts/tsmc_2330_atr2_low_quality_oof")
    a=p.parse_args(); print(json.dumps(run(a.events,a.output),ensure_ascii=False,indent=2))
