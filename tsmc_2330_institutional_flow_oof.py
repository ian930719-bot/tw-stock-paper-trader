"""Annual OOF test: do institutional flows improve causal ATR-2 low-event quality?

The target is the predeclared complete 60-session outcome label. Institutional
flows are sampled at confirmation close and the hypothetical execution is next
session open. Broker branch features are excluded because coverage is unavailable.
This is an event-ranking diagnostic, not a strategy P&L backtest.
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

TECH = [
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
CHIP = [
    "foreign_net_to_volume", "foreign_net_5d_lots", "foreign_net_20d_lots",
    "foreign_net_positive_streak", "foreign_net_negative_streak",
    "foreign_dealer_net_to_volume",
    "trust_net_to_volume", "trust_net_5d_lots", "trust_net_20d_lots",
    "trust_net_positive_streak", "trust_net_negative_streak",
    "dealer_net_to_volume", "dealer_net_5d_lots", "dealer_net_20d_lots",
    "dealer_net_positive_streak", "dealer_net_negative_streak",
    "institutional_net_to_volume", "institutional_net_5d_lots",
    "institutional_net_20d_lots", "institutional_net_positive_streak",
    "institutional_net_negative_streak",
]

def score(y, p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    if len(y)==0 or len(np.unique(y))<2:
        return {"n":int(len(y)),"error":"requires both target classes"}
    n=max(1,int(np.ceil(len(y)*.10)))
    top=np.argsort(p)[-n:]
    prevalence=float(y.mean())
    return {
        "n":int(len(y)), "positive_rate":prevalence,
        "roc_auc":float(roc_auc_score(y,p)),
        "pr_auc":float(average_precision_score(y,p)),
        "top_decile_n":int(n),
        "top_decile_precision":float(y[top].mean()),
        "top_decile_lift":float(y[top].mean()/prevalence) if prevalence else None,
        "top_decile_recall":float(y[top].sum()/y.sum()) if y.sum() else None,
    }

def model():
    return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
        LogisticRegression(C=.3,class_weight="balanced",max_iter=3000,random_state=2330))

def run(events_path,chip_path,output):
    e=pd.read_csv(events_path,parse_dates=["confirmation_date"])
    e=e[np.isclose(pd.to_numeric(e.atr_multiplier,errors="coerce"),2.0)].copy()
    e["corporate_action_window_at_confirmation"]=e.corporate_action_window_at_confirmation.astype(str).str.lower().isin(["true","1","yes"])
    e=e[~e.corporate_action_window_at_confirmation].copy()
    e=e[pd.to_numeric(e.observed_sessions_60d,errors="coerce").eq(60)].dropna(subset=["mfe_60d","close_return_60d"])
    e["quality_label"]=((e.close_return_60d>0)&(e.mfe_60d>=.08)).astype(int)
    chip=pd.read_csv(chip_path,parse_dates=["date"])
    chip=chip.sort_values("date").drop_duplicates("date",keep="last")
    # Join only the same-day institutional figures: known after that day's close.
    chip_cols=["date"]+CHIP+["institutional_data_available"]
    absent=[c for c in chip_cols if c not in chip.columns]
    if absent: raise ValueError(f"Chip file missing columns: {absent}")
    e=e.merge(chip[chip_cols],left_on="confirmation_date",right_on="date",how="left",validate="many_to_one")
    e=e[e.institutional_data_available.fillna(0).eq(1)].copy()
    e=e.sort_values("confirmation_date").reset_index(drop=True)
    e["year"]=e.confirmation_date.dt.year
    preds={k:[] for k in ("technical_only","technical_plus_institutional")}
    folds=[]
    for year in sorted(e.year.unique()):
        cutoff=pd.Timestamp(year=int(year)-2,month=12,day=31)
        train=e[e.confirmation_date<=cutoff]
        test=e[e.year==year]
        if len(train)<40 or train.quality_label.nunique()<2 or test.empty:
            folds.append({"year":int(year),"train_n":len(train),"test_n":len(test),"status":"insufficient_matured_training"})
            continue
        fold={"year":int(year),"training_cutoff":cutoff.date().isoformat(),"train_n":int(len(train)),"test_n":int(len(test))}
        for name,cols in (("technical_only",TECH),("technical_plus_institutional",TECH+CHIP)):
            Xtr=train[cols].replace([np.inf,-np.inf],np.nan)
            Xte=test[cols].replace([np.inf,-np.inf],np.nan)
            # Remove features entirely missing in the training fold; no validation-fit preprocessing.
            use=[c for c in cols if Xtr[c].notna().any()]
            m=model(); m.fit(Xtr[use],train.quality_label)
            prob=m.predict_proba(Xte[use])[:,1]
            for (_,row),p in zip(test.iterrows(),prob):
                preds[name].append({"date":row.confirmation_date.date().isoformat(),"year":int(year),
                    "actual":int(row.quality_label),"probability":float(p)})
            fold[name]=score(test.quality_label.to_numpy(),prob)
            fold[name]["feature_count"]=len(use)
        folds.append(fold)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,values in preds.items():
        pd.DataFrame(values).to_csv(out/f"{name}_oof_predictions.csv",index=False)
        rows.append({"model":name,**score([x["actual"] for x in values],[x["probability"] for x in values])})
    comparison=pd.DataFrame(rows)
    comparison.to_csv(out/"institutional_flow_oof_comparison.csv",index=False)
    tech={x["probability"]:x["actual"] for x in preds["technical_only"]}
    chipm={x["probability"]:x["actual"] for x in preds["technical_plus_institutional"]}
    result={
      "symbol":"2330","classification":"ANNUAL_EXPANDING_OOF_EVENT_RANKING_NOT_TRADING_STRATEGY",
      "source_event_count":int(len(e)),
      "data_source":"FinMind TaiwanStockInstitutionalInvestorsBuySellWide; source artifact captured 2026-11-12",
      "institutional_first_date":str(chip.date.min().date()),"institutional_last_date":str(chip.date.max().date()),
      "target":"ATR2 confirmed LOW; complete 60-session close return >0 and MFE >=8%; event labels only",
      "feature_timing":"Institutional daily net flow is joined on confirmation date, after-close signal; next-session-open execution remains hypothetical.",
      "models":{"technical_only":TECH,"technical_plus_institutional":TECH+CHIP},
      "protocol":"Expanding annual OOF, train labels confirmed no later than Dec 31 two years before validation year; minimum 40 matured training events; fixed logistic C=.3; no feature/threshold tuning.",
      "overall_comparison":rows,"annual_folds":folds,
      "institutional_data_coverage":int(len(e)),
      "broker_branch_features":"Excluded; source manifest reports no broker data and no paid token configured.",
      "limitations":["Event ranking is not a portfolio backtest and cannot establish CAGR, win rate or drawdown.",
        "FinMind historical data starts 2012-05-02, so institutional comparison period is shorter than price history.",
        "Institutional flow is observed after the daily close; same-day values cannot be used for earlier execution.",
        "Annual OOF reduces leakage but historical research-family selection bias remains; no pristine blind OOS claim."]
    }
    (out/"tsmc_2330_institutional_flow_oof.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--events",required=True);p.add_argument("--chip",required=True);p.add_argument("--output",default="artifacts/tsmc_2330_institutional_flow_oof")
    a=p.parse_args();print(json.dumps(run(a.events,a.chip,a.output),ensure_ascii=False,indent=2))
