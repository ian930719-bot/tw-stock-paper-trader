"""Annual expanding OOF ablation of foreign and investment-trust flow only.

Dealers, foreign-dealer proprietary flow, broker branches, and all aggregate
institutional totals are excluded by design. Event outcome is diagnostic only.
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
 "confirm_return_1","confirm_return_5","confirm_return_20","confirm_atr_pct14",
 "confirm_rsi14","confirm_macd_hist_change_3","confirm_distance_sma_20",
 "confirm_distance_ema_20","confirm_distance_sma_50","confirm_distance_sma_200",
 "confirm_relative_volume20","confirm_close_location","confirm_intraday_range_pct",
 "confirm_obv_flow_ratio_5","confirm_plus_di14","confirm_minus_di14",
 "confirm_mfi14","confirm_stoch_k14","confirmation_lag_sessions",
 "reversal_at_confirmation_pct",
]
FOREIGN = ["foreign_net_to_volume","foreign_net_5d_lots","foreign_net_20d_lots",
 "foreign_net_positive_streak","foreign_net_negative_streak"]
TRUST = ["trust_net_to_volume","trust_net_5d_lots","trust_net_20d_lots",
 "trust_net_positive_streak","trust_net_negative_streak"]
GROUPS = {"technical_only":TECH,"technical_plus_foreign":TECH+FOREIGN,
 "technical_plus_trust":TECH+TRUST,"technical_plus_foreign_trust":TECH+FOREIGN+TRUST}

def score(y,p):
 y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
 if len(y)==0 or len(np.unique(y))<2:return {"n":int(len(y)),"error":"requires both target classes"}
 n=max(1,int(np.ceil(len(y)*.10)));top=np.argsort(p)[-n:];base=float(y.mean())
 return {"n":int(len(y)),"positive_rate":base,"roc_auc":float(roc_auc_score(y,p)),
  "pr_auc":float(average_precision_score(y,p)),"top_decile_n":int(n),
  "top_decile_precision":float(y[top].mean()),
  "top_decile_lift":float(y[top].mean()/base) if base else None,
  "top_decile_recall":float(y[top].sum()/y.sum()) if y.sum() else None}

def make_model():
 return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
  LogisticRegression(C=.3,class_weight="balanced",max_iter=3000,random_state=2330))

def maturity_dates(events, bars_path):
 bars=pd.read_csv(bars_path,parse_dates=['date']).sort_values('date').reset_index(drop=True)
 date_to_index={pd.Timestamp(d):i for i,d in enumerate(bars.date)}
 indices=events.confirmation_date.map(lambda d:date_to_index.get(pd.Timestamp(d),-1)).to_numpy()
 maturity=[]
 for i in indices:
  j=int(i)+60
  maturity.append(bars.date.iloc[j] if i>=0 and j<len(bars) else pd.NaT)
 return pd.to_datetime(pd.Series(maturity,index=events.index))

def run(events_path,chip_path,bars_path,output):
 e=pd.read_csv(events_path,parse_dates=["confirmation_date"])
 e=e[np.isclose(pd.to_numeric(e.atr_multiplier,errors="coerce"),2.0)].copy()
 e["corporate_action_window_at_confirmation"]=e.corporate_action_window_at_confirmation.astype(str).str.lower().isin(["true","1","yes"])
 e=e[~e.corporate_action_window_at_confirmation]
 e=e[pd.to_numeric(e.observed_sessions_60d,errors="coerce").eq(60)].dropna(subset=["mfe_60d","close_return_60d"]).copy()
 e["quality_label"]=((e.close_return_60d>0)&(e.mfe_60d>=.08)).astype(int)
 chip=pd.read_csv(chip_path,parse_dates=["date"]).sort_values("date").drop_duplicates("date",keep="last")
 chip_cols=["date"]+FOREIGN+TRUST+["institutional_data_available"]
 missing=[c for c in chip_cols if c not in chip.columns]
 if missing:raise ValueError(f"Chip file missing required foreign/trust columns: {missing}")
 # Exact-date join: these flows are available only after confirmation-day close.
 e=e.merge(chip[chip_cols],left_on="confirmation_date",right_on="date",how="left",validate="many_to_one")
 e=e[e.institutional_data_available.fillna(0).eq(1)].sort_values("confirmation_date").reset_index(drop=True)
 e["year"]=e.confirmation_date.dt.year
 e["label_maturity_date"]=maturity_dates(e,bars_path)
 e=e.dropna(subset=["label_maturity_date"]).copy()
 predictions={k:[] for k in GROUPS};folds=[]
 for year in sorted(e.year.unique()):
  cutoff=pd.Timestamp(year=int(year),month=1,day=1)
  train=e[e.label_maturity_date<cutoff];test=e[e.year==year]
  if len(train)<40 or train.quality_label.nunique()<2 or test.empty:
   folds.append({"year":int(year),"train_n":int(len(train)),"test_n":int(len(test)),"status":"insufficient_matured_training"});continue
  fold={"year":int(year),"training_labels_matured_before":cutoff.date().isoformat(),"train_n":int(len(train)),"test_n":int(len(test))}
  for name,cols in GROUPS.items():
   xtr=train[cols].replace([np.inf,-np.inf],np.nan);xte=test[cols].replace([np.inf,-np.inf],np.nan)
   use=[c for c in cols if xtr[c].notna().any()]
   fitted=make_model();fitted.fit(xtr[use],train.quality_label);prob=fitted.predict_proba(xte[use])[:,1]
   for (_,row),p in zip(test.iterrows(),prob):
    predictions[name].append({"date":row.confirmation_date.date().isoformat(),"year":int(year),"actual":int(row.quality_label),"probability":float(p)})
   fold[name]={**score(test.quality_label.to_numpy(),prob),"feature_count":len(use)}
  folds.append(fold)
 out=Path(output);out.mkdir(parents=True,exist_ok=True);overall=[]
 for name,values in predictions.items():
  pd.DataFrame(values).to_csv(out/f"{name}_oof_predictions.csv",index=False)
  overall.append({"model":name,**score([v["actual"] for v in values],[v["probability"] for v in values])})
 pd.DataFrame(overall).to_csv(out/"institutional_flow_oof_comparison.csv",index=False)
 result={"symbol":"2330","classification":"EXPANDING_ANNUAL_OOF_EVENT_RANKING_NOT_STRATEGY_BACKTEST",
  "event_count_after_complete_horizon_and_chip_join":int(len(e)),
  "source":"FinMind TaiwanStockInstitutionalInvestorsBuySellWide; source artifact captured 2026-11-12",
  "institutional_first_date":str(chip.date.min().date()),"institutional_last_date":str(chip.date.max().date()),
  "target":"ATR2 confirmed LOW; full 60-session horizon; close return >0 and MFE >=8%.",
  "timing":"Flow data joined on confirmation date, after close; cannot be used for same-day earlier fills.",
  "features":{"technical":TECH,"foreign_only":FOREIGN,"trust_only":TRUST},
  "excluded":["dealer","foreign dealer proprietary","broker branch","three-institution aggregate"],
  "protocol":"Annual expanding OOF; exact label maturity uses confirmation session + 60 trading sessions from the official historical price calendar; training labels must mature before Jan 1 of validation year; minimum 30 matured training events; fixed logistic C=.3; preprocessing fit only on training fold.",
  "overall_comparison":overall,"annual_folds":folds,
  "limitations":["This ranks pivot-event quality; it is not an executable strategy backtest and provides no CAGR, win rate or drawdown.",
   "Only 46 OOF observations were available in the preceding combined run; small validation folds make annual metrics unstable.",
   "Institutional dataset starts 2012-05-02; full price history is longer.",
   "Historical research-family selection bias remains; no pristine blind test claim."]}
 (out/"tsmc_2330_institutional_flow_oof.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
 return result

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--events",required=True);p.add_argument("--chip",required=True);p.add_argument("--bars",required=True);p.add_argument("--output",default="artifacts/tsmc_2330_institutional_flow_oof")
 a=p.parse_args();print(json.dumps(run(a.events,a.chip,a.bars,a.output),ensure_ascii=False,indent=2))
