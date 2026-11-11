"""Walk-forward split of TSMC swing pivots into small/large waves.

Wave size is |reversal_pct|. Each fold learns LOW/HIGH 35th and 65th
percentile cutoffs using only pivots before its validation window. Models
train on earlier daily rows, calibrate an alert-rate threshold on the preceding
two years, and test on the next calendar year. Signals execute next session
open in all-in/all-out simulations. Exploratory; not a forward independent test.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THRESHOLD_ID="p0.020_atr1.0"
LEADS=(1,3,5,10,20)
START_TEST_YEAR=2016
CALIBRATION_YEARS=2
ALERT_QUANTILE=.97
INITIAL_CAPITAL=100000.0
FEE_RATE=.001425
SELL_TAX_RATE=.003
SLIPPAGE=.001
MIN_COMMISSION=20
INSTITUTIONAL_START=pd.Timestamp("2012-05-02")
WAVE_QUANTILES={"small":.35,"large":.65}
TECH_FEATURES=(
 "return_1","return_3","return_5","return_10","return_20",
 "distance_sma_5","distance_sma_20","distance_sma_50","distance_sma_200",
 "sma20_slope_5","sma50_slope_10","atr_pct14","rsi14","rsi_change_3",
 "macd_pct","macd_hist_pct","macd_hist_change_3","bb_width20","bb_position20",
 "relative_volume20","turnover_ratio20","realized_vol20","realized_vol60",
 "volatility_compression","overnight_gap","intraday_range_pct","close_location",
 "upper_wick_pct","lower_wick_pct","distance_prior_high20","distance_prior_low20",
 "bearish_rsi_divergence20","bullish_rsi_divergence20","bearish_macd_divergence20",
 "bullish_macd_divergence20","bearish_volume_divergence20","bullish_volume_divergence20",
 "stoch_k14","stoch_d3","cci20","williams_r14","adx14","plus_di14","minus_di14",
 "mfi14","chaikin_money_flow20","volume_zscore20","volume_sma5_ratio","volume_sma20_ratio",
 "ppo_hist","donchian_position20","obv_flow_ratio_5","obv_flow_ratio_20")
FLOW_FEATURES=("foreign_net_to_volume","foreign_net_5d_lots","foreign_net_20d_lots",
 "foreign_net_positive_streak","foreign_net_negative_streak","trust_net_to_volume",
 "trust_net_5d_lots","trust_net_20d_lots","trust_net_positive_streak","trust_net_negative_streak")
FEATURE_SETS={"technical":TECH_FEATURES,"technical_plus_foreign_trust":TECH_FEATURES+FLOW_FEATURES}

def make_labels(n,pivots,horizon):
    y=np.zeros(n,dtype=int)
    for p in pivots:
        p=int(p)
        y[max(0,p-horizon):min(n,p)]=1
    known=np.arange(n)<=n-horizon-1
    return y,known

def make_model():
    return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
      LogisticRegression(C=.08,max_iter=4000,class_weight="balanced",random_state=23301112))

def event_size_cutoffs(events, cutoff_date):
    prior=events.loc[events.pivot_date<pd.Timestamp(cutoff_date)].copy()
    result={}
    for side in ("HIGH","LOW"):
        vals=pd.to_numeric(prior.loc[prior.type.astype(str).str.upper()==side,"reversal_pct"],
          errors="coerce").abs().dropna()
        if len(vals)<30: return None
        qsmall=float(vals.quantile(WAVE_QUANTILES["small"]))
        qlarge=float(vals.quantile(WAVE_QUANTILES["large"]))
        if not np.isfinite(qsmall+qlarge) or qsmall>=qlarge: return None
        result[side]={"small_max_abs_reversal_pct":qsmall,
          "large_min_abs_reversal_pct":qlarge,"training_pivots":int(len(vals))}
    return result

def classify_pivots(events,cutoffs):
    e=events.copy()
    e["wave_size"]=pd.to_numeric(e.reversal_pct,errors="coerce").abs()
    e["side"]=e.type.astype(str).str.upper()
    e["wave_class"]="medium"
    for side in ("HIGH","LOW"):
        q=cutoffs[side]
        m=e.side.eq(side)
        e.loc[m&(e.wave_size<=q["small_max_abs_reversal_pct"]),"wave_class"]="small"
        e.loc[m&(e.wave_size>=q["large_min_abs_reversal_pct"]),"wave_class"]="large"
    return e

def commission(notional):
    return max(MIN_COMMISSION,int(math.ceil(notional*FEE_RATE))) if notional>0 else 0

def daily_metrics(y,score,signal):
    y=np.asarray(y,dtype=int);score=np.asarray(score,dtype=float);signal=np.asarray(signal,dtype=bool)
    tp=int(((y==1)&signal).sum());fp=int(((y==0)&signal).sum())
    fn=int(((y==1)&~signal).sum());tn=int(((y==0)&~signal).sum())
    return {"base_rate":float(y.mean()),"signal_rate":float(signal.mean()),
      "precision":tp/(tp+fp) if tp+fp else np.nan,
      "recall":tp/(tp+fn) if tp+fn else np.nan,
      "false_positive_rate":fp/(fp+tn) if fp+tn else np.nan,
      "auc":float(roc_auc_score(y,score)) if len(np.unique(y))==2 else np.nan,
      "tp":tp,"fp":fp,"fn":fn,"tn":tn}

def simulate_all_in_all_out(d,low_signal,high_signal,start_date,capital=INITIAL_CAPITAL):
    dates=d.date
    ix=np.flatnonzero((dates>=pd.Timestamp(start_date)).to_numpy())
    if len(ix)<2: raise ValueError("Not enough rows for execution simulation")
    cash=float(capital);shares=0;entry_cost=0.;entry_date=None
    equity=[];trades=[];conflicts=0
    for k,i in enumerate(ix):
        row=d.iloc[i]
        if k:
            prev=ix[k-1]
            lo=bool(low_signal[prev]);hi=bool(high_signal[prev])
            if lo and hi: conflicts+=1;lo=hi=False
            if shares and hi:
                fill=float(row.raw_open)*(1-SLIPPAGE);notional=shares*fill
                proceeds=notional-commission(notional)-int(math.floor(notional*SELL_TAX_RATE))
                pnl=proceeds-entry_cost
                trades.append({"entry_date":entry_date,"exit_date":row.date.date().isoformat(),
                  "shares":shares,"entry_cost":entry_cost,"exit_proceeds":proceeds,
                  "net_pnl":pnl,"net_return":pnl/entry_cost,"exit_reason":"high_alert_next_open"})
                cash+=proceeds;shares=0;entry_cost=0.;entry_date=None
            elif not shares and lo:
                fill=float(row.raw_open)*(1+SLIPPAGE)
                qty=int(cash//fill) if fill>0 else 0
                while qty:
                    cost=qty*fill+commission(qty*fill)
                    if cost<=cash+1e-8:break
                    qty-=1
                if qty:
                    entry_cost=qty*fill+commission(qty*fill);cash-=entry_cost
                    shares=qty;entry_date=row.date.date().isoformat()
        close=float(row.raw_close)
        equity.append({"date":row.date.date().isoformat(),"equity":cash+shares*close,
          "cash":cash,"shares":shares})
    if shares:
        row=d.iloc[ix[-1]];fill=float(row.raw_close)*(1-SLIPPAGE);notional=shares*fill
        proceeds=notional-commission(notional)-int(math.floor(notional*SELL_TAX_RATE))
        pnl=proceeds-entry_cost
        trades.append({"entry_date":entry_date,"exit_date":row.date.date().isoformat(),
          "shares":shares,"entry_cost":entry_cost,"exit_proceeds":proceeds,
          "net_pnl":pnl,"net_return":pnl/entry_cost,"exit_reason":"end_of_test_close"})
        cash+=proceeds;shares=0;equity[-1].update({"equity":cash,"cash":cash,"shares":0})
    curve=pd.DataFrame(equity);curve["peak"]=curve.equity.cummax()
    curve["drawdown"]=curve.equity/curve.peak-1
    tr=pd.DataFrame(trades)
    elapsed=max((pd.Timestamp(curve.date.iloc[-1])-pd.Timestamp(curve.date.iloc[0])).days/365.25,1/365.25)
    final=float(curve.equity.iloc[-1])
    metrics={"initial_capital":capital,"final_equity":final,"total_return":final/capital-1,
      "cagr":(final/capital)**(1/elapsed)-1 if final>0 else -1,
      "max_drawdown":float(curve.drawdown.min()),"trades":len(tr),
      "win_rate":float((tr.net_pnl>0).mean()) if len(tr) else np.nan,
      "exposure":float((curve.shares>0).mean()),"same_day_conflicts_ignored":conflicts}
    return metrics,tr,curve

def run(features_path,events_path,output_dir):
    d=pd.read_csv(features_path,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    e=pd.read_csv(events_path)
    e=e.loc[e.threshold_id.astype(str)==THRESHOLD_ID].copy()
    e["pivot_date"]=pd.to_datetime(e.pivot_date)
    e["pivot_index"]=pd.to_numeric(e.pivot_index,errors="raise").astype(int)
    if d.date.duplicated().any():raise ValueError("Duplicate feature dates")
    if (e.reversal_pct.isna().mean()>.05):raise ValueError("Too many missing pivot reversal_pct values")
    missing={name:sorted(set(cols)-set(d.columns)) for name,cols in FEATURE_SETS.items()}
    if missing["technical"]:raise ValueError(f"Missing technical columns: {missing['technical']}")
    if missing["technical_plus_foreign_trust"]:raise ValueError(f"Missing flow columns: {missing['technical_plus_foreign_trust']}")
    n=len(d);years=d.date.dt.year.to_numpy();dates=d.date
    last_year=int(years.max());folds=[];predictions=[];cutoff_rows=[]
    for test_year in range(START_TEST_YEAR,last_year+1):
        val_start=pd.Timestamp(f"{test_year-CALIBRATION_YEARS}-01-01")
        cutoffs=event_size_cutoffs(e,val_start)
        if cutoffs is None:continue
        classified=classify_pivots(e.loc[e.pivot_date<val_start],cutoffs)
        test_classified=classify_pivots(e.loc[e.pivot_date>=val_start],cutoffs)
        # Labels for all dates use only pivot events available before the test year,
        # plus retrospective test-year labels solely for held-out scoring.
        fold_events=pd.concat([classified,test_classified],ignore_index=True)
        fold_events=fold_events.drop_duplicates(subset=["pivot_index","side"])
        cutoff_rows.append({"test_year":test_year,"calibration_start":val_start.date().isoformat(),
          "size_cutoffs_json":json.dumps(cutoffs,ensure_ascii=False)})
        for wave in ("small","large"):
          for side in ("HIGH","LOW"):
            piv=fold_events.loc[(fold_events.side==side)&(fold_events.wave_class==wave),"pivot_index"].to_numpy(int)
            for horizon in LEADS:
              y,known=make_labels(n,piv,horizon)
              for model_name,cols in FEATURE_SETS.items():
                eligible=np.ones(n,dtype=bool)
                if model_name=="technical_plus_foreign_trust":
                    eligible &= dates.ge(INSTITUTIONAL_START).to_numpy()
                train_end=np.flatnonzero((dates<val_start).to_numpy())
                val_start_idx=pd.Timestamp(f"{test_year-CALIBRATION_YEARS}-01-01")
                val_end_idx=pd.Timestamp(f"{test_year}-01-01")
                val=np.flatnonzero(((dates>=val_start_idx)&(dates<val_end_idx)).to_numpy()&eligible)
                test=np.flatnonzero((years==test_year)&eligible)
                train=train_end[eligible[train_end]&known[train_end]]
                train=train[train<train_end[-1]-horizon+1] if len(train_end) else train
                if len(train)<500 or len(val)<100 or len(test)<20 or len(np.unique(y[train]))<2:continue
                X=d[list(cols)].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
                model=make_model();model.fit(X.iloc[train],y[train])
                val_scores=model.predict_proba(X.iloc[val])[:,1]
                cutoff=float(np.quantile(val_scores,ALERT_QUANTILE))
                test_scores=model.predict_proba(X.iloc[test])[:,1]
                test_known=known[test]
                eval_y=y[test][test_known];eval_scores=test_scores[test_known]
                eval_signal=eval_scores>=cutoff
                metrics=daily_metrics(eval_y,eval_scores,eval_signal)
                folds.append({"test_year":test_year,"pivot_type":side,"wave_class":wave,
                  "horizon":horizon,"model":model_name,"train_rows":len(train),
                  "validation_rows":len(val),"test_rows":int(test_known.sum()),
                  "alert_quantile":ALERT_QUANTILE,"score_cutoff":cutoff,**metrics})
                for ix,score,sig,kn in zip(test,test_scores,test_scores>=cutoff,test_known):
                    predictions.append({"date":dates.iloc[ix],"year":int(years[ix]),
                      "pivot_type":side,"wave_class":wave,"horizon":horizon,"model":model_name,
                      "score":float(score),"threshold":cutoff,"signal":bool(sig),
                      "label":int(y[ix]) if kn else np.nan,"label_known":bool(kn)})
    fold_df=pd.DataFrame(folds);pred=pd.DataFrame(predictions)
    if fold_df.empty or pred.empty:raise ValueError("No valid small/large wave walk-forward folds")
    summaries=(fold_df.groupby(["pivot_type","wave_class","horizon","model"])
      .agg(year_folds=("test_year","nunique"),mean_auc=("auc","mean"),
       mean_precision=("precision","mean"),mean_recall=("recall","mean"),
       mean_fpr=("false_positive_rate","mean"),mean_base_rate=("base_rate","mean"),
       mean_signal_rate=("signal_rate","mean")).reset_index())
    # Signals are frozen per fold. Compare small-only, large-only, and union
    # policies for every horizon and feature set; no post-hoc best strategy selection.
    barcols=["date","raw_open","raw_high","raw_low","raw_close"]
    bars=d[barcols].copy()
    bt_rows=[];trade_parts=[];curve_parts=[]
    for model_name in FEATURE_SETS:
      for horizon in LEADS:
        z=pred[(pred.model==model_name)&(pred.horizon==horizon)]
        pivot={}
        for side in ("HIGH","LOW"):
          for wave in ("small","large"):
            part=z[(z.pivot_type==side)&(z.wave_class==wave)].set_index("date")
            pivot[(side,wave)]=part.signal.reindex(bars.date).eq(True).to_numpy()
        for policy in ("small_only","large_only","all_waves"):
          if policy=="small_only":lo=pivot[("LOW","small")];hi=pivot[("HIGH","small")]
          elif policy=="large_only":lo=pivot[("LOW","large")];hi=pivot[("HIGH","large")]
          else:
            lo=pivot[("LOW","small")]|pivot[("LOW","large")]
            hi=pivot[("HIGH","small")]|pivot[("HIGH","large")]
          m,tr,curve=simulate_all_in_all_out(bars,lo,hi,f"{START_TEST_YEAR}-01-01")
          m.update({"model":model_name,"horizon":horizon,"wave_policy":policy})
          bt_rows.append(m)
          if len(tr):tr.insert(0,"wave_policy",policy);tr.insert(0,"horizon",horizon);tr.insert(0,"model",model_name);trade_parts.append(tr)
          curve.insert(0,"wave_policy",policy);curve.insert(0,"horizon",horizon);curve.insert(0,"model",model_name);curve_parts.append(curve)
    # Consistent buy-and-hold comparator using same raw open/close and cost assumptions.
    z=bars.loc[bars.date>=pd.Timestamp(f"{START_TEST_YEAR}-01-01")].reset_index(drop=True)
    entry=float(z.raw_open.iloc[0])*(1+SLIPPAGE);qty=int(INITIAL_CAPITAL//entry)
    while qty and qty*entry+commission(qty*entry)>INITIAL_CAPITAL:qty-=1
    cost=qty*entry+commission(qty*entry);spare=INITIAL_CAPITAL-cost
    exit_fill=float(z.raw_close.iloc[-1])*(1-SLIPPAGE);notional=qty*exit_fill
    bh_final=spare+notional-commission(notional)-int(math.floor(notional*SELL_TAX_RATE))
    elapsed=max((z.date.iloc[-1]-z.date.iloc[0]).days/365.25,1/365.25)
    bh_curve=spare+qty*z.raw_close
    bh_dd=float((bh_curve/bh_curve.cummax()-1).min())
    bt_rows.append({"model":"buy_and_hold","horizon":0,"wave_policy":"benchmark",
      "initial_capital":INITIAL_CAPITAL,"final_equity":bh_final,"total_return":bh_final/INITIAL_CAPITAL-1,
      "cagr":(bh_final/INITIAL_CAPITAL)**(1/elapsed)-1,"max_drawdown":bh_dd,
      "trades":1,"win_rate":np.nan,"exposure":1.,"same_day_conflicts_ignored":0})
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    fold_df.to_csv(out/"tsmc_2330_small_large_wave_fold_metrics.csv",index=False)
    summaries.to_csv(out/"tsmc_2330_small_large_wave_model_summary.csv",index=False)
    pred.to_csv(out/"tsmc_2330_small_large_wave_predictions.csv",index=False)
    pd.DataFrame(bt_rows).to_csv(out/"tsmc_2330_small_large_wave_backtest_summary.csv",index=False)
    if trade_parts:pd.concat(trade_parts,ignore_index=True).to_csv(out/"tsmc_2330_small_large_wave_trades.csv",index=False)
    if curve_parts:pd.concat(curve_parts,ignore_index=True).to_csv(out/"tsmc_2330_small_large_wave_equity.csv",index=False)
    meta={"symbol":"2330","threshold_id":THRESHOLD_ID,"rows":n,
      "data_start":dates.min().date().isoformat(),"data_end":dates.max().date().isoformat(),
      "event_rows":len(e),"wave_size_measure":"absolute reversal_pct from pivot event file",
      "classification":"per fold and per side: small <= 35th percentile and large >= 65th percentile of pivot amplitudes observed before validation start; middle 30% omitted from class-specific positives",
      "walk_forward":"expanding train; preceding 2 years calibrate score cutoff; next year test; horizon-purged training labels",
      "alert_quantile":ALERT_QUANTILE,"feature_sets":{k:list(v) for k,v in FEATURE_SETS.items()},
      "institutional":"foreign investors and investment trusts only; eligible from 2012-05-02; dealer excluded",
      "execution":"all available cash, integer shares, next-session open, commission 0.1425% (NT$20 min), sell tax 0.3%, slippage 0.1% each side; no stop, no time exit; forced liquidation at final close",
      "limitations":["Single-stock exploratory analysis; prior full-sample feature research creates selection bias.",
       "Pivot amplitude is retrospective and available only after reversal confirmation.",
       "No cash dividends, corporate-action cashflows, official exchange price-limit/fill simulation, liquidity impact or odd-lot microstructure.",
       "2026 is partial. No independent forward validation or 200% CAGR claim."],
      "fold_cutoffs":cutoff_rows}
    (out/"tsmc_2330_small_large_wave_summary.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    return fold_df,summaries,pd.DataFrame(bt_rows),meta

def main():
    p=argparse.ArgumentParser();p.add_argument("--features",required=True)
    p.add_argument("--events",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();f,s,b,m=run(a.features,a.events,a.output)
    print("MODEL SUMMARY\n",s.to_string(index=False))
    print("\nBACKTEST SUMMARY\n",b.to_string(index=False))
if __name__=="__main__":main()
