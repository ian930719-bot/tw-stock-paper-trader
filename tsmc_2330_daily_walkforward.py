"""Causal walk-forward daily pivot alerts and cost-aware long-only paper simulation.
Signals form at close and execute no earlier than next open. Exploratory only.
Raw-price P&L excludes cash dividends; adjusted indicators are scale-invariant.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THRESHOLD_ID = "p0.020_atr1.0"
LEADS = (1, 3, 5, 10, 20)
CALIBRATION_YEARS = 2
SIGNAL_QUANTILE = 0.95
START_TEST_YEAR = 2016
INITIAL_CAPITAL = 100_000.0
FEE, SELL_TAX, SLIPPAGE = 0.001425, 0.003, 0.001
MAX_HOLD, STOP_ATR = 20, 2.0
INSTITUTIONAL_START = pd.Timestamp("2012-05-02")
FEATURES = {
 "HIGH": {
  "technical": ["distance_sma_200","sma50_slope_10","overnight_gap","bearish_volume_divergence20","return_1"],
  "technical_plus_institutional": ["distance_sma_200","sma50_slope_10","overnight_gap","bearish_volume_divergence20","return_1","foreign_net_to_volume","trust_net_to_volume","trust_net_negative_streak"]},
 "LOW": {
  "technical": ["distance_sma_200","minus_di14","sma50_slope_10","overnight_gap"],
  "technical_plus_institutional": ["distance_sma_200","minus_di14","sma50_slope_10","overnight_gap","foreign_net_to_volume","trust_net_to_volume","trust_net_negative_streak"]},
}

def make_labels(n, pivot_indices, horizon):
    y=np.zeros(n,dtype=int)
    for p in pivot_indices:
        lo,hi=max(0,int(p)-horizon),min(n,int(p))
        if hi>lo: y[lo:hi]=1
    known=np.arange(n)<=n-horizon-1
    return y,known

def make_model():
    return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
        LogisticRegression(C=0.1,max_iter=3000,class_weight="balanced",random_state=23301111))

def run(features_path,events_path,output_dir):
    d=pd.read_csv(features_path,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    e=pd.read_csv(events_path)
    if d.date.duplicated().any() or not d.date.is_monotonic_increasing:
        raise ValueError("Feature dates must be unique and ascending")
    e=e[e.threshold_id.astype(str)==THRESHOLD_ID].copy()
    if e.empty: raise ValueError(f"No events for frozen threshold {THRESHOLD_ID}")
    n=len(d); years=d.date.dt.year.to_numpy(); dates=d.date
    pivots={k:e.loc[e.type.astype(str).str.upper()==k,"pivot_index"].astype(int).to_numpy() for k in ("HIGH","LOW")}
    labels={k:{h:make_labels(n,pivots[k],h) for h in LEADS} for k in ("HIGH","LOW")}
    last_year=int(years.max()); fold_rows=[]; pred_rows=[]
    for kind,sets in FEATURES.items():
      for horizon in LEADS:
       y_all,known=labels[kind][horizon]
       for model_name,cols in sets.items():
        missing=[c for c in cols if c not in d]
        if missing: raise ValueError(f"Missing features: {missing}")
        eligible=np.ones(n,dtype=bool)
        if model_name=="technical_plus_institutional":
            eligible &= dates.ge(INSTITUTIONAL_START).to_numpy()
        X=d[cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
        for test_year in range(START_TEST_YEAR,last_year+1):
            val_start=test_year-CALIBRATION_YEARS
            prior=np.flatnonzero(years<val_start)
            val=np.flatnonzero((years>=val_start)&(years<test_year)&eligible)
            test=np.flatnonzero((years==test_year)&eligible)
            if len(prior)<=horizon or len(val)<80 or len(test)<20: continue
            train=prior[eligible[prior]&known[prior]]
            train=train[train<prior[-1]-horizon+1]  # purge labels crossing into calibration
            if len(train)<300 or len(np.unique(y_all[train]))<2: continue
            model=make_model(); model.fit(X.iloc[train],y_all[train])
            cutoff=float(np.quantile(model.predict_proba(X.iloc[val])[:,1],SIGNAL_QUANTILE))
            scores=model.predict_proba(X.iloc[test])[:,1]; signals=scores>=cutoff
            eval_idx=test[known[test]]; eval_y=y_all[eval_idx]
            eval_score=scores[known[test]]; eval_signal=signals[known[test]]
            auc=float(roc_auc_score(eval_y,eval_score)) if len(np.unique(eval_y))==2 else float("nan")
            tp=int(((eval_y==1)&eval_signal).sum()); fp=int(((eval_y==0)&eval_signal).sum())
            fn=int(((eval_y==1)&~eval_signal).sum()); tn=int(((eval_y==0)&~eval_signal).sum())
            fold_rows.append({"pivot_type":kind,"horizon":horizon,"model":model_name,
              "test_year":test_year,"validation_years":f"{val_start}-{test_year-1}",
              "train_rows":len(train),"validation_rows":len(val),"test_rows":len(eval_idx),
              "base_rate":float(eval_y.mean()),"threshold":cutoff,
              "signal_rate":float(eval_signal.mean()),"auc":auc,
              "precision":tp/(tp+fp) if tp+fp else np.nan,
              "recall":tp/(tp+fn) if tp+fn else np.nan,
              "false_positive_rate":fp/(fp+tn) if fp+tn else np.nan,
              "true_positive":tp,"false_positive":fp,"true_negative":tn,"false_negative":fn})
            for idx,score,sig in zip(test,scores,signals):
                pred_rows.append({"date":dates.iloc[idx],"year":int(years[idx]),
                  "pivot_type":kind,"horizon":horizon,"model":model_name,
                  "score":float(score),"threshold":cutoff,"signal":bool(sig),
                  "label":int(y_all[idx]) if known[idx] else np.nan,"label_known":bool(known[idx])})
    folds=pd.DataFrame(fold_rows); predictions=pd.DataFrame(pred_rows)
    if folds.empty or predictions.empty: raise ValueError("No valid walk-forward folds")
    summary=(folds.groupby(["pivot_type","horizon","model"])
      .agg(year_folds=("test_year","nunique"),mean_auc=("auc","mean"),median_auc=("auc","median"),
       mean_precision=("precision","mean"),mean_recall=("recall","mean"),
       mean_false_positive_rate=("false_positive_rate","mean"),mean_base_rate=("base_rate","mean"),
       mean_signal_rate=("signal_rate","mean")).reset_index())

    # Raw-price fills, adjusted ATR rescaled to raw price, and same cost model for all variants.
    bars=d[["date","raw_open","raw_high","raw_low","raw_close","open","close","atr14"]].copy()
    bars["atr14_raw"]=bars.atr14*bars.raw_close/bars.close
    bars=bars.rename(columns={"raw_open":"open","raw_high":"high","raw_low":"low",
                              "raw_close":"close","atr14_raw":"atr14"})
    bars=bars.loc[:,~bars.columns.duplicated()]
    bars=bars[["date","open","high","low","close","atr14"]].replace([np.inf,-np.inf],np.nan)
    trades=[]; equity_rows=[]; bt_rows=[]
    for model_name in ("technical","technical_plus_institutional"):
      for horizon in LEADS:
        sub=predictions[(predictions.horizon==horizon)&(predictions.model==model_name)]
        hi=sub[sub.pivot_type=="HIGH"].set_index("date")
        lo=sub[sub.pivot_type=="LOW"].set_index("date")
        sig=pd.DataFrame(index=bars.date)
        sig["high"]=hi.signal.reindex(sig.index).eq(True)
        sig["low"]=lo.signal.reindex(sig.index).eq(True)
        cash,qty,cost,entry_date,stop=INITIAL_CAPITAL,0,0.0,None,None
        age,peak,maxdd,returns=0,INITIAL_CAPITAL,0.0,[]
        active=np.flatnonzero((bars.date>=pd.Timestamp(f"{START_TEST_YEAR}-01-01")).to_numpy())
        for i in active:
            row=bars.iloc[i]; prev=sig.loc[bars.date.iloc[i-1]] if i else None
            op,low_px,close=[float(row[k]) if pd.notna(row[k]) else np.nan for k in ("open","low","close")]
            closed=False
            if qty:
                exit_px=reason=None
                if prev is not None and bool(prev.high): exit_px,reason=op,"high_signal_next_open"
                elif age>=MAX_HOLD: exit_px,reason=op,"time_stop_next_open"
                if exit_px is None and stop is not None:
                    if op<=stop: exit_px,reason=op,"gap_stop"
                    elif low_px<=stop: exit_px,reason=stop,"atr_stop"
                if exit_px is not None and np.isfinite(exit_px):
                    gross=qty*exit_px*(1-SLIPPAGE); proceeds=gross*(1-FEE-SELL_TAX)
                    pnl=proceeds-cost; returns.append(pnl/cost)
                    trades.append({"model":model_name,"horizon":horizon,"entry_date":entry_date,
                      "exit_date":row.date,"shares":qty,"entry_cost":cost,"exit_proceeds":proceeds,
                      "net_pnl":pnl,"net_return":pnl/cost,"exit_reason":reason})
                    cash+=proceeds; qty,cost,entry_date,stop,age=0,0.0,None,None,0; closed=True
            if qty==0 and not closed and prev is not None and bool(prev.low) and not bool(prev.high) and np.isfinite(op):
                atr=float(bars.atr14.iloc[i-1]) if i and pd.notna(bars.atr14.iloc[i-1]) else np.nan
                fill=op*(1+SLIPPAGE); shares=int(cash//(fill*(1+FEE))) if fill>0 else 0
                if shares and np.isfinite(atr) and atr>0:
                    gross=shares*fill; cost=gross*(1+FEE); cash-=cost
                    qty,entry_date,stop,age=shares,row.date,fill-STOP_ATR*atr,0
                    if low_px<=stop:
                        px=(op if op<=stop else stop)*(1-SLIPPAGE)
                        sell=qty*px*(1-FEE-SELL_TAX); pnl=sell-cost; returns.append(pnl/cost)
                        trades.append({"model":model_name,"horizon":horizon,"entry_date":entry_date,
                          "exit_date":row.date,"shares":qty,"entry_cost":cost,"exit_proceeds":sell,
                          "net_pnl":pnl,"net_return":pnl/cost,"exit_reason":"entry_day_atr_stop"})
                        cash+=sell; qty,cost,entry_date,stop,age=0,0.0,None,None,0; closed=True
            if qty and entry_date!=row.date: age+=1
            equity=cash+qty*close if np.isfinite(close) else cash
            peak=max(peak,equity); maxdd=min(maxdd,equity/peak-1)
            equity_rows.append({"date":row.date,"model":model_name,"horizon":horizon,
                                "cash":cash,"shares":qty,"close":close,"equity":equity})
        if qty:
            final_close=float(bars.close.iloc[active[-1]])
            gross=qty*final_close*(1-SLIPPAGE); proceeds=gross*(1-FEE-SELL_TAX)
            pnl=proceeds-cost; returns.append(pnl/cost)
            trades.append({"model":model_name,"horizon":horizon,"entry_date":entry_date,
              "exit_date":bars.date.iloc[active[-1]],"shares":qty,"entry_cost":cost,
              "exit_proceeds":proceeds,"net_pnl":pnl,"net_return":pnl/cost,
              "exit_reason":"end_of_sample_liquidation"})
            cash+=proceeds; qty=0
            equity_rows[-1].update({"cash":cash,"shares":0,"equity":cash})
            maxdd=min(maxdd,cash/peak-1 if peak else 0)
        final=cash; start=bars.date.iloc[active[0]]; end=bars.date.iloc[active[-1]]
        elapsed=max((end-start).days/365.25,1/365.25)
        bt_rows.append({"model":model_name,"horizon":horizon,"initial_capital":INITIAL_CAPITAL,
          "final_net_equity":final,"total_return":final/INITIAL_CAPITAL-1,
          "cagr_price_only":(final/INITIAL_CAPITAL)**(1/elapsed)-1 if final>0 else -1,
          "max_drawdown":maxdd,"closed_trades":len(returns),
          "win_rate":float(np.mean(np.array(returns)>0)) if returns else np.nan,
          "mean_trade_net_return":float(np.nanmean(returns)) if returns else np.nan,
          "fee_rate_each_side":FEE,"sell_tax_rate":SELL_TAX,"slippage_each_side":SLIPPAGE,
          "max_holding_sessions":MAX_HOLD,"atr_stop_multiple":STOP_ATR})
    active=np.flatnonzero((bars.date>=pd.Timestamp(f"{START_TEST_YEAR}-01-01")).to_numpy())
    if len(active):
        first,last=bars.iloc[active[0]],bars.iloc[active[-1]]
        fill=float(first.open)*(1+SLIPPAGE); shares=int(INITIAL_CAPITAL//(fill*(1+FEE)))
        cost=shares*fill*(1+FEE); spare=INITIAL_CAPITAL-cost
        gross=shares*float(last.close)*(1-SLIPPAGE); final=gross*(1-FEE-SELL_TAX)+spare
        path=spare+shares*bars.close.iloc[active].to_numpy(dtype=float)
        dd=float(np.min(path/np.maximum.accumulate(path)-1))
        elapsed=max((last.date-first.date).days/365.25,1/365.25)
        bt_rows.append({"model":"buy_and_hold","horizon":0,"initial_capital":INITIAL_CAPITAL,
          "final_net_equity":final,"total_return":final/INITIAL_CAPITAL-1,
          "cagr_price_only":(final/INITIAL_CAPITAL)**(1/elapsed)-1,"max_drawdown":dd,
          "closed_trades":1,"win_rate":np.nan,"mean_trade_net_return":np.nan,
          "fee_rate_each_side":FEE,"sell_tax_rate":SELL_TAX,"slippage_each_side":SLIPPAGE,
          "max_holding_sessions":np.nan,"atr_stop_multiple":np.nan})
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    folds.to_csv(out/"tsmc_2330_daily_walkforward_fold_metrics.csv",index=False)
    summary.to_csv(out/"tsmc_2330_daily_walkforward_summary.csv",index=False)
    predictions.to_csv(out/"tsmc_2330_daily_walkforward_predictions.csv",index=False)
    pd.DataFrame(trades).to_csv(out/"tsmc_2330_cost_aware_trades.csv",index=False)
    pd.DataFrame(equity_rows).to_csv(out/"tsmc_2330_cost_aware_equity.csv",index=False)
    pd.DataFrame(bt_rows).to_csv(out/"tsmc_2330_cost_aware_backtest_summary.csv",index=False)
    result={"symbol":"2330","threshold_id":THRESHOLD_ID,"data_start":dates.min().date().isoformat(),
      "data_end":dates.max().date().isoformat(),"rows":n,"events_high":len(pivots["HIGH"]),
      "events_low":len(pivots["LOW"]),"walk_forward":"Expanding chronological training; preceding two years calibrate a fixed 95th-percentile alert threshold; next year held out. Training labels purged by forecast horizon.",
      "targets":"Retrospective high/low pivot within next 1/3/5/10/20 sessions.",
      "features":FEATURES,"dealer":"excluded","institutional_coverage_start":"2012-05-02",
      "trade_simulation":{"horizons":list(LEADS),"entry":"LOW alert at close; buy next open",
       "exit":"HIGH alert at close; sell next open, 2 ATR stop, or 20-session time stop",
       "initial_capital_twd":INITIAL_CAPITAL,"fee_each_side":FEE,"sell_tax":SELL_TAX,
       "slippage_each_side":SLIPPAGE,"position_size":"all cash, integer shares, no leverage/shorting",
       "price_basis":"raw OHLC fills; scale-invariant adjusted features; excludes cash dividends"},
      "limitations":["Single-stock exploratory study; candidate features were preselected from earlier full-sample pivot analysis.",
       "Retrospective pivot targets are labels, not guaranteed tradable turning points.",
       "P&L excludes dividends, corporate-action cashflows, minimum fees, odd-lot auction microstructure, liquidity impact and official execution reconciliation.",
       "Costs are estimates, not broker-specific charges. 2026 is partial.",
       "No claim of 200% CAGR or future profitability; independent replication and forward paper trading are required."],
      "outputs":["tsmc_2330_daily_walkforward_fold_metrics.csv","tsmc_2330_daily_walkforward_summary.csv",
       "tsmc_2330_daily_walkforward_predictions.csv","tsmc_2330_cost_aware_trades.csv",
       "tsmc_2330_cost_aware_equity.csv","tsmc_2330_cost_aware_backtest_summary.csv"]}
    (out/"tsmc_2330_daily_walkforward_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",required=True); p.add_argument("--events",required=True); p.add_argument("--output",required=True)
    a=p.parse_args(); print(json.dumps(run(a.features,a.events,a.output),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
