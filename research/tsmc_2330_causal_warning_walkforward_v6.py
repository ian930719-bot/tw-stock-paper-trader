from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

FEATURES = ["rsi","macdh_pct","atrp","ret5","ret10","ret20","ret40","dd20","dd40","ma20dev","ma60dev","vol5","vol20","bbpct","bbwidth","rsi_chg5","macdh_chg5","vol_chg5","atr_chg5"]
HORIZON, MOVE = 10, .08
BUY_FEE, SELL_FEE_TAX, SLIPPAGE = .001425, .004425, .001

def build_data(path):
    d = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    for col in ["open","high","low","close","volume"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    c,h,l,v = d.close,d.high,d.low,d.volume
    change = c.diff()
    up = change.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-change.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["rsi"] = 100 - 100/(1 + up/down.replace(0,np.nan))
    macd = c.ewm(span=12,adjust=False).mean() - c.ewm(span=26,adjust=False).mean()
    d["macdh_pct"] = (macd-macd.ewm(span=9,adjust=False).mean())/c
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    d["atrp"] = tr.rolling(14).mean()/c
    for n in [5,10,20,40,60]:
        d[f"ret{n}"] = c.pct_change(n)
        d[f"ma{n}dev"] = c/c.rolling(n).mean()-1
        d[f"vol{n}"] = v/v.rolling(n).mean()
    d["dd20"] = c/h.rolling(20).max()-1
    d["dd40"] = c/h.rolling(40).max()-1
    d["bbpct"] = (c-c.rolling(20).mean())/(2*c.rolling(20).std())
    d["bbwidth"] = 4*c.rolling(20).std()/c.rolling(20).mean()
    d["rsi_chg5"] = d.rsi.diff(5)
    d["macdh_chg5"] = d.macdh_pct.diff(5)
    d["vol_chg5"] = d.vol5.diff(5)
    d["atr_chg5"] = d.atrp.pct_change(5)
    future_high = pd.concat([h.shift(-k) for k in range(1,HORIZON+1)],axis=1).max(axis=1)
    future_low = pd.concat([l.shift(-k) for k in range(1,HORIZON+1)],axis=1).min(axis=1)
    d["y_low"] = (future_high/c-1 >= MOVE).astype(float)
    d["y_high"] = (future_low/c-1 <= -MOVE).astype(float)
    d.loc[d.index >= len(d)-HORIZON, ["y_low","y_high"]] = np.nan
    return d

def fit_model(train, target):
    z = train.dropna(subset=[target])
    positives = int(z[target].sum())
    if len(z)<500 or z[target].nunique()<2 or positives<20 or len(z)-positives<20:
        return None
    model = Pipeline([
        ("impute",SimpleImputer(strategy="median",add_indicator=True)),
        ("scale",StandardScaler()),
        ("lr",LogisticRegression(C=.15,class_weight="balanced",max_iter=2000,random_state=2330))
    ])
    model.fit(z[FEATURES],z[target].astype(int))
    return model

def run(input_csv, output_dir):
    d=build_data(input_csv)
    d["p_low"],d["p_high"]=np.nan,np.nan
    training=[]
    for year in sorted(d.date.dt.year.unique()):
        if year<2016: continue
        test_mask=d.date.dt.year.eq(year)
        test_start=int(d.index[test_mask][0])
        # A training label is eligible only if all 10 future sessions ended before test_start.
        train_mask=(d.index+HORIZON<test_start)
        train=d.loc[train_mask & d.y_low.notna()]
        low_model=fit_model(train,"y_low")
        high_model=fit_model(train,"y_high")
        if low_model is None or high_model is None:
            training.append({"year":int(year),"train_rows":len(train),"status":"insufficient training"})
            continue
        pred_idx=d.index[test_mask & d[FEATURES].notna().any(axis=1)]
        d.loc[pred_idx,"p_low"]=low_model.predict_proba(d.loc[pred_idx,FEATURES])[:,1]
        d.loc[pred_idx,"p_high"]=high_model.predict_proba(d.loc[pred_idx,FEATURES])[:,1]
        training.append({"year":int(year),"train_rows":len(train),
            "low_positive_rate":float(train.y_low.mean()),"high_positive_rate":float(train.y_high.mean()),"status":"fit"})
    # OOS-only account. Signals are generated at close and filled at next session open.
    cash,shares,entry,entry_idx=100000.,0,0.,None
    trades,equity=[],[]
    pending=None
    oos_start=int(d.index[d.p_low.notna() & d.p_high.notna()][0])
    for i,row in d.iloc[oos_start:].iterrows():
        if pending:
            action=pending["action"]
            px=float(row.open)*(1+SLIPPAGE if action=="buy" else 1-SLIPPAGE)
            if action=="buy" and shares==0:
                qty=int(cash/(px*(1+BUY_FEE)))
                if qty:
                    cash-=qty*px*(1+BUY_FEE); shares=qty; entry=px; entry_idx=i
                    trades.append({"entry_date":str(row.date.date()),"entry_price":px,"shares":qty})
            elif action=="sell" and shares:
                proceeds=shares*px*(1-SELL_FEE_TAX); cash+=proceeds
                trades[-1].update({"exit_date":str(row.date.date()),"exit_price":px,
                    "net_trade_return":proceeds/(shares*entry*(1+BUY_FEE))-1,"hold_days":i-entry_idx})
                shares,entry,entry_idx=0,0.,None
            pending=None
        equity.append({"date":row.date,"equity":cash+shares*float(row.close)})
        if shares:
            held=i-entry_idx
            if row.close<=entry*.92 or held>=60 or (pd.notna(row.p_high) and row.p_high>=.60):
                pending={"action":"sell"}
        elif pd.notna(row.p_low) and row.p_low>=.65:
            pending={"action":"buy"}
    if shares:
        last=d.iloc[-1]; px=float(last.close)*(1-SLIPPAGE); proceeds=shares*px*(1-SELL_FEE_TAX); cash+=proceeds
        trades[-1].update({"exit_date":str(last.date.date()),"exit_price":px,
            "net_trade_return":proceeds/(shares*entry*(1+BUY_FEE))-1,"hold_days":len(d)-1-entry_idx})
    eq=pd.DataFrame(equity)
    eq.loc[len(eq)-1,"equity"]=cash
    span=(d.date.iloc[-1]-d.date.iloc[oos_start]).days/365.25
    strategy_cagr=(cash/100000)**(1/span)-1
    strategy_mdd=float((eq.equity/eq.equity.cummax()-1).min())
    bh_shares=100000/(float(d.open.iloc[oos_start])*(1+BUY_FEE))
    bh_final=bh_shares*float(d.close.iloc[-1])*(1-SELL_FEE_TAX)
    bh_curve=(bh_shares*d.close.iloc[oos_start:]*(1-SELL_FEE_TAX))
    bh_mdd=float((bh_curve/bh_curve.cummax()-1).min())
    closed=[t for t in trades if "exit_date" in t]
    summary={"symbol":"2330","rows":len(d),"history_start":str(d.date.min().date()),
      "oos_start":str(d.date.iloc[oos_start].date()),"data_end":str(d.date.iloc[-1].date()),
      "source":"Yahoo Finance auto_adjust=True","initial_capital":100000,
      "targets":{"low_warning":"within next 10 sessions, a future daily high reaches +8% from signal close",
                 "high_warning":"within next 10 sessions, a future daily low reaches -8% from signal close"},
      "features":FEATURES,"walk_forward":"expanding yearly; training labels mature before each test year; close signal, next open fill",
      "fixed_rules":{"buy_probability":.65,"sell_probability":.60,"close_stop_loss":.08,"max_hold_sessions":60},
      "costs":{"buy_fee":BUY_FEE,"sell_fee_and_tax":SELL_FEE_TAX,"slippage_each_side":SLIPPAGE},
      "strategy_final_equity":float(cash),"strategy_cagr":float(strategy_cagr),
      "strategy_max_drawdown":strategy_mdd,"buy_hold_final_equity":float(bh_final),
      "buy_hold_cagr":float((bh_final/100000)**(1/span)-1),"buy_hold_max_drawdown":bh_mdd,
      "closed_trades":len(closed),"win_rate":float(np.mean([t["net_trade_return"]>0 for t in closed])) if closed else None,
      "average_trade_return":float(np.mean([t["net_trade_return"] for t in closed])) if closed else None,
      "yearly_training":training,
      "limitations":["Single-stock exploratory model; no claim of generalization.",
        "Forward labels are training targets only and never features; training rows are purged by the full 10-session label horizon.",
        "Two labels may both be true within the same 10-session window; this is an excursion-warning classifier, not an exact pivot classifier.",
        "Adjusted Yahoo OHLCV and simplified next-open fills are not official execution-grade data.",
        "Thresholds are fixed in advance for this experiment and were not optimized on the OOS period."]}
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    d[["date","close","p_low","p_high","y_low","y_high"]].to_csv(out/"walk_forward_daily_scores.csv",index=False)
    pd.DataFrame(trades).to_csv(out/"walk_forward_trades.csv",index=False)
    eq.to_csv(out/"equity_curve.csv",index=False)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="data/top10_adjusted_prices/bars/2330.csv")
    p.add_argument("--output",default="artifacts/tsmc_swing_v6")
    a=p.parse_args()
    print(json.dumps(run(a.input,a.output),ensure_ascii=False,indent=2))
