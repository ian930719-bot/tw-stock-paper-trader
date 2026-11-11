"""Exploratory all-in/all-out next-open tests of fixed TSMC swing rules."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

THRESHOLD_ID="p0.020_atr1.0"
START_DATE="2016-01-04"
FEE_RATE=.001425
SELL_TAX_RATE=.003
SLIPPAGE=.001
MIN_COMMISSION=20
INITIAL_CAPITAL=100000.0
HORIZONS=(1,3,5,10,20)

def build_rules(d):
    low=(d.relative_volume20<=.8)&(d.distance_prior_low20<=.05)
    obv=low&(d.obv_flow_ratio_5>0)
    green=low&(d.return_1>0)
    bear=(d.bearish_rsi_divergence20>0)|(d.bearish_macd_divergence20>0)|(d.bearish_volume_divergence20>0)
    blow=(d.relative_volume20>=1.5)&(d.return_1<0)&(d.close_location<=.35)
    low_cap=(d.relative_volume20>=1.2)&(d.return_1<=-.01)&(d.stoch_k14<=30)&(d.close_location<=.25)
    low_cap_obv=low_cap&(d.obv_flow_ratio_5<=0)
    low_hv=(d.relative_volume20>=1.2)&(d.return_1<=-.01)&(d.close_location<=.25)
    high_climax=(d.return_1>=.01)&(d.close_location>=.7)&(d.intraday_range_pct>=.016)&(d.distance_sma_200>=.10)
    high_ext=(d.distance_sma_200>=.10)&(d.macd_hist_change_3<=0)&(d.sma50_slope_10>0)
    high_both=high_climax&high_ext
    return {
      "dryup_vs_bear_div":(low,bear),
      "dryup_OBV_vs_blowoff":(obv,blow),
      "dryup_green_vs_bear_div":(green,bear),
      "dryup_vs_bear_or_blowoff":(low,bear|blow),
      "dryup_OBV_vs_bear_div":(obv,bear),
      "dryup_green_vs_blowoff":(green,blow),
      "capitulation_vs_climax":(low_cap,high_climax),
      "capitulation_OBV_vs_climax":(low_cap_obv,high_climax),
      "high_volume_selloff_vs_climax":(low_hv,high_climax),
      "capitulation_vs_climax_extension":(low_cap,high_both),
      "capitulation_OBV_vs_climax_extension":(low_cap_obv,high_both),
      "high_volume_selloff_vs_climax_extension":(low_hv,high_both),
      "capitulation_OBV_vs_extension_fade":(low_cap_obv,high_ext),
    }

def commission(notional):
    return max(MIN_COMMISSION,int(math.ceil(notional*FEE_RATE))) if notional>0 else 0

def forward_label(n,pivots,horizon):
    y=np.zeros(n,dtype=bool)
    for p in pivots:
        p=int(p); y[max(0,p-horizon):p]=True
    return y

def alert_metrics(signal,pivots,n,dates,horizon):
    y=forward_label(n,pivots,horizon)
    mask=((dates>=pd.Timestamp(START_DATE)).to_numpy()&(np.arange(n)<=n-horizon-1))
    s=np.asarray(signal.fillna(False),dtype=bool)&mask
    truth=y&mask
    tp=int((s&truth).sum()); fp=int((s&~truth).sum())
    fn=int((~s&truth).sum()); tn=int((~s&~truth).sum())
    return {"rows":int(mask.sum()),"signal_days":int(s.sum()),"base_rate":float(truth.mean()),
      "precision":tp/(tp+fp) if tp+fp else np.nan,
      "recall":tp/(tp+fn) if tp+fn else np.nan,
      "false_positive_rate":fp/(fp+tn) if fp+tn else np.nan,
      "true_positive":tp,"false_positive":fp,"false_negative":fn,"true_negative":tn}

def simulate(d,low_signal,high_signal,start=START_DATE,end=None,capital=INITIAL_CAPITAL):
    d=d.sort_values("date").reset_index(drop=True)
    end=end or d.date.max().date().isoformat()
    ix=np.flatnonzero(((d.date>=pd.Timestamp(start))&(d.date<=pd.Timestamp(end))).to_numpy())
    if len(ix)<2: raise ValueError("Insufficient rows in requested backtest period")
    cash=float(capital); shares=0; entry_cost=0.; entry_date=None
    eq_rows=[]; trade_rows=[]; conflicts=0
    for k,i in enumerate(ix):
        row=d.iloc[i]
        if k>0:
            prev=ix[k-1]
            lo=bool(low_signal.iloc[prev]) if pd.notna(low_signal.iloc[prev]) else False
            hi=bool(high_signal.iloc[prev]) if pd.notna(high_signal.iloc[prev]) else False
            if lo and hi: conflicts+=1; lo=hi=False
            if shares>0 and hi:
                raw=float(row.raw_open); fill=raw*(1-SLIPPAGE); notional=shares*fill
                fee=commission(notional); tax=int(math.floor(notional*SELL_TAX_RATE))
                proceeds=notional-fee-tax; pnl=proceeds-entry_cost
                trade_rows.append({"entry_date":entry_date,"exit_date":row.date.date().isoformat(),
                  "shares":shares,"entry_cost":entry_cost,"exit_proceeds":proceeds,"net_pnl":pnl,
                  "net_return":pnl/entry_cost,"exit_reason":"high_signal_next_open"})
                cash+=proceeds; shares=0; entry_cost=0.; entry_date=None
            elif shares==0 and lo:
                raw=float(row.raw_open); fill=raw*(1+SLIPPAGE); qty=int(cash//fill)
                while qty>0:
                    notional=qty*fill
                    if notional+commission(notional)<=cash+1e-8: break
                    qty-=1
                if qty>0:
                    notional=qty*fill; fee=commission(notional)
                    entry_cost=notional+fee; cash-=entry_cost
                    shares=qty; entry_date=row.date.date().isoformat()
        eq_rows.append({"date":row.date.date().isoformat(),"equity":cash+shares*float(row.raw_close),
          "cash":cash,"shares":shares})
    if shares:
        row=d.iloc[ix[-1]]; fill=float(row.raw_close)*(1-SLIPPAGE); notional=shares*fill
        proceeds=notional-commission(notional)-int(math.floor(notional*SELL_TAX_RATE))
        pnl=proceeds-entry_cost
        trade_rows.append({"entry_date":entry_date,"exit_date":row.date.date().isoformat(),
          "shares":shares,"entry_cost":entry_cost,"exit_proceeds":proceeds,"net_pnl":pnl,
          "net_return":pnl/entry_cost,"exit_reason":"end_of_test_close"})
        cash+=proceeds; eq_rows[-1].update({"equity":cash,"cash":cash,"shares":0})
    curve=pd.DataFrame(eq_rows); curve["peak"]=curve.equity.cummax()
    curve["drawdown"]=curve.equity/curve.peak-1
    trades=pd.DataFrame(trade_rows)
    elapsed=(pd.Timestamp(curve.date.iloc[-1])-pd.Timestamp(curve.date.iloc[0])).days/365.25
    final=float(curve.equity.iloc[-1])
    metrics={"start":curve.date.iloc[0],"end":curve.date.iloc[-1],"initial_capital":capital,
      "final_equity":final,"total_return":final/capital-1,
      "cagr":(final/capital)**(1/elapsed)-1 if elapsed>0 and final>0 else np.nan,
      "max_drawdown":float(curve.drawdown.min()),"closed_trades":len(trades),
      "win_rate":float((trades.net_pnl>0).mean()) if len(trades) else np.nan,
      "mean_trade_return":float(trades.net_return.mean()) if len(trades) else np.nan,
      "exposure":float((curve.shares>0).mean()),"same_day_signal_conflicts_ignored":conflicts}
    return metrics,trades,curve

def buy_hold(d,start=START_DATE,end=None,capital=INITIAL_CAPITAL):
    end=end or d.date.max().date().isoformat()
    z=d.loc[(d.date>=pd.Timestamp(start))&(d.date<=pd.Timestamp(end))].sort_values("date").reset_index(drop=True)
    if len(z)<2: raise ValueError("Insufficient buy-and-hold rows")
    entry=float(z.raw_open.iloc[0])*(1+SLIPPAGE); qty=int(capital//entry)
    while qty>0:
        cost=qty*entry+commission(qty*entry)
        if cost<=capital+1e-8: break
        qty-=1
    cost=qty*entry+commission(qty*entry); cash=capital-cost
    curve=cash+qty*z.raw_close
    px=float(z.raw_close.iloc[-1])*(1-SLIPPAGE); notional=qty*px
    proceeds=notional-commission(notional)-int(math.floor(notional*SELL_TAX_RATE))
    final=cash+proceeds; curve.iloc[-1]=final
    elapsed=(z.date.iloc[-1]-z.date.iloc[0]).days/365.25
    metrics={"start":z.date.iloc[0].date().isoformat(),"end":z.date.iloc[-1].date().isoformat(),
      "initial_capital":capital,"final_equity":float(final),"total_return":final/capital-1,
      "cagr":(final/capital)**(1/elapsed)-1 if elapsed>0 and final>0 else np.nan,
      "max_drawdown":float((curve/curve.cummax()-1).min()),"closed_trades":1,
      "win_rate":1.,"mean_trade_return":(proceeds-cost)/cost if cost>0 else np.nan,
      "exposure":1.,"same_day_signal_conflicts_ignored":0}
    eq=pd.DataFrame({"date":z.date.dt.date.astype(str),"equity":curve,"cash":np.nan,"shares":qty})
    eq["peak"]=eq.equity.cummax(); eq["drawdown"]=eq.equity/eq.peak-1
    trade=pd.DataFrame([{"entry_date":z.date.iloc[0].date().isoformat(),
      "exit_date":z.date.iloc[-1].date().isoformat(),"shares":qty,"entry_cost":cost,
      "exit_proceeds":proceeds,"net_pnl":proceeds-cost,"net_return":(proceeds-cost)/cost,
      "exit_reason":"buy_and_hold_end"}])
    return metrics,trade,eq

def run(features_path,events_path,output_dir):
    d=pd.read_csv(features_path,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    required={"raw_open","raw_high","raw_low","raw_close","relative_volume20","distance_prior_low20",
      "obv_flow_ratio_5","return_1","close_location","bearish_rsi_divergence20",
      "bearish_macd_divergence20","bearish_volume_divergence20"}
    missing=sorted(required-set(d.columns))
    if missing: raise ValueError(f"Missing required columns: {missing}")
    if d.date.duplicated().any(): raise ValueError("Duplicate dates")
    e=pd.read_csv(events_path); e=e.loc[e.threshold_id.astype(str)==THRESHOLD_ID].copy()
    if e.empty: raise ValueError(f"No events for {THRESHOLD_ID}")
    high=e.loc[e.type.astype(str).str.upper()=="HIGH","pivot_index"].astype(int).to_numpy()
    low=e.loc[e.type.astype(str).str.upper()=="LOW","pivot_index"].astype(int).to_numpy()
    alerts=[]; summaries=[]; trades_all=[]; equity_all=[]
    action_window=d.corporate_action_window.fillna(True).astype(bool) if "corporate_action_window" in d else pd.Series(False,index=d.index)
    for name,(lo,hi) in build_rules(d).items():
        lo=lo.fillna(False)&~action_window; hi=hi.fillna(False)&~action_window
        for kind,sig,piv in (("LOW",lo,low),("HIGH",hi,high)):
            for h in HORIZONS:
                alerts.append({"strategy":name,"pivot_type":kind,"horizon_sessions":h,
                  **alert_metrics(sig,piv,len(d),d.date,h)})
        m,t,c=simulate(d,lo,hi); m.update({"strategy":name,"type":"all_in_all_out"})
        summaries.append(m)
        if len(t): t.insert(0,"strategy",name); trades_all.append(t)
        c.insert(0,"strategy",name); equity_all.append(c)
    m,t,c=buy_hold(d); m.update({"strategy":"buy_and_hold","type":"benchmark"})
    summaries.append(m); t.insert(0,"strategy","buy_and_hold"); trades_all.append(t)
    c.insert(0,"strategy","buy_and_hold"); equity_all.append(c)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(alerts).to_csv(out/"tsmc_2330_trader_rule_daily_alert_metrics.csv",index=False)
    pd.DataFrame(summaries).to_csv(out/"tsmc_2330_all_in_all_out_backtest_summary.csv",index=False)
    pd.concat(trades_all,ignore_index=True).to_csv(out/"tsmc_2330_all_in_all_out_trades.csv",index=False)
    pd.concat(equity_all,ignore_index=True).to_csv(out/"tsmc_2330_all_in_all_out_equity.csv",index=False)
    return pd.DataFrame(summaries),pd.DataFrame(alerts)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--features",required=True); p.add_argument("--events",required=True)
    p.add_argument("--output",default="artifacts/all_in_all_out"); a=p.parse_args()
    summary,_=run(a.features,a.events,a.output); print(summary.to_string(index=False))
if __name__=="__main__": main()
