"""Causal swing-low scoring and shared-capital validation on adjusted Taiwan OHLCV.

The low label is retrospective and is used only for model training/evaluation:
within the next 40 sessions, price must reach +15% upside while avoiding an
8% adverse excursion. The live signal uses only features known at the close.
All fills are simulated on the next session's adjusted open; this is research,
not a broker-fill replay or a claim of guaranteed future performance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SYMBOLS = ["2330", "2454", "1303", "2327", "2408", "2308", "3037", "3008", "2383", "2303"]
NAMES = {"2330":"台積電","2454":"聯發科","1303":"南亞","2327":"國巨","2408":"南亞科","2308":"台達電","3037":"欣興","3008":"大立光","2383":"台光電","2303":"聯電"}
FEATURES = [
    "rsi14","atr_pct","dist_sma20","dist_sma50","dist_sma100","dist_sma200",
    "mom20","mom60","mom120","dd_high20","dd_high60","dd_high120",
    "dist_low20","dist_low60","dist_low120","vol_ratio","close_loc","gap",
    "bb_width_pct","obv_slope","breadth200",
]
BUY_FEE = 0.001425
SELL_FEE_TAX = 0.001425 + 0.003
SLIPPAGE = 0.001
INITIAL_CAPITAL = 1_000_000.0
TRAIN_END = "2020-11-01"
VALIDATION_START = "2021-03-01"
VALIDATION_END = "2023-11-01"
TEST_START = "2024-03-01"


def future_extreme(series: pd.Series, window: int, operation: str) -> pd.Series:
    """Future-only rolling extreme, excluding the current row."""
    reverse = series.shift(-1).iloc[::-1]
    rolling = reverse.rolling(window, min_periods=window)
    return getattr(rolling, operation)().iloc[::-1]


def load_adjusted_frames(input_dir: str | Path) -> tuple[dict[str, pd.DataFrame], dict]:
    root = Path(input_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("缺少 manifest.json，無法確認還原股價資料來源")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("collection_complete") or int(manifest.get("symbols_collected", 0)) != 10:
        raise ValueError("十檔還原股價資料未完整收集")
    frames = {}
    for symbol in SYMBOLS:
        path = root / "bars" / f"{symbol}.csv"
        if not path.exists():
            raise ValueError(f"缺少 {symbol}.csv")
        d = pd.read_csv(path)
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        for col in ("open","high","low","close","volume"):
            d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna(subset=["date","open","high","low","close","volume"])
        valid = (
            (d[["open","high","low","close"]] > 0).all(axis=1)
            & (d.volume >= 0)
            & (d.high >= d[["open","close","low"]].max(axis=1))
            & (d.low <= d[["open","close","high"]].min(axis=1))
        )
        d = d.loc[valid].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        if len(d) < 1000:
            raise ValueError(f"{symbol} 有效資料少於 1,000 根")
        d["symbol"] = symbol
        d["name"] = NAMES[symbol]
        frames[symbol] = d
    return frames, manifest


def build_feature_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build same-close features and strictly future swing labels."""
    rows = []
    for symbol, raw in frames.items():
        d = raw.sort_values("date").copy().reset_index(drop=True)
        c,h,l,o,v = d.close,d.high,d.low,d.open,d.volume
        tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14,adjust=False).mean()
        delta = c.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
        d["rsi14"] = 100 - 100/(1+gain/loss.replace(0,np.nan))
        d["atr"] = atr
        d["atr_pct"] = atr/c
        for n in (20,50,100,200):
            sma = c.rolling(n,min_periods=n).mean()
            d[f"sma{n}"] = sma
            d[f"dist_sma{n}"] = c/sma-1
        for n in (20,60,120):
            d[f"mom{n}"] = c.pct_change(n)
            prior_high = h.shift(1).rolling(n,min_periods=n).max()
            prior_low = l.shift(1).rolling(n,min_periods=n).min()
            d[f"dd_high{n}"] = c/prior_high-1
            d[f"dist_low{n}"] = c/prior_low-1
        d["vol_ratio"] = v/v.shift(1).rolling(20,min_periods=20).mean()
        d["close_loc"] = (c-l)/(h-l).replace(0,np.nan)
        d["gap"] = o/c.shift()-1
        sd = c.rolling(20,min_periods=20).std(ddof=0)
        bb_width = 4*sd/c.rolling(20,min_periods=20).mean()
        d["bb_width_pct"] = bb_width.rolling(120,min_periods=60).rank(pct=True)
        obv = (np.sign(c.diff()).fillna(0)*v).cumsum()
        d["obv_slope"] = obv.diff(20)/v.rolling(20,min_periods=20).sum().replace(0,np.nan)
        # This is an adjusted-price x reported-volume ADV proxy, not raw turnover.
        d["adv20_proxy"] = (c*v).shift(1).rolling(20,min_periods=20).mean()
        d["future_max40"] = future_extreme(h,40,"max")
        d["future_min40"] = future_extreme(l,40,"min")
        d["low_label40"] = (
            (d.future_max40/c-1 >= .15) & (d.future_min40/c-1 >= -.08)
        )
        d["high_label40"] = (
            (d.future_min40/c-1 <= -.15) & (d.future_max40/c-1 <= .08)
        )
        d["symbol"] = symbol
        d["name"] = NAMES[symbol]
        rows.append(d)
    panel = pd.concat(rows,ignore_index=True).sort_values(["date","symbol"]).reset_index(drop=True)
    above = panel.assign(above200=panel.dist_sma200>0).groupby("date").above200.mean()
    panel["breadth200"] = panel.date.map(above)
    return panel


def apply_cooldown(panel: pd.DataFrame, signal_col: str, cooldown: int = 40) -> pd.Series:
    result = pd.Series(False,index=panel.index)
    for _, group in panel.groupby("symbol",sort=False):
        last = -cooldown
        for idx, flag in zip(group.index,group[signal_col].fillna(False).to_numpy()):
            if flag and idx-last >= cooldown:
                result.loc[idx] = True
                last = idx
    return result


def fit_probability_model(panel: pd.DataFrame, label: str) -> tuple[object, dict]:
    train = panel.loc[panel.date < pd.Timestamp(TRAIN_END)].dropna(subset=FEATURES+[label])
    val = panel.loc[
        (panel.date >= pd.Timestamp(VALIDATION_START))
        & (panel.date < pd.Timestamp(VALIDATION_END))
    ].dropna(subset=FEATURES+[label])
    test = panel.loc[panel.date >= pd.Timestamp(TEST_START)].dropna(subset=FEATURES+[label])
    if len(train)<1000 or len(val)<500 or len(test)<500:
        raise ValueError("訓練/驗證/測試樣本不足")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=.01,class_weight="balanced",max_iter=1000),
    )
    model.fit(train[FEATURES],train[label].astype(int))
    p_val = model.predict_proba(val[FEATURES])[:,1]
    p_test = model.predict_proba(test[FEATURES])[:,1]
    report = {
        "train_rows":int(len(train)),"validation_rows":int(len(val)),"test_rows":int(len(test)),
        "validation_auc":float(roc_auc_score(val[label],p_val)),
        "test_auc":float(roc_auc_score(test[label],p_test)),
        "validation_average_precision":float(average_precision_score(val[label],p_val)),
        "test_average_precision":float(average_precision_score(test[label],p_test)),
        "validation_positive_rate":float(val[label].mean()),
        "test_positive_rate":float(test[label].mean()),
        "validation_threshold_95pct":float(np.quantile(p_val,.95)),
    }
    imputer, scaler, classifier = model.named_steps["simpleimputer"],model.named_steps["standardscaler"],model.named_steps["logisticregression"]
    formula = {
        "type":"logistic probability over standardized same-close features",
        "equation":"p = 1 / (1 + exp(-(intercept + sum(coef_i * ((x_i - mean_i) / scale_i)))))",
        "intercept":float(classifier.intercept_[0]),
        "features":[
            {"name":name,"imputer_median":float(imputer.statistics_[i]),
             "mean":float(scaler.mean_[i]),"scale":float(scaler.scale_[i]),
             "coefficient":float(classifier.coef_[0][i])}
            for i,name in enumerate(FEATURES)
        ],
    }
    scores = pd.Series(np.nan,index=panel.index,dtype=float)
    scores.loc[val.index] = p_val
    scores.loc[test.index] = p_test
    return model, {"report":report,"formula":formula,"scores":scores,"validation":val,"test":test}


def simulate_portfolio(
    panel: pd.DataFrame, start: str, end: str, *,
    hold_bars: int, trail_atr: float, risk_fraction: float,
    max_positions: int = 5, max_weight: float = .40,
    adv_participation: float = .01, initial_capital: float = INITIAL_CAPITAL,
) -> tuple[dict,pd.DataFrame,pd.DataFrame]:
    """Cash-only shared-capital simulation; signal close -> next open."""
    a,b = pd.Timestamp(start),pd.Timestamp(end)
    row_lookup = {
        s:{pd.Timestamp(r.date):r for r in g.sort_values("date").itertuples(index=False)}
        for s,g in panel.groupby("symbol")
    }
    all_dates = sorted(set().union(*[set(x) for x in row_lookup.values()]))
    dates = [d for d in all_dates if a<=d<=b]
    last_dates = {s:max((d for d in lookup if a<=d<=b),default=None) for s,lookup in row_lookup.items()}
    schedules = {}
    for symbol,g in panel.groupby("symbol"):
        g=g.sort_values("date").reset_index(drop=True)
        for i in np.flatnonzero(g.signal_low_cooldown.fillna(False).to_numpy()):
            if i+1>=len(g): continue
            sig=g.iloc[i]
            entry_date=pd.Timestamp(g.date.iloc[i+1])
            signal_date=pd.Timestamp(sig.date)
            if not(a<=signal_date<entry_date<=b): continue
            if pd.isna(sig.atr) or sig.atr<=0 or pd.isna(sig.adv20_proxy) or sig.adv20_proxy<=0: continue
            schedules.setdefault(entry_date,[]).append({
                "symbol":symbol,"signal_date":signal_date,"atr":float(sig.atr),
                "adv20":float(sig.adv20_proxy),"score":float(sig.low_score),
            })
    for day in schedules:
        schedules[day].sort(key=lambda x:x["score"],reverse=True)
    cash=float(initial_capital)
    positions={}
    last_close={}
    trades=[]
    curve=[]
    for day in dates:
        today={s:lookup[day] for s,lookup in row_lookup.items() if day in lookup}
        closed_today=set()
        # Pending exits and overnight gaps are filled before new entries.
        for symbol in list(positions):
            if symbol not in today: continue
            p=positions[symbol]; row=today[symbol]; op=float(row.open)
            if p["pending"]:
                fill,reason=op,p["pending_reason"]
            elif op<=p["stop"]:
                fill,reason=op,"gap_stop"
            else:
                fill=reason=None
            if fill is not None:
                fill*=1-SLIPPAGE
                proceeds=p["shares"]*fill*(1-SELL_FEE_TAX)
                trades.append({**p,"symbol":symbol,"exit_date":day,"exit_price":fill,
                    "exit_reason":reason,"net_return":(fill/p["entry_price"])*(1-BUY_FEE)*(1-SELL_FEE_TAX)-1,
                    "pnl":proceeds-p["entry_cost"]})
                cash+=proceeds; del positions[symbol]; closed_today.add(symbol)
        # New positions are sized from the current equity and shared cash.
        for candidate in schedules.get(day,[]):
            symbol=candidate["symbol"]
            if symbol in positions or symbol in closed_today or symbol not in today or len(positions)>=max_positions: continue
            row=today[symbol]
            entry=float(row.open)*(1+SLIPPAGE)
            atr=max(candidate["atr"],entry*.005)
            stop=entry-3.0*atr
            equity_now=cash+sum(p["shares"]*last_close.get(s,p["entry_price"]) for s,p in positions.items())
            risk_budget=equity_now*risk_fraction
            by_risk=math.floor(risk_budget/max(entry-stop,.01))
            by_weight=math.floor(equity_now*max_weight/(entry*(1+BUY_FEE)))
            by_cash=math.floor(cash/(entry*(1+BUY_FEE)))
            by_adv=math.floor(candidate["adv20"]*adv_participation/entry)
            shares=max(0,min(by_risk,by_weight,by_cash,by_adv))
            if shares<=0: continue
            entry_cost=shares*entry*(1+BUY_FEE)
            cash-=entry_cost
            positions[symbol]={
                "signal_date":candidate["signal_date"],"entry_date":day,
                "entry_price":entry,"shares":shares,"entry_cost":entry_cost,
                "stop":stop,"high_water":entry,"atr":atr,"bars_held":0,
                "pending":False,"pending_reason":"","score":candidate["score"],
                "adv20_proxy":candidate["adv20"],
            }
        # Intraday stop uses only the stop known before today's close.
        for symbol in list(positions):
            if symbol not in today: continue
            p=positions[symbol]; row=today[symbol]
            if float(row.low)<=p["stop"]:
                fill=min(float(row.open),p["stop"])*(1-SLIPPAGE)
                proceeds=p["shares"]*fill*(1-SELL_FEE_TAX)
                reason="gap_stop" if float(row.open)<=p["stop"] else "atr_stop"
                trades.append({**p,"symbol":symbol,"exit_date":day,"exit_price":fill,
                    "exit_reason":reason,"net_return":(fill/p["entry_price"])*(1-BUY_FEE)*(1-SELL_FEE_TAX)-1,
                    "pnl":proceeds-p["entry_cost"]})
                cash+=proceeds; del positions[symbol]; closed_today.add(symbol)
        # Trailing high-water stop is updated after the completed close.
        for symbol in list(positions):
            if symbol not in today: continue
            p=positions[symbol]; row=today[symbol]
            p["bars_held"]+=1
            p["high_water"]=max(p["high_water"],float(row.high),float(row.close))
            atr_now=float(row.atr) if pd.notna(row.atr) and float(row.atr)>0 else p["atr"]
            p["stop"]=max(p["stop"],p["high_water"]-trail_atr*atr_now)
            p["atr"]=atr_now
            if p["bars_held"]>=hold_bars:
                p["pending"]=True; p["pending_reason"]=f"fixed_{hold_bars}_next_open"
            last_close[symbol]=float(row.close)
        # Close any name on its last available bar within this period.
        for symbol in list(positions):
            if last_dates.get(symbol)==day and symbol in today:
                p=positions[symbol]; fill=float(today[symbol].close)*(1-SLIPPAGE)
                proceeds=p["shares"]*fill*(1-SELL_FEE_TAX)
                trades.append({**p,"symbol":symbol,"exit_date":day,"exit_price":fill,
                    "exit_reason":"period_end_close","net_return":(fill/p["entry_price"])*(1-BUY_FEE)*(1-SELL_FEE_TAX)-1,
                    "pnl":proceeds-p["entry_cost"]})
                cash+=proceeds; del positions[symbol]
        for symbol,row in today.items(): last_close[symbol]=float(row.close)
        market_value=sum(p["shares"]*last_close.get(s,p["entry_price"]) for s,p in positions.items())
        equity=cash+market_value
        curve.append({"date":day,"equity":equity,"cash":cash,"market_value":market_value,
                      "positions":len(positions),"exposure":market_value/equity if equity>0 else 0.0})
    if positions:
        for symbol,p in list(positions.items()):
            day=last_dates[symbol]; row=row_lookup[symbol][day]
            fill=float(row.close)*(1-SLIPPAGE)
            proceeds=p["shares"]*fill*(1-SELL_FEE_TAX)
            trades.append({**p,"symbol":symbol,"exit_date":day,"exit_price":fill,
                "exit_reason":"period_end_close","net_return":(fill/p["entry_price"])*(1-BUY_FEE)*(1-SELL_FEE_TAX)-1,
                "pnl":proceeds-p["entry_cost"]})
            cash+=proceeds
        if curve:
            curve[-1].update({"equity":cash,"cash":cash,"market_value":0.0,"positions":0,"exposure":0.0})
    trade_frame=pd.DataFrame(trades)
    equity_frame=pd.DataFrame(curve)
    metrics=summarize(equity_frame,trade_frame,initial_capital)
    metrics.update({"hold_bars":hold_bars,"trail_atr":trail_atr,"risk_fraction":risk_fraction,
                    "max_positions":max_positions,"max_weight":max_weight,
                    "adv_participation_proxy":adv_participation})
    return metrics,trade_frame,equity_frame


def summarize(equity: pd.DataFrame,trades: pd.DataFrame,initial: float) -> dict:
    if equity.empty:
        return {"trades":0,"CAGR":0.0,"total_return":0.0,"max_drawdown":0.0,"profit_factor":0.0}
    values=equity.equity.astype(float)
    years=max((pd.Timestamp(equity.date.iloc[-1])-pd.Timestamp(equity.date.iloc[0])).days/365.25,1/252)
    ending=float(values.iloc[-1])
    dd=values/np.maximum.accumulate(np.r_[initial,values.to_numpy()])[1:]-1
    returns=values.pct_change().fillna(values.iloc[0]/initial-1)
    pnl=pd.to_numeric(trades.get("pnl",pd.Series(dtype=float)),errors="coerce")
    wins,losses=pnl[pnl>0],pnl[pnl<0]
    pf=float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum() else (float("inf") if len(wins) else 0.0)
    net=pd.to_numeric(trades.get("net_return",pd.Series(dtype=float)),errors="coerce").dropna()
    vol=float(returns.std(ddof=1)) if len(returns)>1 else 0.0
    return {
        "trades":int(len(trades)),"win_rate":float((net>0).mean()) if len(net) else 0.0,
        "profit_factor":pf,"ending_equity":ending,"total_return":ending/initial-1,
        "CAGR":float((ending/initial)**(1/years)-1) if ending>0 else -1.0,
        "max_drawdown":float(dd.min()) if len(dd) else 0.0,
        "Sharpe":float(returns.mean()/vol*np.sqrt(252)) if vol>0 else 0.0,
        "avg_exposure":float(equity.exposure.mean()),
        "avg_hold_bars":float(trades.bars_held.mean()) if len(trades) and "bars_held" in trades else 0.0,
    }


def equal_weight_benchmark(panel: pd.DataFrame,start: str,end: str,initial: float=INITIAL_CAPITAL) -> dict:
    a,b=pd.Timestamp(start),pd.Timestamp(end)
    selected={s:g[(g.date>=a)&(g.date<=b)].sort_values("date") for s,g in panel.groupby("symbol")}
    selected={s:g for s,g in selected.items() if not g.empty}
    if not selected:return {}
    lookup={s:{pd.Timestamp(r.date):r for r in g.itertuples(index=False)} for s,g in selected.items()}
    dates=sorted(set().union(*[set(x) for x in lookup.values()]))
    last_dates={s:max(x) for s,x in lookup.items()}
    cash=initial; shares={}; last={}; allocation=initial/len(selected); values=[]
    for day in dates:
        for symbol,rows in lookup.items():
            if symbol in shares or day not in rows:continue
            price=float(rows[day].open)*(1+SLIPPAGE)
            quantity=math.floor(allocation/(price*(1+BUY_FEE)))
            cost=quantity*price*(1+BUY_FEE)
            if quantity>0 and cost<=cash:
                shares[symbol]=quantity;cash-=cost
        for symbol,rows in lookup.items():
            if day in rows:last[symbol]=float(rows[day].close)
        for symbol in list(shares):
            if last_dates[symbol]==day:
                cash+=shares.pop(symbol)*last[symbol]*(1-SLIPPAGE)*(1-SELL_FEE_TAX)
        market_value=sum(q*last.get(s,0.0) for s,q in shares.items())
        values.append((day,cash+market_value))
    for symbol,q in shares.items():
        cash+=q*last[symbol]*(1-SLIPPAGE)*(1-SELL_FEE_TAX)
    if values:values[-1]=(values[-1][0],cash)
    curve=pd.DataFrame(values,columns=["date","equity"])
    v=curve.equity.astype(float)
    years=max((pd.Timestamp(curve.date.iloc[-1])-pd.Timestamp(curve.date.iloc[0])).days/365.25,1/252)
    dd=v/np.maximum.accumulate(np.r_[initial,v.to_numpy()])[1:]-1
    return {"stocks":len(selected),"ending_equity":float(v.iloc[-1]),
            "total_return":float(v.iloc[-1]/initial-1),
            "CAGR":float((v.iloc[-1]/initial)**(1/years)-1),
            "max_drawdown":float(dd.min())}


def run_research(input_dir: str|Path,output_dir: str|Path) -> dict:
    frames,manifest=load_adjusted_frames(input_dir)
    panel=build_feature_panel(frames)
    low_model,low=fit_probability_model(panel,"low_label40")
    high_model,high=fit_probability_model(panel,"high_label40")
    panel["low_score"]=low["scores"]
    threshold=low["report"]["validation_threshold_95pct"]
    panel["signal_low_raw"]=(panel.low_score>=threshold).fillna(False)
    panel["signal_low_cooldown"]=apply_cooldown(panel,"signal_low_raw",40)
    latest=min(pd.Timestamp(panel.date.max()),pd.Timestamp("2026-11-09"))
    val_end=min(pd.Timestamp("2023-10-31"),latest)
    test_end=latest
    # Predeclared validation grid; the blind test period is never used to select.
    grid=[]
    for risk in (.02,.04,.06,.08):
        for hold in (20,30,40,50,60,80):
            for trail in (2.5,3.0,3.5,4.0,5.0):
                metrics,_,_=simulate_portfolio(
                    panel,VALIDATION_START,str(val_end.date()),hold_bars=hold,
                    trail_atr=trail,risk_fraction=risk,max_positions=5,max_weight=.40,
                    adv_participation=.01,
                )
                if metrics["trades"]>=20 and metrics["max_drawdown"]>=-.20:
                    grid.append({"metrics":metrics,"risk":risk,"hold":hold,"trail":trail})
    if not grid:
        raise RuntimeError("驗證期沒有符合最低交易數與回撤門檻的組態")
    chosen=max(grid,key=lambda x:x["metrics"]["CAGR"])
    config={"risk_fraction":chosen["risk"],"hold_bars":chosen["hold"],"trail_atr":chosen["trail"],
            "max_positions":5,"max_weight":.40,"adv_participation_proxy":.01,
            "selection":"validation only; maximum CAGR subject to >=20 trades and max drawdown >= -20%"}
    val_metrics,val_trades,val_curve=simulate_portfolio(
        panel,VALIDATION_START,str(val_end.date()),hold_bars=config["hold_bars"],
        trail_atr=config["trail_atr"],risk_fraction=config["risk_fraction"],
        max_positions=5,max_weight=.40,adv_participation=.01,
    )
    test_metrics,test_trades,test_curve=simulate_portfolio(
        panel,TEST_START,str(test_end.date()),hold_bars=config["hold_bars"],
        trail_atr=config["trail_atr"],risk_fraction=config["risk_fraction"],
        max_positions=5,max_weight=.40,adv_participation=.01,
    )
    val_benchmark=equal_weight_benchmark(panel,VALIDATION_START,str(val_end.date()))
    test_benchmark=equal_weight_benchmark(panel,TEST_START,str(test_end.date()))
    annual={}
    if not test_curve.empty:
        tmp=test_curve.copy();tmp["year"]=pd.to_datetime(tmp.date).dt.year
        for year,g in tmp.groupby("year"):
            annual[str(year)]=float(g.equity.iloc[-1]/g.equity.iloc[0]-1)
    by_symbol={}
    if not test_trades.empty:
        for symbol,g in test_trades.groupby("symbol"):
            by_symbol[symbol]={"trades":int(len(g)),"win_rate":float((g.net_return>0).mean()),
                               "pnl":float(g.pnl.sum())}
    report={
        "data_last_date":str(latest.date()),
        "data_provider_by_symbol":{x["symbol"]:x["provider"] for x in manifest.get("status_rows",[])},
        "price_basis_note":"Yahoo Finance auto_adjust=True 還原 OHLC；股利/拆併股已反映於調整價格，不另加股利。",
        "universe_bias_note":"十檔股票依 2026 年近期流動性事後選取，存在存活者及選股偏誤；不是歷史逐日可投資股票池。",
        "execution_note":"收盤訊號、下一交易日開盤進場；初始停損 3 ATR，移動停損為持倉後最高價減 5 ATR，持有 50 根 K 棒後下一開盤出場。現金帳戶、最多五檔、單檔權重上限 40%。",
        "cost_note":"買進手續費 0.1425% + 滑價 0.10%；賣出手續費 0.1425% + 證交稅 0.30% + 滑價 0.10%。",
        "adv_note":"1% 成交量參與率限制以還原收盤價 × Yahoo 回報量估算 ADV proxy，不是原始成交金額；本次實際成交量參與率需另以官方原始成交金額資料核對。",
        "label_note":"低點標籤：訊號日後 40 交易日內最高價至少上漲 15%，且期間最低價不得低於訊號收盤價 8%；高點標籤反向定義。標籤只供訓練/評估，訊號特徵只使用當日收盤可得資料。",
        "low_point_model":low["report"],
        "high_point_model_diagnostic_only":high["report"],
        "low_point_formula":low["formula"],
        "validation_selected_config":config,
        "validation_portfolio":val_metrics,
        "test_locked_portfolio":test_metrics,
        "validation_equal_weight_benchmark":val_benchmark,
        "test_equal_weight_benchmark":test_benchmark,
        "test_calendar_year_returns":annual,
        "test_trades_by_symbol":by_symbol,
        "validation_grid_count":len(grid),
    }
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"swing_turning_point_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame([{"period":"validation",**val_metrics},{"period":"test",**test_metrics}]).to_csv(out/"swing_portfolio_metrics.csv",index=False)
    pd.concat([val_trades.assign(period="validation"),test_trades.assign(period="test")],ignore_index=True).to_csv(out/"swing_trade_log.csv",index=False)
    pd.concat([val_curve.assign(period="validation"),test_curve.assign(period="test")],ignore_index=True).to_csv(out/"swing_equity_curve.csv",index=False)
    pd.DataFrame(grid).drop(columns=["metrics"]).assign(
        CAGR=[x["metrics"]["CAGR"] for x in grid],
        max_drawdown=[x["metrics"]["max_drawdown"] for x in grid],
        trades=[x["metrics"]["trades"] for x in grid],
        profit_factor=[x["metrics"]["profit_factor"] for x in grid],
    ).to_csv(out/"validation_parameter_grid.csv",index=False)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",default="data/top10_adjusted")
    parser.add_argument("--output",default="artifacts/swing_turning_point")
    args=parser.parse_args()
    print(json.dumps(run_research(args.input,args.output),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
