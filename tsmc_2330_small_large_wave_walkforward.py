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


def true_range_atr(d, period=14):
    """Wilder-style rolling ATR using only OHLC known by the current close."""
    high = pd.to_numeric(d["raw_high"], errors="coerce")
    low = pd.to_numeric(d["raw_low"], errors="coerce")
    close = pd.to_numeric(d["raw_close"], errors="coerce")
    previous = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-previous).abs(),
                    (low-previous).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def _new_low_reentry_signal(d, low_signal, high_signal, i, last_exit_i, wait_sessions):
    """Require a fresh low signal after the mandatory cooldown; old alerts cannot re-enter."""
    # Entry executes at open i from the signal known at close prev_i.
    # Wait N *complete* sessions after stop: eligible signal must be dated
    # at the close of the Nth full session after exit; fill is the next open.
    signal_i = i - 1
    eligible_signal_i = last_exit_i + int(wait_sessions)
    return (signal_i >= eligible_signal_i and bool(low_signal[signal_i])
            and not bool(high_signal[signal_i]))


def simulate_atr_reentry(d, low_signal, high_signal, start_date,
                         stop_multiple=2.0, wait_sessions=1,
                         capital=INITIAL_CAPITAL, atr_period=14):
    """All-in/all-out ATR stop experiment using causal signals and next-open execution.

    Initial stop is entry fill minus k * prior-close ATR. Trailing stop uses the
    highest completed close since entry minus k * that day's ATR, and can only
    ratchet upward. A stop hit intraday fills at the stop; an opening gap below
    it fills at the open. Following a stop, the previous LOW alert is discarded:
    re-entry requires a fresh LOW alert formed after the cooldown.
    """
    if stop_multiple not in (1.5, 2.0, 2.5, 3.0):
        raise ValueError("stop_multiple must be one of 1.5, 2.0, 2.5, 3.0")
    if wait_sessions not in (1, 3, 5, 10):
        raise ValueError("wait_sessions must be one of 1, 3, 5, 10")
    required = {"date", "raw_open", "raw_high", "raw_low", "raw_close"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Missing execution columns: {sorted(missing)}")
    if len(low_signal) != len(d) or len(high_signal) != len(d):
        raise ValueError("Signal lengths must match bars")
    bars = d.reset_index(drop=True).copy()
    atr = true_range_atr(bars, atr_period).to_numpy(dtype=float)
    lows = np.asarray(low_signal, dtype=bool)
    highs = np.asarray(high_signal, dtype=bool)
    action = (bars.corporate_action_window.fillna(True).astype(bool).to_numpy()
              if "corporate_action_window" in bars else np.zeros(len(bars), dtype=bool))
    active = np.flatnonzero((bars.date >= pd.Timestamp(start_date)).to_numpy())
    if len(active) < 2:
        raise ValueError("Not enough rows for ATR stop simulation")

    cash, shares, entry_cost = float(capital), 0, 0.0
    entry_date = None
    entry_price = np.nan
    initial_stop = np.nan
    stop_price = np.nan
    peak_close = np.nan
    peak_high_since_entry = np.nan
    entry_i = None
    last_exit_i = -10**9
    stopped_out = False
    reentries = 0
    trade_id = 0
    trades, equity = [], []
    conflicts = 0
    previous_stop = np.nan

    def close_position(i, fill_reference, reason, stop_distance=np.nan):
        nonlocal cash, shares, entry_cost, entry_date, entry_price
        nonlocal initial_stop, stop_price, peak_close, peak_high_since_entry, entry_i
        nonlocal stopped_out, last_exit_i, previous_stop
        row = bars.iloc[i]
        fill = float(fill_reference) * (1 - SLIPPAGE)
        notional = shares * fill
        fees = commission(notional)
        tax = int(math.floor(notional * SELL_TAX_RATE))
        proceeds = notional - fees - tax
        pnl = proceeds - entry_cost
        trades.append({
            "trade_id": trade_id, "entry_date": entry_date,
            "exit_date": row.date.date().isoformat(),
            "entry_index": int(entry_i), "exit_index": int(i),
            "shares": int(shares), "entry_fill": float(entry_price),
            "exit_fill": float(fill), "entry_cost": float(entry_cost),
            "exit_proceeds": float(proceeds), "net_pnl": float(pnl),
            "net_return": float(pnl / entry_cost) if entry_cost else np.nan,
            "exit_reason": reason, "stop_multiple": float(stop_multiple),
            "stop_width_at_entry": float(entry_price - initial_stop),
            "stop_profit_lock_at_exit": float(stop_distance) if np.isfinite(stop_distance) else np.nan,
            "max_favorable_excursion_pct": float(max(0.0, peak_high_since_entry / entry_price - 1)),
            "max_adverse_excursion_pct": float(min(0.0, min_low_since_entry / entry_price - 1)),
            "bars_held": int(i - entry_i),
            "reentry_count_before_trade": int(reentries),
        })
        cash += proceeds
        last_exit_i = i
        stopped_out = reason in ("atr_stop", "gap_stop", "entry_day_atr_stop", "entry_day_gap_stop")
        previous_stop = stop_price
        shares, entry_cost = 0, 0.0
        entry_date = entry_price = initial_stop = stop_price = peak_close = peak_high_since_entry = np.nan
        entry_i = None

    min_low_since_entry = np.nan
    for k, i in enumerate(active):
        row = bars.iloc[i]
        op, hi, lo, close = (float(row[c]) for c in
                             ("raw_open", "raw_high", "raw_low", "raw_close"))
        exited_today = False

        if k > 0 and shares:
            prev_i = active[k-1]
            # Previous close's HIGH alert is known before today's open.
            if highs[prev_i] and not lows[prev_i]:
                close_position(i, op, "high_alert_next_open", stop_price - entry_price)
                exited_today = True

        if shares and not exited_today:
            # Stop level carried from the prior close is active at today's open.
            if op <= stop_price:
                close_position(i, op, "gap_stop", stop_price - entry_price)
                exited_today = True
            elif lo <= stop_price:
                close_position(i, stop_price, "atr_stop", stop_price - entry_price)
                exited_today = True

        # A fresh low signal known at the previous close submits a next-open buy.
        # Stops are evaluated before any re-entry on the same day.
        if not shares and not exited_today and k > 0:
            prev_i = active[k-1]
            if not action[i] and lows[prev_i] and not highs[prev_i]:
                # If the last exit was a stop, old signals are invalidated and
                # this signal must itself be dated after the cooldown.
                if stopped_out:
                    can_reenter = _new_low_reentry_signal(
                        bars, lows, highs, i, last_exit_i, wait_sessions
                    )
                else:
                    can_reenter = True
                if can_reenter and np.isfinite(atr[prev_i]) and atr[prev_i] > 0 and op > 0:
                    fill = op * (1 + SLIPPAGE)
                    qty = int(cash // fill) if fill > 0 else 0
                    while qty:
                        gross = qty * fill
                        cost = gross + commission(gross)
                        if cost <= cash + 1e-8:
                            break
                        qty -= 1
                    if qty:
                        trade_id += 1
                        shares = qty
                        entry_price = fill
                        entry_cost = qty * fill + commission(qty * fill)
                        cash -= entry_cost
                        entry_date = row.date.date().isoformat()
                        entry_i = i
                        initial_stop = entry_price - stop_multiple * atr[prev_i]
                        stop_price = initial_stop
                        peak_close = close
                        min_low_since_entry = lo
                        # Intraday OHLC does not identify whether the day's low
                        # occurred before or after the opening fill. Activate the
                        # initial stop from the next session to avoid path leakage.
                        stopped_out = False

        if shares:
            peak_close = max(float(peak_close), close)
            # Exclude the entry session's full low from MAE: OHLC does not tell
            # whether that low occurred before or after the next-open entry.
            if i > int(entry_i):
                min_low_since_entry = min(float(min_low_since_entry), lo)
            # Update stop using today's completed close only. It becomes active
            # from the next session, avoiding look-ahead within today's range.
            candidate = peak_close - stop_multiple * atr[i] if np.isfinite(atr[i]) else np.nan
            if np.isfinite(candidate):
                stop_price = max(float(stop_price), float(candidate))
            equity_value = cash + shares * close
            equity.append({
                "date": row.date.date().isoformat(), "equity": equity_value,
                "cash": cash, "shares": shares, "close": close,
                "active_stop": stop_price, "peak_close": peak_close,
                "stop_multiple": stop_multiple, "wait_sessions": wait_sessions,
            })
        else:
            equity.append({
                "date": row.date.date().isoformat(), "equity": cash,
                "cash": cash, "shares": 0, "close": close,
                "active_stop": np.nan, "peak_close": np.nan,
                "stop_multiple": stop_multiple, "wait_sessions": wait_sessions,
            })

    if shares:
        last_i = int(active[-1])
        close_position(last_i, float(bars.iloc[last_i].raw_close),
                       "end_of_test_close", stop_price - entry_price)
        equity[-1].update({"equity": cash, "cash": cash, "shares": 0,
                           "active_stop": np.nan})

    curve = pd.DataFrame(equity)
    curve["peak"] = curve.equity.cummax()
    curve["drawdown"] = curve.equity / curve.peak - 1
    tr = pd.DataFrame(trades)
    elapsed = max((pd.Timestamp(curve.date.iloc[-1]) -
                   pd.Timestamp(curve.date.iloc[0])).days / 365.25, 1/365.25)
    final = float(curve.equity.iloc[-1])
    metrics = {
        "initial_capital": float(capital), "final_equity": final,
        "total_return": final / capital - 1,
        "cagr": (final / capital) ** (1 / elapsed) - 1 if final > 0 else -1,
        "max_drawdown": float(curve.drawdown.min()),
        "trades": int(len(tr)), "win_rate": float((tr.net_pnl > 0).mean()) if len(tr) else np.nan,
        "exposure": float((curve.shares > 0).mean()),
        "average_mfe_pct": float(tr.max_favorable_excursion_pct.mean()) if len(tr) else np.nan,
        "average_mae_pct": float(tr.max_adverse_excursion_pct.mean()) if len(tr) else np.nan,
        "stops": int(tr.exit_reason.isin(["atr_stop", "gap_stop", "entry_day_atr_stop", "entry_day_gap_stop"]).sum()) if len(tr) else 0,
        "reentries": int(reentries),
        "same_day_conflicts_ignored": int(conflicts),
        "stop_multiple": float(stop_multiple), "wait_sessions": int(wait_sessions),
    }
    return metrics, tr, curve


def summarize_post_stop_behavior(d, trades, horizon=10):
    """Causal descriptive post-stop price behaviour; no future field feeds a signal."""
    if trades.empty:
        return pd.DataFrame()
    rows = []
    dates = pd.to_datetime(d.date).reset_index(drop=True)
    close = pd.to_numeric(d.raw_close, errors="coerce").to_numpy(float)
    low = pd.to_numeric(d.raw_low, errors="coerce").to_numpy(float)
    high = pd.to_numeric(d.raw_high, errors="coerce").to_numpy(float)
    for t in trades.itertuples(index=False):
        if t.exit_reason not in ("atr_stop", "gap_stop", "entry_day_atr_stop", "entry_day_gap_stop"):
            continue
        i = int(t.exit_index)
        base = float(t.exit_fill)
        for day in (1, 3, 5, 10, 20):
            j = i + day
            if j >= len(d):
                continue
            future = slice(i + 1, j + 1)
            path = close[future]
            rows.append({
                "trade_id": int(t.trade_id), "stop_date": dates.iloc[i].date().isoformat(),
                "sessions_after_stop": day,
                "stop_price": base,
                "forward_close_return": float(close[j] / base - 1),
                "max_rebound_pct": float(np.nanmax(high[future]) / base - 1),
                "max_drawdown_after_stop_pct": float(np.nanmin(low[future]) / base - 1),
                "recovered_above_stop": bool(np.nanmax(close[future]) > base),
            })
    return pd.DataFrame(rows)


def compare_atr_stop_grid(d, low_signal, high_signal, start_date,
                          base_metrics=None, base_trades=None, base_curve=None):
    rows, trade_frames, curves, post_stop = [], [], [], []
    for multiple in (1.5, 2.0, 2.5, 3.0):
        for wait in (1, 3, 5, 10):
            metrics, trades, curve = simulate_atr_reentry(
                d, low_signal, high_signal, start_date, multiple, wait
            )
            metrics.update({"model": "technical", "stop_multiple": multiple,
                            "wait_sessions": wait, "baseline": False})
            rows.append(metrics)
            if not trades.empty:
                x = trades.copy()
                x["stop_multiple"], x["wait_sessions"] = multiple, wait
                trade_frames.append(x)
                post_stop.append(summarize_post_stop_behavior(d, trades))
            y = curve.copy()
            y["model"] = "technical"
            curves.append(y)

    if base_metrics is not None:
        x = dict(base_metrics)
        x.update({"model": "no_stop_baseline", "stop_multiple": np.nan,
                  "wait_sessions": 0, "baseline": True})
        rows.append(x)
        if base_trades is not None and not base_trades.empty:
            y = base_trades.copy()
            y["model"] = "no_stop_baseline"
            trade_frames.append(y)
        if base_curve is not None and not base_curve.empty:
            y = base_curve.copy()
            y["model"] = "no_stop_baseline"
            curves.append(y)
    return (pd.DataFrame(rows),
            pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(),
            pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
            pd.concat([x for x in post_stop if not x.empty], ignore_index=True) if any(not x.empty for x in post_stop) else pd.DataFrame())



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
    bt_rows=[];trade_parts=[];curve_parts=[];stop_summary_parts=[];stop_trade_parts=[];stop_curve_parts=[];post_stop_parts=[]
    for model_name in FEATURE_SETS:
      for horizon in LEADS:
        z=pred[(pred.model==model_name)&(pred.horizon==horizon)]
        pivot={}
        for side in ("HIGH","LOW"):
          for wave in ("small","large"):
            part=z[(z.pivot_type==side)&(z.wave_class==wave)].set_index("date")
            pivot[(side,wave)]=part.signal.reindex(bars.date).eq(True).to_numpy()
        action_window=d.corporate_action_window.fillna(True).astype(bool).to_numpy() if "corporate_action_window" in d else np.zeros(n,dtype=bool)
        for policy in ("small_only","large_only","all_waves"):
          if policy=="small_only":lo=pivot[("LOW","small")].copy();hi=pivot[("HIGH","small")].copy()
          elif policy=="large_only":lo=pivot[("LOW","large")].copy();hi=pivot[("HIGH","large")].copy()
          else:
            lo=pivot[("LOW","small")]|pivot[("LOW","large")]
            hi=pivot[("HIGH","small")]|pivot[("HIGH","large")]
          lo &= ~action_window
          hi &= ~action_window
          m,tr,curve=simulate_all_in_all_out(bars,lo,hi,f"{START_TEST_YEAR}-01-01")
          m.update({"model":model_name,"horizon":horizon,"wave_policy":policy})
          bt_rows.append(m)
          if len(tr):tr.insert(0,"wave_policy",policy);tr.insert(0,"horizon",horizon);tr.insert(0,"model",model_name);trade_parts.append(tr)
          curve.insert(0,"wave_policy",policy);curve.insert(0,"horizon",horizon);curve.insert(0,"model",model_name);curve_parts.append(curve)
          # Stop grid is applied to the same frozen OOF LOW/HIGH signals as the
          # no-stop comparator. ATR is calculated from raw executable prices.
          if model_name == "technical":
            grid, stop_trades, stop_curves, post = compare_atr_stop_grid(
                bars, lo, hi, f"{START_TEST_YEAR}-01-01", m, tr, curve
            )
            grid["horizon"] = horizon
            grid["wave_policy"] = policy
            stop_summary_parts.append(grid)
            if not stop_trades.empty:
                stop_trades["horizon"], stop_trades["wave_policy"] = horizon, policy
                stop_trade_parts.append(stop_trades)
            if not stop_curves.empty:
                stop_curves["horizon"], stop_curves["wave_policy"] = horizon, policy
                stop_curve_parts.append(stop_curves)
            if not post.empty:
                post["horizon"], post["wave_policy"] = horizon, policy
                post_stop_parts.append(post)
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
    if stop_summary_parts:
        pd.concat(stop_summary_parts, ignore_index=True).to_csv(
            out/"tsmc_2330_atr_stop_reentry_summary.csv", index=False)
    if stop_trade_parts:
        pd.concat(stop_trade_parts, ignore_index=True).to_csv(
            out/"tsmc_2330_atr_stop_reentry_trades.csv", index=False)
    if stop_curve_parts:
        pd.concat(stop_curve_parts, ignore_index=True).to_csv(
            out/"tsmc_2330_atr_stop_reentry_equity.csv", index=False)
    if post_stop_parts:
        pd.concat(post_stop_parts, ignore_index=True).to_csv(
            out/"tsmc_2330_post_stop_behaviour.csv", index=False)
    meta={"symbol":"2330","threshold_id":THRESHOLD_ID,"rows":n,
      "data_start":dates.min().date().isoformat(),"data_end":dates.max().date().isoformat(),
      "event_rows":len(e),"wave_size_measure":"absolute reversal_pct from pivot event file",
      "classification":"per fold and per side: small <= 35th percentile and large >= 65th percentile of pivot amplitudes observed before validation start; middle 30% omitted from class-specific positives",
      "walk_forward":"expanding train; preceding 2 years calibrate score cutoff; next year test; horizon-purged training labels",
      "alert_quantile":ALERT_QUANTILE,"feature_sets":{k:list(v) for k,v in FEATURE_SETS.items()},
      "institutional":"foreign investors and investment trusts only; eligible from 2012-05-02; dealer excluded",
      "execution":"all available cash, integer shares, next-session open, commission 0.1425% (NT$20 min), sell tax 0.3%, slippage 0.1% each side; baseline has no stop; ATR stop grid uses 1.5/2.0/2.5/3.0 ATR and post-stop fresh-low re-entry waits 1/3/5/10 sessions; forced liquidation at final close",
      "limitations":["Single-stock exploratory analysis; prior full-sample feature research creates selection bias.",
       "Pivot amplitude is retrospective and available only after reversal confirmation.",
       "No cash dividends, corporate-action cashflows, official exchange price-limit/fill simulation, liquidity impact or odd-lot microstructure.",
       "2026 is partial. No independent forward validation or 200% CAGR claim."],
      "stop_grid":{"multiples":[1.5,2.0,2.5,3.0],"reentry_wait_sessions":[1,3,5,10],"atr_period":14,"trailing_rule":"highest completed close since entry minus k times current completed-session ATR; stop ratchets upward only; execution begins next session","reentry_rule":"prior LOW alert is invalidated on stop; a new LOW signal after cooldown is required","outputs":["tsmc_2330_atr_stop_reentry_summary.csv","tsmc_2330_atr_stop_reentry_trades.csv","tsmc_2330_atr_stop_reentry_equity.csv","tsmc_2330_post_stop_behaviour.csv"]},
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
