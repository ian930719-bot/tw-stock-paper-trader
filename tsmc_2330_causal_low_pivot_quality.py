"""Causal low-pivot quality event study for TSMC 2330.

This is an event-level diagnostic, NOT a trading strategy or P&L result.
Low pivot extreme dates are retained only as retrospective descriptors. Every
feature used for prediction is sampled on/before confirmation close; execution
reference is the next session open. Forward MFE/MAE are labels only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from tsmc_2330_4019_replication import inputs
from tsmc_2330_causal_zigzag_wave_oof import ATR_MULTIPLIERS, causal_atr_zigzag

HORIZONS = (20, 40, 60, 120)
MULTIPLIERS = ATR_MULTIPLIERS

def safe_num(row, key):
    v = row.get(key, np.nan)
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def build_events(d):
    d = d.sort_values("date").reset_index(drop=True)
    records = []
    close = d.raw_close.to_numpy(float)
    high = d.raw_high.to_numpy(float)
    low = d.raw_low.to_numpy(float)
    op = d.raw_open.to_numpy(float)
    volume_key = next((k for k in ("raw_volume", "volume", "trading_volume") if k in d.columns), None)
    feature_cols = [
        "return_1", "return_5", "return_20", "atr_pct14", "rsi14",
        "macd_hist_change_3", "distance_sma_20", "distance_ema_20",
        "distance_sma_50", "distance_sma_200", "relative_volume20",
        "close_location", "intraday_range_pct", "obv_flow_ratio_5",
        "plus_di14", "minus_di14", "mfi14", "stoch_k14",
    ]
    # Rolling volume context is causal and anchored at the pivot date.
    if volume_key:
        vol = pd.to_numeric(d[volume_key], errors="coerce")
        d["_volume"] = vol
        d["_vol_med20"] = vol.rolling(20, min_periods=10).median()
        d["_vol_med60"] = vol.rolling(60, min_periods=20).median()
    dates = pd.to_datetime(d.date).dt.strftime("%Y-%m-%d").to_numpy()
    for mult in MULTIPLIERS:
        low_signal, _, events = causal_atr_zigzag(d, mult)
        for e in events:
            if e["pivot_type"] != "LOW":
                continue
            p, c = int(e["pivot_index"]), int(e["confirmation_index"])
            if c + 1 >= len(d):
                continue
            pivot, confirm = d.iloc[p], d.iloc[c]
            start_price = op[c + 1]
            if not np.isfinite(start_price) or start_price <= 0:
                continue
            rec = {
                "atr_multiplier": float(mult),
                "pivot_date_diagnostic_only": dates[p],
                "confirmation_date": dates[c],
                "confirmation_lag_sessions": c-p,
                "pivot_price_diagnostic_only": float(e["pivot_price"]),
                "pivot_atr_pct": safe_num(pivot, "atr_pct14"),
                "reversal_at_confirmation_pct": float(e["reversal_at_confirmation"]),
                "next_session_open_reference": float(start_price),
                "corporate_action_window_at_confirmation": bool(confirm.get("corporate_action_window", True)),
            }
            for k in feature_cols:
                rec["pivot_"+k] = safe_num(pivot, k)
                rec["confirm_"+k] = safe_num(confirm, k)
            if volume_key:
                rec["pivot_volume_to_med20"] = (
                    float(pivot["_volume"] / pivot["_vol_med20"])
                    if pd.notna(pivot["_vol_med20"]) and pivot["_vol_med20"] else None)
                rec["pivot_volume_to_med60"] = (
                    float(pivot["_volume"] / pivot["_vol_med60"])
                    if pd.notna(pivot["_vol_med60"]) and pivot["_vol_med60"] else None)
            # Pre-pivot drawdown and volatility trajectory: no future observations.
            for window in (5, 10, 20, 40, 60):
                a = max(0, p-window+1)
                segment = close[a:p+1]
                rec[f"pre_pivot_{window}d_return"] = float(close[p]/close[a]-1) if len(segment)>1 and close[a]>0 else None
                rec[f"pre_pivot_{window}d_range_pct"] = float((np.max(high[a:p+1])-np.min(low[a:p+1]))/close[p]) if len(segment)>1 and close[p]>0 else None
            # Forward labels begin at next-session open after confirmation.
            for h in HORIZONS:
                end = min(len(d), c+1+h)
                if end <= c+1:
                    rec[f"mfe_{h}d"] = None
                    rec[f"mae_{h}d"] = None
                    rec[f"close_return_{h}d"] = None
                    continue
                future_high = high[c+1:end]
                future_low = low[c+1:end]
                rec[f"mfe_{h}d"] = float(np.max(future_high)/start_price-1)
                rec[f"mae_{h}d"] = float(np.min(future_low)/start_price-1)
                rec[f"close_return_{h}d"] = float(close[end-1]/start_price-1)
                rec[f"observed_sessions_{h}d"] = int(end-(c+1))
            records.append(rec)
    return pd.DataFrame(records)

def summarize(events):
    out = {}
    if events.empty:
        return out
    for (mult, action), g in events.groupby(["atr_multiplier","corporate_action_window_at_confirmation"], dropna=False):
        key = f"atr_{mult:.1f}_action_{int(bool(action))}"
        out[key] = {"events": int(len(g)), "median_confirmation_lag": float(g.confirmation_lag_sessions.median()),
                    "median_reversal_pct": float(g.reversal_at_confirmation_pct.median())}
        for h in HORIZONS:
            for metric in ("mfe","mae","close_return"):
                s = pd.to_numeric(g[f"{metric}_{h}d"], errors="coerce").dropna()
                out[key][f"{metric}_{h}d"] = {
                    "n": int(len(s)), "median": float(s.median()) if len(s) else None,
                    "mean": float(s.mean()) if len(s) else None,
                    "positive_fraction": float((s>0).mean()) if len(s) else None,
                }
    # Rank-based descriptive contrasts, with outcome labels kept separate from predictors.
    for h in HORIZONS:
        col=f"mfe_{h}d"
        valid=events.dropna(subset=[col]).copy()
        if len(valid) >= 10:
            threshold=float(valid[col].quantile(.75))
            valid["upper_quartile_mfe"] = valid[col] >= threshold
            out[f"descriptive_upper_quartile_mfe_{h}d"] = {
                "threshold": threshold, "n": int(len(valid)),
                "upper_quartile_count": int(valid.upper_quartile_mfe.sum()),
                "median_confirmation_lag_upper": float(valid.loc[valid.upper_quartile_mfe,"confirmation_lag_sessions"].median()),
                "median_confirmation_lag_rest": float(valid.loc[~valid.upper_quartile_mfe,"confirmation_lag_sessions"].median()),
                "warning": "Retrospective outcome grouping only; not a predictive classifier or OOF strategy."
            }
    return out

def run(features, limits, dividends, output):
    d, _, _ = inputs(features, limits, dividends)
    ev = build_events(d)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    ev.to_csv(out/"causal_low_pivot_quality_events.csv", index=False)
    summary = {
        "symbol":"2330",
        "classification":"CAUSAL_EVENT_STUDY_DESCRIPTIVE_NOT_STRATEGY",
        "data_period":[d.date.min().date().isoformat(), d.date.max().date().isoformat()],
        "event_count":int(len(ev)),
        "atr_multipliers":list(MULTIPLIERS),
        "horizons_sessions":list(HORIZONS),
        "feature_timing":"Pivot descriptors sampled at retrospective pivot date; predictive candidates sampled no later than confirmation close; execution reference next session open.",
        "label_timing":"MFE/MAE/close returns use only post-confirmation observations and are labels, never predictors.",
        "group_summaries":summarize(ev),
        "limitations":[
            "Historical 2010-2026 data has been examined in prior research; no pristine blind OOS claim.",
            "Events at different ATR thresholds overlap and are not independent observations.",
            "Daily OHLC cannot identify intraday path or queue priority.",
            "Last 120 sessions have incomplete forward labels and are excluded per horizon.",
            "This artifact contains event statistics, not portfolio returns or a trading strategy."
        ]
    }
    (out/"tsmc_2330_causal_low_pivot_quality.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--features",required=True); p.add_argument("--limits",required=True)
    p.add_argument("--dividends",required=True); p.add_argument("--output",default="artifacts/tsmc_2330_causal_low_pivot_quality")
    a=p.parse_args()
    print(json.dumps(run(a.features,a.limits,a.dividends,a.output),ensure_ascii=False,indent=2,default=float))
