"""Minervini-style trend-template + volatility-contraction breakout candidate.

This is a transparent OHLCV proxy, not a claim to reproduce proprietary SEPA/VCP.
All rolling highs/lows and volume baselines are shifted to avoid lookahead.
"""
from __future__ import annotations
from typing import Mapping
import numpy as np
import pandas as pd
from corporate_actions import point_in_time_recalculate


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    z=frame.sort_values("date").reset_index(drop=True).copy()
    c=pd.to_numeric(z["close"],errors="coerce")
    h=pd.to_numeric(z["high"],errors="coerce")
    l=pd.to_numeric(z["low"],errors="coerce")
    v=pd.to_numeric(z["volume"],errors="coerce").astype(float)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr14=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    atr50=tr.rolling(50,min_periods=40).mean()
    z["sma50"]=c.rolling(50,min_periods=50).mean()
    z["sma150"]=c.rolling(150,min_periods=150).mean()
    z["sma200"]=c.rolling(200,min_periods=200).mean()
    z["sma50_lag20"]=z["sma50"].shift(20)
    z["sma200_lag20"]=z["sma200"].shift(20)
    z["high252"]=h.shift(1).rolling(252,min_periods=200).max()
    z["low252"]=l.shift(1).rolling(252,min_periods=200).min()
    z["high20"]=h.shift(1).rolling(20,min_periods=20).max()
    z["low10"]=l.shift(1).rolling(10,min_periods=10).min()
    z["volavg50"]=v.shift(1).rolling(50,min_periods=40).mean()
    z["mom126"]=c.pct_change(126)
    z["mom63"]=c.pct_change(63)
    z["atr_ratio"]=atr14/atr50.replace(0,np.nan)
    z["ema21"]=c.ewm(span=21,adjust=False,min_periods=21).mean()
    z["volume"]=v
    return z[["date","close","volume","sma50","sma150","sma200","sma50_lag20",
              "sma200_lag20","high252","low252","high20","low10","volavg50",
              "mom126","mom63","atr_ratio","ema21"]]


def build_minervini_vcp_overrides(frames: Mapping[str,pd.DataFrame], *,
        corporate_actions=None, relative_strength_quantile: float=0.90,
        minimum_universe: int=15, volume_multiple: float=1.50,
        contraction_ratio: float=0.85) -> dict[str,tuple[pd.Series,pd.Series]]:
    if not 0.5 < relative_strength_quantile < 1.0:
        raise ValueError("relative_strength_quantile 必須介於 0.5 與 1")
    rows=[]
    for code,frame in frames.items():
        if frame is None or frame.empty: continue
        events=(corporate_actions or {}).get(str(code))
        part=point_in_time_recalculate(frame,events,_features).copy()
        part["symbol"]=str(code)
        rows.append(part)
    if not rows: return {}
    panel=pd.concat(rows,ignore_index=True)
    panel["date"]=pd.to_datetime(panel["date"])
    panel=panel.sort_values(["date","symbol"]).reset_index(drop=True)
    panel["rs_rank"]=panel.groupby("date")["mom126"].rank(pct=True)
    panel["universe_count"]=panel.groupby("date")["symbol"].transform("nunique")
    out={}
    for code,p in panel.groupby("symbol",sort=False):
        p=p.sort_values("date")
        template=(
            (p["close"]>p["sma50"]) & (p["sma50"]>p["sma150"])
            & (p["sma150"]>p["sma200"])
            & (p["sma50"]>p["sma50_lag20"])
            & (p["sma200"]>p["sma200_lag20"])
            & (p["close"]>=1.30*p["low252"])
            & (p["close"]>=0.80*p["high252"])
            & (p["mom63"]>0) & (p["mom126"]>0)
        )
        contraction=p["atr_ratio"]<=float(contraction_ratio)
        breakout=(p["close"]>p["high20"]) & (
            p["volume"]>=float(volume_multiple)*p["volavg50"])
        valid=p["universe_count"]>=int(minimum_universe)
        entry=(valid & template & contraction
               & (p["rs_rank"]>=float(relative_strength_quantile))
               & breakout).fillna(False)
        exit_=(~valid | (p["close"]<p["ema21"])
               | (p["close"]<p["low10"]) | (p["mom63"]<0)).fillna(False)
        out[str(code)]=(pd.Series(entry.to_numpy(dtype=bool)),
                        pd.Series(exit_.to_numpy(dtype=bool)))
    return out
