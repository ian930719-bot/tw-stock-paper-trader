"""Exact frozen TSMC high-hold strategy cross-symbol audit.

Requires official raw TWSE bars, verified per-symbol corporate actions, and
reconstructed exchange price limits. It fails closed if any prerequisite is
missing. Signals are calculated from back-adjusted OHLCV; fills use untouched
raw TWSE OHLC. This is a fixed-rule transfer test, not per-symbol retuning.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from corporate_action_adjustments import apply_back_adjustments
from tsmc_swing_research import build_causal_features
from tsmc_2330_4019_replication import RULES,build_candidate_signals,simulate,annual_metrics,INITIAL_CAPITAL,SLIPPAGE_BPS
from tsmc_2330_trend_reentry_annual_oof import tick_size

SYMBOLS=["2330","2454","1303","2327","2408","2308","3037","3008","2383","2303"]
NAMES={"2330":"台積電","2454":"聯發科","1303":"南亞","2327":"國巨","2408":"南亞科","2308":"台達電","3037":"欣興","3008":"大立光","2383":"台光電","2303":"聯電"}

def limit_price(ref,band,side):
 if not np.isfinite(ref) or ref<=0:raise ValueError("invalid reference price")
 raw=ref*(1+band if side=="UP" else 1-band)
 rounding=math.floor if side=="UP" else math.ceil
 candidate=raw
 for _ in range(12):
  tick=tick_size(candidate)
  units=rounding((raw-1e-12)/tick if side=="UP" else (raw+1e-12)/tick)
  rounded=round(units*tick,2)
  if abs(tick_size(rounded)-tick)<1e-9:return rounded
  candidate=rounded
 raise RuntimeError(f"tick rounding did not converge: {ref=} {band=} {side=}")

def read_events(action_root,symbol):
 root=Path(action_root)/"corporate_actions"
 manifest=root/"manifest.json"
 if not manifest.exists():raise RuntimeError("Missing corporate_actions/manifest.json")
 m=json.loads(manifest.read_text(encoding="utf-8"))
 if not m.get("corporate_actions_complete") or m.get("event_conflicts",0):
  raise RuntimeError(f"Corporate action data incomplete/conflicting: {m}")
 p=root/"by_symbol"/f"{symbol}.csv"
 if not p.exists():raise RuntimeError(f"Missing corporate-action file for {symbol}")
 div=pd.read_csv(p,dtype={"stock_id":str})
 if div.empty:return pd.DataFrame(),{},{}
 div["date"]=pd.to_datetime(div.date,errors="coerce").dt.normalize()
 refmap={}
 for r in div.itertuples(index=False):
  ref=getattr(r,"reference_price",np.nan)
  if pd.notna(ref) and float(ref)>0:
   key=pd.Timestamp(r.date).strftime("%Y-%m-%d")
   refmap.setdefault(key,[]).append(float(ref))
  if str(getattr(r,"stock_or_cache_dividend","")).strip()=="息":
   cash=getattr(r,"stock_and_cache_dividend",np.nan)
   if pd.notna(cash) and float(cash)>0:
    refmap.setdefault("__cash__"+pd.Timestamp(r.date).strftime("%Y-%m-%d"),[]).append(float(cash))
 # Read global action datasets and add their official post-event reference prices.
 globals_=[
  ("TaiwanStockSplitPrice","after_price"),
  ("TaiwanStockCapitalReductionReferencePrice","PostReductionReferencePrice"),
  ("TaiwanStockParValueChange","after_ref_close"),
 ]
 extra=[]
 for filename,col in globals_:
  f=root/f"{filename}.csv"
  if not f.exists():raise RuntimeError(f"Missing global corporate-action source: {filename}")
  g=pd.read_csv(f,dtype={"stock_id":str})
  if g.empty:continue
  g=g[g.stock_id.astype(str).str.strip()==symbol]
  for r in g.itertuples(index=False):
   v=getattr(r,col,np.nan)
   if pd.notna(v) and float(v)>0:
    key=pd.Timestamp(r.date).strftime("%Y-%m-%d")
    refmap.setdefault(key,[]).append(float(v))
    extra.append({"date":key,"source":filename,"reference_price":float(v)})
 # Resolve same-date source conflicts strictly.
 refs={}
 for k,vals in refmap.items():
  unique=sorted(set(round(float(v),6) for v in vals))
  if len(unique)>1:raise RuntimeError(f"{symbol} conflicting reference/cash event {k}: {unique}")
  refs[k]=unique[0]
 events=pd.read_csv(root/"events.csv",dtype={"symbol":str})
 events=events[events.symbol.astype(str).str.strip()==symbol].copy() if not events.empty else pd.DataFrame()
 if not events.empty and events.conflict.fillna(False).any():raise RuntimeError(f"{symbol} corporate action event conflict")
 cash={k.replace("__cash__",""):v for k,v in refs.items() if k.startswith("__cash__")}
 ref_prices={k:v for k,v in refs.items() if not k.startswith("__cash__")}
 return events,cash,ref_prices

def build_limits(bars,ref_prices,symbol):
 out={};prev_close=None;missing=[];conflicts=[]
 for r in bars.itertuples(index=False):
  day=pd.Timestamp(r.date).strftime("%Y-%m-%d")
  if day in ref_prices:ref=ref_prices[day];src="verified_corporate_action_reference"
  elif prev_close is not None:ref=prev_close;src="prior_traded_close"
  else:
   if day >= "2015-01-01": missing.append(day)
   continue
  band=.07 if day<"2015-06-01" else .10
  up=limit_price(ref,band,"UP");down=limit_price(ref,band,"DOWN")
  hi=float(r.high);lo=float(r.low)
  # Any observed traded price outside reconstructed official bounds means
  # an unmodeled event or special rule; do not backtest through it.
  if hi>up+1e-6 or lo<down-1e-6:
   conflicts.append({"date":day,"high":hi,"low":lo,"up":up,"down":down,"reference":ref,"source":src})
  out[day]=(up,down)
  if float(r.volume)>0:prev_close=float(r.close)
  else:prev_close=None
 if missing:raise RuntimeError(f"{symbol}: missing price-limit reference on {len(missing)} sessions; first={missing[:5]}")
 if conflicts:raise RuntimeError(f"{symbol}: {len(conflicts)} OHLC/official-rule conflicts; first={conflicts[:5]}")
 return out

def make_features(raw,events):
 adj=apply_back_adjustments(raw,events)
 f=adj.copy()
 for col in ("open","high","low","close"):f[col]=f["adj_"+col]
 f=build_causal_features(f)
 f["raw_open"]=raw.open.to_numpy(float)
 f["raw_high"]=raw.high.to_numpy(float)
 f["raw_low"]=raw.low.to_numpy(float)
 f["raw_close"]=raw.close.to_numpy(float)
 # Features and signals are adjusted; execution remains unadjusted.
 f["corporate_action_window"]=f.corporate_action_window.fillna(True).astype(bool)
 return f

def hold_gate(d):
 action=d.corporate_action_window.fillna(True).astype(bool)
 x=(d.distance_sma_200>=.25)&(d.distance_ema_20>0)&d.rsi14.between(50,65)&(d.plus_di14>d.minus_di14)&~action
 return x.fillna(False).to_numpy(bool)

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--raw-bars",default="data/top10_official_raw/bars")
 p.add_argument("--actions",default="data/top10_actions")
 p.add_argument("--output",default="artifacts/top10_frozen_high_hold")
 a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 rows=[];blocked=[]
 for symbol in SYMBOLS:
  try:
   raw=pd.read_csv(Path(a.raw_bars)/f"{symbol}.csv",parse_dates=["date"])
   raw=raw.sort_values("date").drop_duplicates("date").reset_index(drop=True)
   if len(raw)<260:raise RuntimeError(f"insufficient raw history {len(raw)}")
   ev,cashdiv,ref=read_events(a.actions,symbol)
   limits=build_limits(raw,ref,symbol)
   d=make_features(raw,ev)
   low,high,risk=build_candidate_signals(d)
   hold=hold_gate(d)
   start_idx=np.flatnonzero((d.date.dt.year>=2015).to_numpy())
   if not len(start_idx):raise RuntimeError("no 2015+ data")
   start=int(start_idx[0]);end=len(d)-1
   for slip in SLIPPAGE_BPS:
    sim=simulate(d,low,high,risk,limits,cashdiv,slip,start,end,high_hold_signal=hold)
    m={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
    rows.append({"symbol":symbol,"name":NAMES[symbol],"variant":"frozen_high_hold","slippage_bps":slip,**m})
    if slip==10:
     pd.DataFrame(sim["trades"]).to_csv(out/f"trades_{symbol}_10bps.csv",index=False)
     pd.DataFrame(sim["equity_curve"]).to_csv(out/f"equity_{symbol}_10bps.csv",index=False)
     pd.DataFrame(annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL)).to_csv(out/f"annual_{symbol}_10bps.csv",index=False)
   print(f"[DONE] {symbol} {NAMES[symbol]}",flush=True)
  except Exception as e:
   blocked.append({"symbol":symbol,"name":NAMES[symbol],"reason":f"{type(e).__name__}: {e}"})
   print(f"[BLOCKED] {symbol}: {e}",flush=True)
 result={"classification":"FIXED_TSMC_RULE_TRANSFER_AUDIT_NOT_PER_SYMBOL_RETUNED",
  "period":"2015 onward through last official raw bar","symbols_expected":len(SYMBOLS),
  "symbols_completed":len(set(x["symbol"] for x in rows)),"blocked":blocked,
  "costs":{"commission_per_side":.001425,"minimum_commission_ntd":20,"sell_tax":.003,"slippage_bps":list(SLIPPAGE_BPS)},
  "target":{"cagr_min":.50,"win_rate_min":.8695652173913043,"max_drawdown_floor":-.25},
  "status":"COMPLETE" if len(set(x["symbol"] for x in rows))==len(SYMBOLS) and not blocked else "BLOCKED_OR_PARTIAL",
  "metrics":rows,"rules":RULES,
  "notes":["No per-symbol parameter tuning; frozen TSMC rule family only.",
   "Raw official TWSE OHLC used for fills; corporate-action back-adjustment used for signal indicators.",
   "Any missing action data, missing limit reference, or OHLC-limit conflict blocks that symbol.",
   "Single-symbol results do not represent shared-capital portfolio results."]}
 (out/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
 pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
 print(json.dumps({k:v for k,v in result.items() if k!="metrics"},ensure_ascii=False,indent=2,default=float))
if __name__=="__main__":main()
