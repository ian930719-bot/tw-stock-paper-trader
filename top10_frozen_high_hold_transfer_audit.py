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

DEFAULT_SYMBOLS=["2330","2454","1303","2327","2408","2308","3037","3008","2383","2303"]
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
 root=Path(action_root)
 manifest=root/"manifest.json"
 if not manifest.exists():raise RuntimeError("Missing official TWT49U manifest")
 m=json.loads(manifest.read_text(encoding="utf-8"))
 if not m.get("complete"):raise RuntimeError(f"Official TWT49U collection incomplete: {m}")
 special_path=root/"special_events.csv"
 if special_path.exists():
  special=pd.read_csv(special_path,dtype={"symbol":str})
  special=special[special.symbol.astype(str).str.strip()==symbol]
  if not special.empty:
   raise RuntimeError(f"{symbol}: official capital-reduction/par-value events require a share-count/cash-refund model: {special.to_dict('records')[:3]}")
 p=root/"twt49u_all.csv"
 if not p.exists():raise RuntimeError("Missing global official TWT49U event table")
 all_events=pd.read_csv(p,dtype={"symbol":str})
 ev=all_events[all_events.symbol.astype(str).str.strip()==symbol].copy()
 if ev.empty:return pd.DataFrame(columns=["symbol","date","adjustment_ratio"]),{}, {}, {}
 ev["date"]=pd.to_datetime(ev.date,errors="coerce").dt.normalize()
 ev["symbol"]=ev.symbol.astype(str).str.strip()
 ev["adjustment_ratio"]=pd.to_numeric(ev.adjustment_ratio,errors="coerce")
 ev=ev.dropna(subset=["date","adjustment_ratio"])
 cash={}
 refs={}
 limits={}
 for r in ev.itertuples(index=False):
  day=pd.Timestamp(r.date).strftime("%Y-%m-%d")
  if str(getattr(r,"cash_dividend_status","resolved"))!="resolved":
   raise RuntimeError(f"{symbol}: mixed rights/dividend event needs separate official cash component {day}")
  stock_value=getattr(r,"stock_dividend",np.nan)
  if pd.notna(stock_value) and float(stock_value)>0:
   raise RuntimeError(f"{symbol}: stock dividend event needs share-count adjustment model {day}")
  cd=getattr(r,"cash_dividend",np.nan)
  if pd.notna(cd) and float(cd)>0:cash[day]=float(cd)
  ref=float(r.reference_price);up=float(r.limit_up);down=float(r.limit_down)
  if day in refs and abs(refs[day]-ref)>1e-6:raise RuntimeError(f"{symbol} duplicate TWT49U reference conflict {day}")
  refs[day]=ref
  limits[day]=(up,down)
 ev["source"]="TWSE TWT49U"
 return ev,cash,refs,limits

def build_limits(bars,ref_prices,event_limits,symbol):
 out={};prev_close=None;missing=[];conflicts=[];seen=False
 for r in bars.itertuples(index=False):
  day=pd.Timestamp(r.date).strftime("%Y-%m-%d")
  note=str(getattr(r,"note","")).strip()
  ohlc=[getattr(r,x,np.nan) for x in ("open","high","low","close")]
  volume=float(getattr(r,"volume",0) or 0)
  if volume<=0 or not all(np.isfinite(float(x)) for x in ohlc):
   if day>="2015-01-01":missing.append(day)
   prev_close=None;seen=True
   continue
  if note in {"**","＊＊"} and day not in event_limits:
   raise RuntimeError(f"{symbol}: official raw-bar special resumption marker without TWT49U event {day}")
  if day in event_limits:
   up,down=event_limits[day];src="TWSE TWT49U official event limit"
   ref=ref_prices[day]
  elif prev_close is not None:
   ref=prev_close;band=.07 if day<"2015-06-01" else .10
   up=limit_price(ref,band,"UP");down=limit_price(ref,band,"DOWN");src="TWSE official band/tick rules from prior traded close"
  elif not seen or day<"2015-01-01":
   seen=True
   prev_close=float(r.close)
   continue
  else:
   if day>="2015-01-01":missing.append(day)
   continue
  hi=float(r.high);lo=float(r.low)
  if hi>up+1e-6 or lo<down-1e-6:
   conflicts.append({"date":day,"high":hi,"low":lo,"up":up,"down":down,"reference":ref,"source":src})
  out[day]=(up,down)
  seen=True
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
 p.add_argument("--raw-bars",default="data/twse_marketwide_turnover/symbol_bars")
 p.add_argument("--universe-file",default="data/twse_marketwide_turnover/historical_daily_top10_by_turnover.csv")
 p.add_argument("--actions",default="data/top10_actions_twt49u")
 p.add_argument("--output",default="artifacts/top10_frozen_high_hold")
 a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 rows=[];blocked=[]
 universe_path=Path(a.universe_file)
 use_universe_filter=universe_path.exists()
 if use_universe_filter:
  uni=pd.read_csv(universe_path,dtype={"symbol":str})
  symbols=sorted(uni.symbol.astype(str).str.strip().unique())
  name_map=uni.drop_duplicates("symbol").set_index("symbol").name.astype(str).to_dict()
  active_dates_by_symbol={s:set(g.date.astype(str)) for s,g in uni.groupby("symbol")}
 else:
  symbols=sorted(p.stem for p in Path(a.raw_bars).glob("*.csv")) or DEFAULT_SYMBOLS
  name_map={}
  active_dates_by_symbol={}
 for symbol in symbols:
  try:
   raw_full=pd.read_csv(Path(a.raw_bars)/f"{symbol}.csv",parse_dates=["date"])
   raw_full=raw_full.sort_values("date").drop_duplicates("date").reset_index(drop=True)
   if len(raw_full)<260:raise RuntimeError(f"insufficient raw history {len(raw_full)}")
   ev,cashdiv,ref,event_limits=read_events(a.actions,symbol)
   limits=build_limits(raw_full,ref,event_limits,symbol)
   raw=raw_full.dropna(subset=["open","high","low","close"]).copy()
   raw=raw[raw.volume>0].reset_index(drop=True)
   if len(raw)<260:raise RuntimeError(f"insufficient traded raw history after no-trade filtering {len(raw)}")
   d=make_features(raw,ev)
   low,high,risk=build_candidate_signals(d)
   if use_universe_filter:
    active_dates=active_dates_by_symbol.get(symbol,set())
    active_mask=d.date.dt.strftime("%Y-%m-%d").isin(active_dates).to_numpy(bool)
    low &= active_mask
   else:
    active_mask=np.ones(len(d),dtype=bool)
   hold=hold_gate(d)
   start_idx=np.flatnonzero((d.date.dt.year>=2015).to_numpy())
   if not len(start_idx):raise RuntimeError("no 2015+ data")
   start=int(start_idx[0]);end=len(d)-1
   for slip in SLIPPAGE_BPS:
    sim=simulate(d,low,high,risk,limits,cashdiv,slip,start,end,high_hold_signal=hold)
    m={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
    rows.append({"symbol":symbol,"name":name_map.get(symbol,NAMES.get(symbol,str(raw.name.iloc[0]) if "name" in raw else symbol)),"variant":"frozen_high_hold_dynamic_top10_entry_filter","slippage_bps":slip,"top10_active_days":int(active_mask.sum()),"low_signal_days_after_universe_filter":int(low.sum()),**m})
    if slip==10:
     pd.DataFrame(sim["trades"]).to_csv(out/f"trades_{symbol}_10bps.csv",index=False)
     pd.DataFrame(sim["equity_curve"]).to_csv(out/f"equity_{symbol}_10bps.csv",index=False)
     pd.DataFrame(annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL)).to_csv(out/f"annual_{symbol}_10bps.csv",index=False)
   print(f"[DONE] {symbol} {name_map.get(symbol,NAMES.get(symbol,symbol))}",flush=True)
  except Exception as e:
   blocked.append({"symbol":symbol,"name":name_map.get(symbol,NAMES.get(symbol,symbol)),"reason":f"{type(e).__name__}: {e}"})
   print(f"[BLOCKED] {symbol}: {e}",flush=True)
 result={"classification":"FIXED_TSMC_RULE_TRANSFER_AUDIT_DAILY_TOP10_TURNOVER_UNIVERSE",
  "period":"2015 onward through last official raw bar","symbols_expected":len(symbols),
  "symbols_completed":len(set(x["symbol"] for x in rows)),"blocked":blocked,
  "costs":{"commission_per_side":.001425,"minimum_commission_ntd":20,"sell_tax":.003,"slippage_bps":list(SLIPPAGE_BPS)},
  "target":{"cagr_min":.50,"win_rate_min":.8695652173913043,"max_drawdown_floor":-.25},
  "status":"COMPLETE" if len(set(x["symbol"] for x in rows))==len(symbols) and not blocked else "BLOCKED_OR_PARTIAL",
  "metrics":rows,"rules":RULES,
  "notes":["No per-symbol parameter tuning; frozen TSMC rule family only.",
   "Entry eligibility uses the official same-day closing turnover top ten; signals are known after close and orders execute no earlier than the next session.",
   "Raw official TWSE OHLC used for fills; corporate-action back-adjustment used for signal indicators.",
   "Any missing action data, missing limit reference, or OHLC-limit conflict blocks that symbol.",
   "Single-symbol results do not represent shared-capital portfolio results."]}
 (out/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
 pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
 print(json.dumps({k:v for k,v in result.items() if k!="metrics"},ensure_ascii=False,indent=2,default=float))
if __name__=="__main__":main()
