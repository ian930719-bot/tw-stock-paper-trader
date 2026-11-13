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
from tsmc_2330_price_limits import limit_price as official_limit_price
from tsmc_2330_trend_reentry_annual_oof import adverse_fill, commission, sell_tax

DEFAULT_SYMBOLS=["2330","2454","1303","2327","2408","2308","3037","3008","2383","2303"]
NAMES={"2330":"台積電","2454":"聯發科","1303":"南亞","2327":"國巨","2408":"南亞科","2308":"台達電","3037":"欣興","3008":"大立光","2383":"台光電","2303":"聯電"}

def limit_price(ref,band,side):
 return official_limit_price(ref,band,side)

def read_events(action_root,symbol):
 root=Path(action_root)
 manifest=root/"manifest.json"
 if not manifest.exists():raise RuntimeError("Missing official TWT49U manifest")
 m=json.loads(manifest.read_text(encoding="utf-8"))
 if not m.get("complete"):raise RuntimeError(f"Official TWT49U collection incomplete: {m}")
 special_manifest_path=root/"special_events_manifest.json"
 if not special_manifest_path.exists():raise RuntimeError("Missing official special-action coverage manifest")
 special_manifest=json.loads(special_manifest_path.read_text(encoding="utf-8"))
 if not special_manifest.get("complete"):raise RuntimeError(f"Official special-action history incomplete: {special_manifest.get('skipped_periods')}")
 special_path=root/"special_events.csv"
 if not special_path.exists():raise RuntimeError("Missing official TWTAUU/TWTB8U event table")
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


def run_shared_capital_portfolio(prepared, universe, slippage_bps=10, initial_capital=INITIAL_CAPITAL):
 """One-position, all-in/all-out portfolio using the frozen TSMC rules.

 The top-ten membership is known at signal close; the order executes next
 session open. If any member of a day's top ten failed the official-data gates,
 that day's new entries are disabled rather than silently selecting a lower
 ranked survivor.
 """
 if universe is None or universe.empty or not prepared:
  return {"status":"NOT_RUN","reason":"Missing point-in-time universe or no eligible symbols"}
 u=universe.copy()
 u["date"]=pd.to_datetime(u.date,errors="coerce").dt.strftime("%Y-%m-%d")
 u["symbol"]=u.symbol.astype(str).str.strip()
 u["rank_by_volume"]=pd.to_numeric(u.rank_by_volume,errors="coerce")
 top10_by_date={}
 for day,g in u.groupby("date",sort=True):
  top10_by_date[day]=[(str(r.symbol),int(r.rank_by_volume) if pd.notna(r.rank_by_volume) else 999)
                      for r in g.sort_values(["rank_by_volume","turnover_ntd"],ascending=[True,False]).itertuples()]
 dates=sorted(d for d in top10_by_date if d>="2015-01-01")
 if len(dates)<260:return {"status":"NOT_RUN","reason":"Insufficient point-in-time top-ten dates"}
 index_maps={}
 for sym,x in prepared.items():
  index_maps[sym]={pd.Timestamp(dt).strftime("%Y-%m-%d"):i for i,dt in enumerate(x["d"].date)}
 total_top10_rows=sum(len(top10_by_date[d]) for d in dates)
 eligible_top10_rows=sum(1 for d in dates for sym,_ in top10_by_date[d] if sym in prepared)
 complete_days=sum(1 for d in dates if all(sym in prepared for sym,_ in top10_by_date[d]))
 coverage=complete_days/len(dates) if dates else 0.0

 cash=float(initial_capital);position=None;shares=0;entry_date=None;entry_reason=None
 entry_cost=0.0;trade_dividend=0.0;pending=None;high_wait_start=None
 risk_exit_fired=False;cooldown_until=-1;trades=[];curve=[]
 total_commission=total_tax=total_dividend=0.0;total_slippage=0.0
 missing_limit=locked_limit=outside_limit=0;exposure_days=0;suppressed_high=0
 for gi,day in enumerate(dates):
  # Ex-dividend entitlement belongs to the position held before today's open.
  if position is not None and day in prepared[position]["dividends"]:
   credit=shares*prepared[position]["dividends"][day]
   cash+=credit;trade_dividend+=credit;total_dividend+=credit
  # Execute any order submitted after the previous close.
  if pending is not None:
   sym=pending["symbol"];x=prepared[sym];di=index_maps[sym].get(day)
   if di is None or day not in x["limits"]:
    missing_limit+=1
    if pending["side"]=="BUY":pending=None
   else:
    row=x["d"].iloc[di];up,down=x["limits"][day]
    op=float(row.raw_open);hi=float(row.raw_high);lo=float(row.raw_low)
    if not all(np.isfinite(v) for v in (op,hi,lo,up,down)):
     missing_limit+=1
     if pending["side"]=="BUY":pending=None
    elif (pending["side"]=="BUY" and op>=up-1e-8) or (pending["side"]=="SELL" and op<=down+1e-8):
     locked_limit+=1
     if pending["side"]=="BUY":pending=None
    else:
     fill=adverse_fill(op,pending["side"],slippage_bps)
     if fill>up+1e-8 or fill<down-1e-8:
      outside_limit+=1
      if pending["side"]=="BUY":pending=None
     elif pending["side"]=="BUY" and position is None:
      qty=int(cash//fill)
      while qty>0 and qty*fill+commission(qty*fill)>cash+1e-8:qty-=1
      if qty>0:
       notional=qty*fill;fee=commission(notional);cash-=notional+fee
       shares=qty;position=sym;entry_date=day;entry_reason=pending["reason"]
       entry_cost=notional+fee;trade_dividend=0.0;risk_exit_fired=False;high_wait_start=None
       total_commission+=fee;total_slippage+=qty*(fill-op)
      pending=None
     elif pending["side"]=="SELL" and position==sym and shares>0:
      notional=shares*fill;fee=commission(notional);tax=sell_tax(notional)
      proceeds=notional-fee-tax;cash+=proceeds
      pnl=proceeds+trade_dividend-entry_cost
      trades.append({"symbol":sym,"entry_date":entry_date,"entry_reason":entry_reason,
       "exit_date":day,"exit_reason":pending["reason"],"shares":shares,
       "entry_cost":round(entry_cost,2),"exit_net_proceeds":round(proceeds,2),
       "cash_dividends":round(trade_dividend,2),"net_pnl":round(pnl,2),
       "net_return":pnl/entry_cost if entry_cost else None})
      total_commission+=fee;total_tax+=tax;total_slippage+=shares*(op-fill)
      if pending["reason"]=="sma200_risk_exit":cooldown_until=gi+RULES["risk_exit_cooldown_sessions"]
      position=None;shares=0;entry_date=None;entry_reason=None;entry_cost=0.0;trade_dividend=0.0
      high_wait_start=None;pending=None
  # Mark portfolio equity at the close.
  if position is not None:
   di=index_maps[position].get(day)
   if di is None:raise RuntimeError(f"Held position {position} missing official bar on {day}")
   row=prepared[position]["d"].iloc[di]
   close=float(row.raw_close);equity=cash+shares*close;exposure_days+=1
  else:
   close=None;equity=cash
  curve.append({"date":day,"equity":equity,"cash":cash,"position":position,
                "shares":shares,"close":close,"pending_order":pending["side"] if pending else None})
  if pending is not None or gi>=len(dates)-1:continue
  if position is None:
   # Fail closed on a day when any member of the daily top ten is not audited.
   members=top10_by_date.get(day,[])
   if not members or not all(sym in prepared for sym,_ in members):continue
   if gi<cooldown_until:continue
   candidates=[]
   for sym,rank in members:
    x=prepared[sym];di=index_maps[sym].get(day)
    if di is not None and bool(x["low"][di]):candidates.append((rank,sym))
   if candidates:
    rank,sym=min(candidates,key=lambda z:z[0])
    pending={"side":"BUY","symbol":sym,"reason":"base_low_after_filters_top10_volume_rank"}
  else:
   x=prepared[position];di=index_maps[position].get(day)
   if di is None:continue
   row=x["d"].iloc[di]
   if bool(x["risk"][di]) and not risk_exit_fired:
    risk_exit_fired=True;high_wait_start=None
    pending={"side":"SELL","symbol":position,"reason":"sma200_risk_exit"}
   elif high_wait_start is None and bool(x["high"][di]):
    if bool(x["hold"][di]):suppressed_high+=1
    else:
     dist_sma20=float(row.distance_sma_20);rsi=float(row.rsi14)
     if dist_sma20<0.0 or rsi<58.0:
      pending={"side":"SELL","symbol":position,"reason":"high_confirm_immediate"}
     else:high_wait_start=gi
   elif high_wait_start is not None and bool(x["hold"][di]):
    suppressed_high+=1;high_wait_start=None
   elif high_wait_start is not None:
    waited=gi-high_wait_start;rsi=float(row.rsi14)
    if rsi<48.0 or waited>=14:
     pending={"side":"SELL","symbol":position,"reason":"high_confirm_rsi_or_timeout"}
 cdf=pd.DataFrame(curve);eq=cdf.equity.astype(float);dd=eq/eq.cummax()-1
 years=(pd.Timestamp(cdf.date.iloc[-1])-pd.Timestamp(cdf.date.iloc[0])).days/365.25
 final=float(eq.iloc[-1]);tdf=pd.DataFrame(trades)
 open_position=None
 if position is not None:
  open_position={"symbol":position,"entry_date":entry_date,"shares":shares,"entry_cost":round(entry_cost,2),
                 "cash_dividends":round(trade_dividend,2),"last_close":close,"mark_to_market_equity":final}
 return {"status":"RUN_COMPLETE_WITH_FULL_UNIVERSE_COVERAGE" if coverage>=.99 else "RUN_COMPLETE_WITH_INCOMPLETE_UNIVERSE_COVERAGE",
  "initial_capital":initial_capital,"final_equity_mark_to_market":final,
  "cagr_mark_to_market":(final/initial_capital)**(1/years)-1 if years>0 and final>0 else None,
  "max_drawdown":float(dd.min()),"completed_trades":len(trades),
  "win_rate_completed_only":float((tdf.net_pnl>0).mean()) if len(tdf) else None,
  "exposure_ratio":exposure_days/len(cdf),"total_commission":total_commission,
  "total_sell_tax":total_tax,"estimated_slippage_cost":total_slippage,
  "total_cash_dividends":total_dividend,"blocked_missing_price_limit":missing_limit,
  "blocked_locked_limit":locked_limit,"blocked_slippage_outside_limit":outside_limit,
  "suppressed_high_signal_count":suppressed_high,"open_position":open_position,
  "daily_top10_dates":len(dates),"daily_top10_union_symbols":len(set(u.symbol)),
  "eligible_symbols":len(prepared),"top10_membership_rows":total_top10_rows,
  "eligible_top10_membership_rows":eligible_top10_rows,"complete_universe_days":complete_days,
  "complete_universe_day_ratio":coverage,"trades":trades,"equity_curve":curve,
  "method":"one shared NT$100,000 account; one position max; all-in/all-out; daily top-ten by official traded shares; same-day close signal and next-session open execution; skip new entries if any daily top-ten member failed official-data gates"}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--raw-bars",default="data/twse_marketwide_turnover/symbol_bars")
 p.add_argument("--universe-file",default="data/twse_marketwide_turnover/historical_daily_top10_by_volume.csv")
 p.add_argument("--actions",default="data/top10_actions_twt49u")
 p.add_argument("--output",default="artifacts/top10_frozen_high_hold")
 a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 rows=[];blocked=[];prepared={};tsmc_reproduction=None
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
   low_unfiltered=low.copy()
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
   if symbol=="2330":
    rep=simulate(d,low_unfiltered,high,risk,limits,cashdiv,10,start,end,high_hold_signal=hold)
    rep_metrics={k:v for k,v in rep.items() if k not in {"trades","equity_curve"}}
    expected={"cagr":0.376871,"win_rate":0.933333,"trade_count":15,"max_drawdown":-0.254384}
    tsmc_reproduction={"metrics":rep_metrics,"expected_previous_benchmark":expected,
     "cagr_delta_pp":(rep_metrics.get("cagr_mark_to_market")-expected["cagr"])*100 if rep_metrics.get("cagr_mark_to_market") is not None else None,
     "win_rate_delta_pp":(rep_metrics.get("win_rate_completed_only")-expected["win_rate"])*100 if rep_metrics.get("win_rate_completed_only") is not None else None,
     "trade_count_delta":rep_metrics.get("completed_trades")-expected["trade_count"] if rep_metrics.get("completed_trades") is not None else None,
     "max_drawdown_delta_pp":(rep_metrics.get("max_drawdown")-expected["max_drawdown"])*100 if rep_metrics.get("max_drawdown") is not None else None}
    deltas=[tsmc_reproduction.get("cagr_delta_pp"),tsmc_reproduction.get("win_rate_delta_pp"),
            tsmc_reproduction.get("trade_count_delta"),tsmc_reproduction.get("max_drawdown_delta_pp")]
    tsmc_reproduction["matches_benchmark_within_tolerance"]=bool(
     all(v is not None for v in deltas) and abs(deltas[0])<=1.0 and
     abs(deltas[1])<=5.0 and abs(deltas[2])<=1 and abs(deltas[3])<=1.0)
    pd.DataFrame(rep["trades"]).to_csv(out/"tsmc_2330_universe_free_reproduction_trades_10bps.csv",index=False)
    pd.DataFrame(rep["equity_curve"]).to_csv(out/"tsmc_2330_universe_free_reproduction_equity_10bps.csv",index=False)
   for slip in SLIPPAGE_BPS:
    sim=simulate(d,low,high,risk,limits,cashdiv,slip,start,end,high_hold_signal=hold)
    m={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
    row_out={"symbol":symbol,"name":name_map.get(symbol,NAMES.get(symbol,str(raw.name.iloc[0]) if "name" in raw else symbol)),"variant":"frozen_high_hold_dynamic_top10_entry_filter","slippage_bps":slip,"top10_active_days":int(active_mask.sum()),"low_signal_days_after_universe_filter":int(low.sum()),**m}
    row_out["cagr_pass_50pct"]=bool(row_out.get("cagr_mark_to_market") is not None and row_out["cagr_mark_to_market"]>=.50)
    row_out["win_rate_pass_floor"]=bool(row_out.get("win_rate_completed_only") is not None and row_out["win_rate_completed_only"]>=.8695652173913043)
    row_out["drawdown_pass_25pct"]=bool(row_out.get("max_drawdown") is not None and row_out["max_drawdown"]>=-.25)
    row_out["all_targets_pass"]=row_out["cagr_pass_50pct"] and row_out["win_rate_pass_floor"] and row_out["drawdown_pass_25pct"]
    rows.append(row_out)
    if slip==10:
     pd.DataFrame(sim["trades"]).to_csv(out/f"trades_{symbol}_10bps.csv",index=False)
     pd.DataFrame(sim["equity_curve"]).to_csv(out/f"equity_{symbol}_10bps.csv",index=False)
     pd.DataFrame(annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL)).to_csv(out/f"annual_{symbol}_10bps.csv",index=False)
   prepared[symbol]={"d":d,"low":low,"high":high,"risk":risk,"hold":hold,"limits":limits,"dividends":cashdiv}
   print(f"[DONE] {symbol} {name_map.get(symbol,NAMES.get(symbol,symbol))}",flush=True)
  except Exception as e:
   blocked.append({"symbol":symbol,"name":name_map.get(symbol,NAMES.get(symbol,symbol)),"reason":f"{type(e).__name__}: {e}"})
   print(f"[BLOCKED] {symbol}: {e}",flush=True)
 shared_portfolio=None
 if use_universe_filter and prepared:
  portfolio=run_shared_capital_portfolio(prepared,uni,slippage_bps=10,initial_capital=INITIAL_CAPITAL)
  pd.DataFrame(portfolio.get("trades",[])).to_csv(out/"shared_capital_trades_10bps.csv",index=False)
  pd.DataFrame(portfolio.get("equity_curve",[])).to_csv(out/"shared_capital_equity_10bps.csv",index=False)
  if portfolio.get("equity_curve"):
   pd.DataFrame(annual_metrics(portfolio["equity_curve"],portfolio["trades"],INITIAL_CAPITAL)).to_csv(out/"shared_capital_annual_10bps.csv",index=False)
  shared_portfolio={k:v for k,v in portfolio.items() if k not in {"trades","equity_curve"}}
 target_assessment=None
 if shared_portfolio:
  target_assessment={
   "cagr_50pct_pass":bool(shared_portfolio.get("cagr_mark_to_market") is not None and shared_portfolio["cagr_mark_to_market"]>=.50),
   "win_rate_86_9565pct_pass":bool(shared_portfolio.get("win_rate_completed_only") is not None and shared_portfolio["win_rate_completed_only"]>=.8695652173913043),
   "max_drawdown_25pct_pass":bool(shared_portfolio.get("max_drawdown") is not None and shared_portfolio["max_drawdown"]>=-.25),
   "universe_coverage_99pct_pass":bool(shared_portfolio.get("complete_universe_day_ratio",0)>=.99)}
  target_assessment["all_targets_and_coverage_pass"]=all(target_assessment.values())
 result={"classification":"FIXED_TSMC_RULE_TRANSFER_AUDIT_DAILY_TOP10_VOLUME_UNIVERSE",
  "period":"2015 onward through last official raw bar","symbols_expected":len(symbols),
  "symbols_completed":len(set(x["symbol"] for x in rows)),"blocked":blocked,
  "costs":{"commission_per_side":.001425,"minimum_commission_ntd":20,"sell_tax":.003,"slippage_bps":list(SLIPPAGE_BPS)},
  "target":{"cagr_min":.50,"win_rate_min":.8695652173913043,"max_drawdown_floor":-.25},
  "status":"COMPLETE" if len(set(x["symbol"] for x in rows))==len(symbols) and not blocked else "BLOCKED_OR_PARTIAL",
  "metrics":rows,"tsmc_2330_universe_free_reproduction":tsmc_reproduction,"shared_capital_portfolio":shared_portfolio,"target_assessment":target_assessment,"rules":RULES,
  "notes":["No per-symbol parameter tuning; frozen TSMC rule family only.",
   "Entry eligibility uses the official same-day closing traded-share top ten; signals are known after close and orders execute no earlier than the next session.",
   "Raw official TWSE OHLC used for fills; corporate-action back-adjustment used for signal indicators.",
   "TWT49U covers historical ex-rights/ex-dividend events from 2010; TWTAUU/TWTB8U special-event audit begins 2011, while performance measurement begins 2015.",
   "Any missing action data, missing limit reference, or OHLC-limit conflict blocks that symbol. The shared-capital portfolio also skips new entries when any daily top-ten constituent is unaudited.",
   "Single-symbol results are independent NT$100,000 per-stock transfer tests; shared_capital_portfolio is a separate one-position shared-NT$100,000 account."]}
 (out/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
 pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
 print(json.dumps({k:v for k,v in result.items() if k!="metrics"},ensure_ascii=False,indent=2,default=float))
if __name__=="__main__":main()
