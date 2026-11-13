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

def read_events(action_root,symbol_key,stock_id,market,accounting_start_date="2015-01-01"):
 root=Path(action_root)
 accounting_start=pd.Timestamp(accounting_start_date).normalize()
 manifest=root/"manifest.json"
 if not manifest.exists():raise RuntimeError("Missing official TWSE TWT49U manifest")
 m=json.loads(manifest.read_text(encoding="utf-8"))
 if not m.get("complete"):raise RuntimeError(f"Official TWT49U collection incomplete: {m}")
 special_manifest_path=root/"special_events_manifest.json"
 if not special_manifest_path.exists():raise RuntimeError("Missing official TWSE/TPEx special-action coverage manifest")
 special_manifest=json.loads(special_manifest_path.read_text(encoding="utf-8"))
 if not special_manifest.get("complete"):raise RuntimeError(f"Official special-action history incomplete: {special_manifest.get('skipped_periods')}")
 if market.upper()=="TPEX" and not special_manifest.get("tpex",{}).get("exDailyQ_rows"):
  raise RuntimeError("Missing verified TPEx exDailyQ coverage")
 special_path=root/"special_events.csv"
 if not special_path.exists():raise RuntimeError("Missing official TWSE/TPEx special-action event table")
 special=pd.read_csv(special_path,dtype={"symbol":str})
 if "market" not in special:special["market"]="TWSE"
 special["date"]=pd.to_datetime(special.date,errors="coerce")
 special=special[(special.market.astype(str).str.upper()==market.upper()) &
                 (special.symbol.astype(str).str.strip()==stock_id) &
                 (special.date>=pd.Timestamp("2014-01-01"))].copy()
 invalid_special_path=root/"special_events_invalid_rows.csv"
 if invalid_special_path.exists():
  invalid_special=pd.read_csv(invalid_special_path,dtype={"symbol":str})
  bad_special=invalid_special[(invalid_special.market.astype(str).str.upper()==market.upper()) &
                              (invalid_special.symbol.astype(str).str.strip()==stock_id)]
  if not bad_special.empty:
   raise RuntimeError(str(symbol_key)+": invalid official special-event references require manual reconciliation: "+str(bad_special.to_dict("records")[:3]))
 strategy_start=pd.Timestamp("2015-01-01")
 special_pre=special[special.date<strategy_start].copy()
 special_post=special[special.date>=strategy_start].copy()
 terms_path=Path(__file__).resolve().parent/"data"/"official_special_action_terms.csv"
 if terms_path.exists():
  terms=pd.read_csv(terms_path,dtype={"symbol":str})
  terms=terms[terms.symbol.astype(str).str.strip()==stock_id].copy()
  terms["event_date"]=pd.to_datetime(terms.event_date,errors="coerce").dt.strftime("%Y-%m-%d")
  special_terms={r.event_date:{"share_multiplier":float(r.share_multiplier),
    "cash_refund":float(r.cash_refund_per_old_share),"source_url":str(r.source_url)}
    for r in terms.itertuples(index=False)}
 else:special_terms={}
 if market.upper()=="TWSE":
  p=root/"twt49u_all.csv"
  if not p.exists():raise RuntimeError("Missing global official TWSE TWT49U event table")
  all_events=pd.read_csv(p,dtype={"symbol":str})
 else:
  p=root/"tpex_exdailyq_all.csv"
  if not p.exists():raise RuntimeError("Missing global official TPEx exDailyQ event table")
  all_events=pd.read_csv(p,dtype={"symbol":str})
  invalid_path=root/"tpex_exdailyq_invalid_rows.csv"
  if invalid_path.exists():
   invalid=pd.read_csv(invalid_path,dtype={"symbol":str})
   bad=invalid[invalid.symbol.astype(str).str.strip()==stock_id]
   if not bad.empty:
    raise RuntimeError(f"{symbol_key}: invalid official TPEx corporate-action reference rows require manual reconciliation: {bad.to_dict('records')[:3]}")
 ev=all_events[all_events.symbol.astype(str).str.strip()==stock_id].copy()
 if ev.empty:ev=pd.DataFrame(columns=["symbol","date","adjustment_ratio"])
 ev["date"]=pd.to_datetime(ev.date,errors="coerce").dt.normalize()
 ev=ev[ev.date>=pd.Timestamp("2014-01-01")].copy()
 ev["symbol"]=symbol_key
 ev["adjustment_ratio"]=pd.to_numeric(ev.adjustment_ratio,errors="coerce")
 ev=ev.dropna(subset=["date","adjustment_ratio"])
 cash={};refs={};limits={};share_adjustments={}
 # Ex-dividend entitlements are applied to the position held before the ex-date open.
 for r in ev.itertuples(index=False):
  day=pd.Timestamp(r.date).strftime("%Y-%m-%d")
  pre_strategy=pd.Timestamp(r.date)<strategy_start
  status=str(getattr(r,"cash_dividend_status","resolved"))
  action_type=str(getattr(r,"action_type",""))
  rights_value=getattr(r,"rights_issue_shares",np.nan)
  stock_value=getattr(r,"stock_dividend",np.nan)
  cash_value=getattr(r,"cash_dividend",np.nan)
  # Explicitly sourced component override for a mixed TWT49U row. The
  # override is accepted only when its cash/stock terms independently
  # reconcile to the exchange-published ex-rights reference price.
  override_path=Path(__file__).resolve().parent/"data"/"official_dividend_component_overrides.csv"
  if status!="resolved" and override_path.exists():
   _ov=pd.read_csv(override_path,dtype={"symbol":str})
   _ov=_ov[(_ov.symbol.astype(str).str.strip()==stock_id) & (_ov.event_date.astype(str)==day)]
   if len(_ov)==1:
    _o=_ov.iloc[0]
    _cash=float(_o.cash_dividend_per_share)
    _stock=float(_o.stock_dividend_per_share)
    _par=float(_o.par_value_per_share)
    _expected=(float(r.previous_close)-_cash)/(1.0+_stock/_par)
    if abs(_expected-float(r.reference_price))>0.03:
     raise RuntimeError(f"{symbol_key}: sourced dividend override fails official reference-price reconciliation on {day}: expected {_expected}, official {r.reference_price}")
    cash_value=_cash;stock_value=_stock;status="resolved"
  ref=float(r.reference_price)
  up=getattr(r,"limit_up",np.nan);down=getattr(r,"limit_down",np.nan)
  if pd.Timestamp(r.date).normalize()<accounting_start:
   if not all(np.isfinite(float(x)) and float(x)>0 for x in (ref,up,down)):
    raise RuntimeError(f"{symbol_key}: invalid official historical corporate-action reference {day}")
   refs[day]=ref;limits[day]=(float(up),float(down))
   # No portfolio position is held before accounting_start. The exchange-published
   # adjustment_ratio still adjusts indicator history; no historical cash/share
   # entitlement is credited to the OOS account.
   continue
  if not pre_strategy:
   if status!="resolved":
    raise RuntimeError(f"{symbol_key}: unresolved official cash-dividend component {day}")
   has_cash_rights_issue=pd.notna(rights_value) and float(rights_value)>0
   if "權" in action_type:
    if pd.isna(stock_value) or (float(stock_value)<=0 and not has_cash_rights_issue):
     raise RuntimeError(f"{symbol_key}: post-2015 rights/stock event lacks a verified stock-dividend component {day} ({action_type})")
    if float(stock_value)>0:
     prev=float(r.previous_close);ref=float(r.reference_price)
     cd=float(cash_value) if pd.notna(cash_value) else 0.0
     # TPEx reports free stock dividends as shares per 1,000 old shares.
     # TWSE TWT49U reports a combined price value, so its share multiplier
     # remains reconciled to the official reference price.
     multiplier=(1.0+float(stock_value)/1000.0) if market.upper()=="TPEX" else (prev-cd)/ref
     if not np.isfinite(multiplier) or multiplier<=1.0:
      raise RuntimeError(f"{symbol_key}: invalid stock-dividend share multiplier {day}: {multiplier}")
     share_adjustments[day]=multiplier
   # Cash-capital-increase rights are deliberately not exercised. The official
   # ex-right adjustment ratio still adjusts indicator history; no new shares
   # or subscription cash are added to the all-in/all-out account.
   if pd.notna(cash_value) and float(cash_value)>0:cash[day]=float(cash_value)
  if not np.isfinite(float(up)) or not np.isfinite(float(down)):
   raise RuntimeError(f"{symbol_key}: missing official event-day limit price {day}")
  up=float(up);down=float(down)
  if day in refs and abs(refs[day]-ref)>1e-6:raise RuntimeError(f"{symbol_key} duplicate corporate-action reference conflict {day}")
  refs[day]=ref;limits[day]=(up,down)
 # Pre-2015 special events only alter the indicator price history; no position existed then.
 pre_rows=[]
 for r in special_pre.itertuples(index=False):
  day=pd.Timestamp(r.date).strftime("%Y-%m-%d")
  prev=float(r.previous_close);ref=float(r.reference_price)
  up=float(r.limit_up);down=float(r.limit_down)
  if not all(np.isfinite(x) and x>0 for x in (prev,ref,up,down)):
   raise RuntimeError(f"{symbol_key}: invalid pre-2015 special-action reference {day}")
  if day in refs and abs(refs[day]-ref)>1e-6:
   raise RuntimeError(f"{symbol_key}: special-action reference conflicts with dividend event {day}")
  refs[day]=ref;limits[day]=(up,down)
  pre_rows.append({"symbol":symbol_key,"date":pd.Timestamp(r.date),
                   "adjustment_ratio":ref/prev,
                   "source":str(getattr(r,"source","official special corporate action"))})
 # Post-2015 loss-recovery reductions and par-value changes have verifiable share ratios.
 post_rows=[]
 for r in special_post.itertuples(index=False):
  day=pd.Timestamp(r.date).strftime("%Y-%m-%d")
  prev=float(r.previous_close);ref=float(r.reference_price)
  up=float(r.limit_up);down=float(r.limit_down)
  event_type=str(getattr(r,"event_type",""))
  reason=str(getattr(r,"reason",""))
  if not all(np.isfinite(x) and x>0 for x in (prev,ref,up,down)):
   raise RuntimeError(f"{symbol_key}: invalid post-2015 special-action reference {day}")
  if pd.Timestamp(r.date).normalize()<accounting_start:
   refs[day]=ref;limits[day]=(up,down)
   post_rows.append({"symbol":symbol_key,"date":pd.Timestamp(r.date),
                     "adjustment_ratio":ref/prev,
                     "source":str(getattr(r,"source","official special corporate action"))})
   # Historical share/cash terms are irrelevant to the OOS account, which
   # starts flat on accounting_start; retain the official price adjustment.
   continue
  if day in special_terms:
   multiplier=special_terms[day]["share_multiplier"]
   refund=special_terms[day]["cash_refund"]
   expected_ref=(prev-refund-cash.get(day,0.0))/multiplier
   if abs(expected_ref-ref)>0.03:
    raise RuntimeError(f"{symbol_key}: documented cash-refund/share terms do not reconcile to official reference price on {day}: expected {expected_ref}, official {ref}")
   cash[day]=cash.get(day,0.0)+refund
  elif event_type=="par_value_change" or "面額" in event_type:
   multiplier=prev/ref
  elif "彌補虧損" in reason:
   multiplier=prev/ref
  else:
   raise RuntimeError(f"{symbol_key}: post-2015 cash-refund/unsupported capital action needs detail-level share and refund terms: {day} {reason}")
  if not np.isfinite(multiplier) or multiplier<=0:
   raise RuntimeError(f"{symbol_key}: invalid special-event share multiplier {day}: {multiplier}")
  if day in refs and abs(refs[day]-ref)>1e-6:
   raise RuntimeError(f"{symbol_key}: special-action reference conflicts with dividend event {day}")
  refs[day]=ref;limits[day]=(up,down)
  share_adjustments[day]=share_adjustments.get(day,1.0)*multiplier
  post_rows.append({"symbol":symbol_key,"date":pd.Timestamp(r.date),
                    "adjustment_ratio":ref/prev,
                    "source":str(getattr(r,"source","official special corporate action"))})
 if pre_rows:ev=pd.concat([ev,pd.DataFrame(pre_rows)],ignore_index=True)
 if post_rows:ev=pd.concat([ev,pd.DataFrame(post_rows)],ignore_index=True)
 if not ev.empty and ev.duplicated(["symbol","date"]).any():
  dup=ev[ev.duplicated(["symbol","date"],keep=False)]
  raise RuntimeError(f"{symbol_key}: duplicate same-day corporate-action adjustments require reconciliation: {dup.to_dict('records')[:3]}")
 ev["source"]=f"{market.upper()} official ex-rights/ex-dividend and special-event adjustments"
 return ev,cash,refs,limits,share_adjustments

def build_limits(bars,ref_prices,event_limits,symbol,strict_from="2015-01-01"):
 out={};prev_close=None;missing=[];conflicts=[];seen=False
 for session_i,r in enumerate(bars.itertuples(index=False)):
  day=pd.Timestamp(r.date).strftime("%Y-%m-%d")
  note=str(getattr(r,"note","")).strip()
  ohlc=[getattr(r,x,np.nan) for x in ("open","high","low","close")]
  volume=float(getattr(r,"volume",0) or 0)
  if volume<=0:
   # A no-trade session has no new close. Preserve the last valid reference
   # instead of poisoning every later session with prev_close=None. If an
   # official ex-rights/capital-action reference occurs on this date, it becomes
   # the next session's reference even when no shares traded.
   if day in event_limits:
    prev_close=float(ref_prices[day])
   seen=True
   continue
  if not all(np.isfinite(float(x)) for x in ohlc):
   if day>=strict_from:missing.append(day)
   seen=True
   continue
  if note in {"**","＊＊"} and day not in event_limits:
   raise RuntimeError(f"{symbol}: official raw-bar special resumption marker without a matching official event {day}")
  if session_i<5 or day<"2014-01-01":
   seen=True;prev_close=float(r.close);continue
  if day in event_limits:
   up,down=event_limits[day];src="official exchange ex-rights/ex-dividend event limit"
   ref=ref_prices[day]
  elif prev_close is not None:
   ref=prev_close;band=.07 if day<"2015-06-01" else .10
   up=limit_price(ref,band,"UP");down=limit_price(ref,band,"DOWN");src="TWSE/TPEx official band/tick rules from prior traded close"
  elif day<strict_from:
   # Price limits before the OOS execution window are not used for fills;
   # seed the next reference from the observed close and continue warm-up.
   seen=True
   prev_close=float(r.close)
   continue
  else:
   missing.append(day)
   continue
  hi=float(r.high);lo=float(r.low)
  if day>=strict_from and (hi>up+1e-6 or lo<down-1e-6):
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


def run_shared_capital_portfolio(prepared, universe, slippage_bps=10, initial_capital=INITIAL_CAPITAL, trailing_atr_multiple=None, atr_activation_profit_pct=0.0, trailing_peak_drawdown_pct=None):
 """One-position, all-in/all-out portfolio using the frozen TSMC rules.

 The top-ten membership is known at signal close; the order executes next
 session open. If any member of a day's top ten failed the official-data gates,
 that day's new entries are disabled rather than silently selecting a lower
 ranked survivor.
 """
 if universe is None or universe.empty or not prepared:
  return {"status":"NOT_RUN","reason":"Missing point-in-time universe or no eligible symbols"}
 if trailing_atr_multiple is not None and (not np.isfinite(trailing_atr_multiple) or trailing_atr_multiple<=0):
  raise ValueError("trailing_atr_multiple must be positive finite or None")
 if not np.isfinite(atr_activation_profit_pct) or not 0.0<=atr_activation_profit_pct<1.0:
  raise ValueError("atr_activation_profit_pct must be in [0,1)")
 if trailing_peak_drawdown_pct is not None and (not np.isfinite(trailing_peak_drawdown_pct) or not 0.0<trailing_peak_drawdown_pct<1.0):
  raise ValueError("trailing_peak_drawdown_pct must be in (0,1) or None")
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
 entry_cost=0.0;trade_dividend=0.0;pending=None;high_wait_start=None;position_last_close=None
 high_water_mark=None;atr_activation_entry_price=None;atr_stop_exit_count=0;peak_stop_exit_count=0
 risk_exit_fired=False;cooldown_until=-1;trades=[];curve=[]
 total_commission=total_tax=total_dividend=0.0;total_slippage=0.0
 missing_limit=locked_limit=outside_limit=0;exposure_days=0;suppressed_high=0
 for gi,day in enumerate(dates):
  # Ex-dividend entitlement belongs to the position held before today's open.
  if position is not None and day in prepared[position]["dividends"]:
   credit=shares*prepared[position]["dividends"][day]
   cash+=credit;trade_dividend+=credit;total_dividend+=credit
  if position is not None and day in prepared[position].get("share_adjustments",{}):
   multiplier=float(prepared[position]["share_adjustments"][day])
   if not np.isfinite(multiplier) or multiplier<=0:
    raise RuntimeError(f"Invalid share adjustment for {position} on {day}: {multiplier}")
   shares=int(math.floor(shares*multiplier+1e-9))
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
       high_water_mark=float(fill);atr_activation_entry_price=float(fill);position_last_close=None
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
      if pending["reason"]=="atr_trailing_stop":atr_stop_exit_count+=1
      if pending["reason"]=="profit_peak_drawdown_stop":peak_stop_exit_count+=1
      position=None;shares=0;entry_date=None;entry_reason=None;entry_cost=0.0;trade_dividend=0.0
      high_water_mark=None;atr_activation_entry_price=None;position_last_close=None
      high_wait_start=None;pending=None
  # Mark portfolio equity at the close.
  stale_mark=False
  if position is not None:
   di=index_maps[position].get(day)
   if di is None:
    if position_last_close is None:raise RuntimeError(f"Held position {position} has no prior official mark on {day}")
    close=position_last_close;stale_mark=True
   else:
    row=prepared[position]["d"].iloc[di]
    close=float(row.raw_close);position_last_close=close
   equity=cash+shares*close;exposure_days+=1
  else:
   close=None;equity=cash
  curve.append({"date":day,"equity":equity,"cash":cash,"position":position,
                "shares":shares,"close":close,"stale_mark":stale_mark,
                "pending_order":pending["side"] if pending else None})
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
   elif trailing_atr_multiple is not None or trailing_peak_drawdown_pct is not None:
    high_water_mark=max(float(high_water_mark or row.raw_high),float(row.raw_high))
    activated=(atr_activation_entry_price is not None and high_water_mark>=atr_activation_entry_price*(1.0+atr_activation_profit_pct))
    atr_hit=False
    if trailing_atr_multiple is not None:
     atr_pct=float(row.atr_pct14) if pd.notna(row.get("atr_pct14",np.nan)) else np.nan
     atr_value=float(row.raw_close)*atr_pct if np.isfinite(atr_pct) and atr_pct>0 else np.nan
     atr_hit=activated and np.isfinite(atr_value) and float(row.raw_close)<=high_water_mark-trailing_atr_multiple*atr_value
    peak_hit=(activated and trailing_peak_drawdown_pct is not None and float(row.raw_close)<=high_water_mark*(1.0-trailing_peak_drawdown_pct))
    if atr_hit or peak_hit:
     high_wait_start=None
     pending={"side":"SELL","symbol":position,"reason":"atr_trailing_stop" if atr_hit else "profit_peak_drawdown_stop"}
   if pending is None and high_wait_start is None and bool(x["high"][di]):
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
  "atr_trailing_stop_exit_count":atr_stop_exit_count,"profit_peak_drawdown_stop_exit_count":peak_stop_exit_count,
  "trailing_atr_multiple":trailing_atr_multiple,"trailing_peak_drawdown_pct":trailing_peak_drawdown_pct,
  "atr_activation_profit_pct":atr_activation_profit_pct,
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
 rows=[];blocked=[];prepared={};tsmc_reproduction=None;annual_rows=[];trade_diagnostics=[];filter_replay_rows=[]
 universe_path=Path(a.universe_file)
 use_universe_filter=universe_path.exists()
 if use_universe_filter:
  uni=pd.read_csv(universe_path,dtype={"symbol":str,"stock_id":str,"market":str})
  symbols=sorted(uni.symbol.astype(str).str.strip().unique())
  name_map=uni.drop_duplicates("symbol").set_index("symbol").name.astype(str).to_dict()
  stock_id_map=uni.drop_duplicates("symbol").set_index("symbol").stock_id.astype(str).to_dict()
  market_map=uni.drop_duplicates("symbol").set_index("symbol").market.astype(str).to_dict()
  active_dates_by_symbol={s:set(g.date.astype(str)) for s,g in uni.groupby("symbol")}
 else:
  symbols=sorted(p.stem for p in Path(a.raw_bars).glob("*.csv")) or DEFAULT_SYMBOLS
  name_map={}
  stock_id_map={s:(s.split("_",1)[1] if "_" in s else s) for s in symbols}
  market_map={s:(s.split("_",1)[0] if "_" in s else "TWSE") for s in symbols}
  active_dates_by_symbol={}
 for symbol in symbols:
  try:
   raw_full=pd.read_csv(Path(a.raw_bars)/f"{symbol}.csv",parse_dates=["date"])
   raw_full=raw_full.sort_values("date").drop_duplicates("date").reset_index(drop=True)
   if len(raw_full)<1:raise RuntimeError("no official raw history")
   ev,cashdiv,ref,event_limits,share_adjustments=read_events(a.actions,symbol,stock_id_map[symbol],market_map[symbol])
   limits=build_limits(raw_full,ref,event_limits,symbol)
   raw=raw_full.dropna(subset=["open","high","low","close"]).copy()
   raw=raw[raw.volume>0].reset_index(drop=True)
   if len(raw)<1:raise RuntimeError("no traded raw history after no-trade filtering")
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
   # Frozen thresholds selected only from 2015-2021 training trades.
   # Evaluate entries strictly out-of-time from 2022-01-01; orders still fill next session open.
   test_idx=np.flatnonzero((d.date>=pd.Timestamp("2022-01-01")).to_numpy())
   if len(test_idx):
    test_start=int(test_idx[0]);test_end=len(d)-1
    candidate_masks={
     "baseline":np.ones(len(d),dtype=bool),
     "sma50_slope_ge_0.006062":pd.to_numeric(d.sma50_slope_10,errors="coerce").ge(0.006062).fillna(False).to_numpy(bool),
     "distance_sma200_ge_0.073873":pd.to_numeric(d.distance_sma_200,errors="coerce").ge(0.073873).fillna(False).to_numpy(bool),
     "sma50_and_sma200":(pd.to_numeric(d.sma50_slope_10,errors="coerce").ge(0.006062)&pd.to_numeric(d.distance_sma_200,errors="coerce").ge(0.073873)).fillna(False).to_numpy(bool)
    }
    for filter_name,entry_mask in candidate_masks.items():
     test_low=low_unfiltered & entry_mask
     test_sim=simulate(d,test_low,high,risk,limits,cashdiv,10,test_start,test_end,
       high_hold_signal=hold,share_adjustments=share_adjustments)
     test_trades=pd.DataFrame(test_sim["trades"])
     filter_replay_rows.append({"symbol":symbol,"stock_id":stock_id_map[symbol],"name":name_map.get(symbol,NAMES.get(stock_id_map[symbol],symbol)),
       "filter":filter_name,"slippage_bps":10,"test_start":str(d.date.iloc[test_start].date()),
       "test_end":str(d.date.iloc[test_end].date()),
       **{k:v for k,v in test_sim.items() if k not in {"trades","equity_curve"}},
       "mean_trade_net_return":float(test_trades.net_return.mean()) if len(test_trades) else None,
       "median_trade_net_return":float(test_trades.net_return.median()) if len(test_trades) else None})
     pd.DataFrame(test_sim["trades"]).to_csv(out/f"filter_test_trades_{symbol}_{filter_name}_10bps.csv",index=False)
     pd.DataFrame(test_sim["equity_curve"]).to_csv(out/f"filter_test_equity_{symbol}_{filter_name}_10bps.csv",index=False)
   start_idx=np.flatnonzero((d.date.dt.year>=2015).to_numpy())
   if not len(start_idx):raise RuntimeError("no 2015+ data")
   start=int(start_idx[0]);end=len(d)-1
   if market_map[symbol]=="TWSE" and stock_id_map[symbol]=="2330":
    benchmark_end_candidates=np.flatnonzero((d.date<=pd.Timestamp("2026-11-11")).to_numpy())
    if not len(benchmark_end_candidates):raise RuntimeError("TSMC benchmark end date 2026-11-11 is unavailable")
    benchmark_end=int(benchmark_end_candidates[-1])
    rep=simulate(d,low_unfiltered,high,risk,limits,cashdiv,10,start,benchmark_end,high_hold_signal=hold,share_adjustments=share_adjustments)
    rep_metrics={k:v for k,v in rep.items() if k not in {"trades","equity_curve"}}
    expected={"period_end":"2026-11-11","cagr":0.376871,"win_rate":0.933333,"trade_count":15,"max_drawdown":-0.254384}
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
   # Drawdown overlay experiment: baseline is retained unchanged; each ATR
   # trail is close-confirmed and filled no earlier than the next session open.
   if not use_universe_filter or stock_id_map[symbol]=="2330":
    # Profit-gated ATR trails.
    for activation_pct in (0.05,0.10,0.15,0.20):
     for atr_multiple in (2.5,3.0,3.5):
      overlay=simulate(d,low,high,risk,limits,cashdiv,10,start,end,
        high_hold_signal=hold,share_adjustments=share_adjustments,
        trailing_atr_multiple=atr_multiple,atr_activation_profit_pct=activation_pct)
      om={k:v for k,v in overlay.items() if k not in {"trades","equity_curve"}}
      rows.append({"symbol":symbol,"stock_id":stock_id_map[symbol],"market":market_map[symbol],
        "name":name_map.get(symbol,NAMES.get(stock_id_map[symbol],symbol)),
        "variant":"profit_gated_atr_trailing_stop","slippage_bps":10,**om})
      if activation_pct==0.10 and atr_multiple==3.0:
       pd.DataFrame(overlay["trades"]).to_csv(out/f"trades_{symbol}_atr3_activation10pct_10bps.csv",index=False)
       pd.DataFrame(overlay["equity_curve"]).to_csv(out/f"equity_{symbol}_atr3_activation10pct_10bps.csv",index=False)
    # Fixed percentage trailing protection, activated only after unrealized
    # profit threshold. Width and activation are independently swept.
    for activation_pct in (0.05,0.10,0.15,0.20):
     for peak_dd in (0.05,0.08,0.10,0.12,0.15):
      overlay=simulate(d,low,high,risk,limits,cashdiv,10,start,end,
        high_hold_signal=hold,share_adjustments=share_adjustments,
        atr_activation_profit_pct=activation_pct,trailing_peak_drawdown_pct=peak_dd)
      om={k:v for k,v in overlay.items() if k not in {"trades","equity_curve"}}
      rows.append({"symbol":symbol,"stock_id":stock_id_map[symbol],"market":market_map[symbol],
        "name":name_map.get(symbol,NAMES.get(stock_id_map[symbol],symbol)),
        "variant":"profit_gated_peak_drawdown_stop","slippage_bps":10,**om})
      if activation_pct==0.10 and peak_dd==0.10:
       pd.DataFrame(overlay["trades"]).to_csv(out/f"trades_{symbol}_peak10_activation10_10bps.csv",index=False)
       pd.DataFrame(overlay["equity_curve"]).to_csv(out/f"equity_{symbol}_peak10_activation10_10bps.csv",index=False)
   for slip in SLIPPAGE_BPS:
    sim=simulate(d,low,high,risk,limits,cashdiv,slip,start,end,high_hold_signal=hold,share_adjustments=share_adjustments)
    if slip==10:
     for annual_row in annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL):
      annual_rows.append({"symbol":symbol,"stock_id":stock_id_map[symbol],"market":market_map[symbol],"name":name_map.get(symbol,symbol),**annual_row})
    m={k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
    row_out={"symbol":symbol,"stock_id":stock_id_map[symbol],"market":market_map[symbol],"name":name_map.get(symbol,NAMES.get(stock_id_map[symbol],str(raw.name.iloc[0]) if "name" in raw else symbol)),"variant":"frozen_high_hold_dynamic_top10_entry_filter","slippage_bps":slip,"share_adjustment_events":int(len(share_adjustments)),"top10_active_days":int(active_mask.sum()),"low_signal_days_after_universe_filter":int(low.sum()),**m}
    row_out["cagr_pass_50pct"]=bool(row_out.get("cagr_mark_to_market") is not None and row_out["cagr_mark_to_market"]>=.50)
    row_out["win_rate_pass_floor"]=bool(row_out.get("win_rate_completed_only") is not None and row_out["win_rate_completed_only"]>=.8695652173913043)
    row_out["drawdown_pass_25pct"]=bool(row_out.get("max_drawdown") is not None and row_out["max_drawdown"]>=-.25)
    row_out["all_targets_pass"]=row_out["cagr_pass_50pct"] and row_out["win_rate_pass_floor"] and row_out["drawdown_pass_25pct"]
    rows.append(row_out)
    if slip==10:
     date_index={pd.Timestamp(dt).strftime("%Y-%m-%d"):i for i,dt in enumerate(d.date)}
     feature_cols=[col for col in d.columns if pd.api.types.is_numeric_dtype(d[col]) or pd.api.types.is_bool_dtype(d[col])]
     enriched=[]
     for trade in sim["trades"]:
      entry_signal_date=str(trade.get("entry_signal_date") or trade["entry_date"])
      exit_signal_date=str(trade.get("exit_signal_date") or trade["exit_date"])
      entry_i=date_index.get(entry_signal_date);exit_i=date_index.get(exit_signal_date)
      if entry_i is None or exit_i is None:
       raise RuntimeError(f"{symbol}: trade signal date missing from causal feature table: {trade}")
      def snapshot(idx):
       result={}
       for col in feature_cols:
        value=d.iloc[idx][col]
        if pd.isna(value):result[col]=None
        elif isinstance(value,(np.bool_,bool)):result[col]=bool(value)
        elif isinstance(value,(np.integer,np.floating,int,float)):result[col]=float(value)
        else:result[col]=str(value)
       return result
      record={**trade,"symbol":symbol,"stock_id":stock_id_map[symbol],"market":market_map[symbol],
       "name":name_map.get(symbol,NAMES.get(stock_id_map[symbol],symbol)),
       "slippage_bps":slip,"entry_features_json":json.dumps(snapshot(entry_i),ensure_ascii=False,separators=(",",":")),
       "exit_features_json":json.dumps(snapshot(exit_i),ensure_ascii=False,separators=(",",":"))}
      enriched.append(record);trade_diagnostics.append(record)
     pd.DataFrame(enriched).to_csv(out/f"trades_{symbol}_10bps.csv",index=False)
     pd.DataFrame(sim["equity_curve"]).to_csv(out/f"equity_{symbol}_10bps.csv",index=False)
     pd.DataFrame(annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL)).to_csv(out/f"annual_{symbol}_10bps.csv",index=False)
   prepared[symbol]={"d":d,"low":low,"high":high,"risk":risk,"hold":hold,"limits":limits,"dividends":cashdiv,"share_adjustments":share_adjustments}
   print(f"[DONE] {symbol} {name_map.get(symbol,NAMES.get(symbol,symbol))}",flush=True)
  except Exception as e:
   blocked.append({"symbol":symbol,"name":name_map.get(symbol,NAMES.get(symbol,symbol)),"reason":f"{type(e).__name__}: {e}"})
   print(f"[BLOCKED] {symbol}: {e}",flush=True)
 pd.DataFrame(annual_rows).to_csv(out/"annual_metrics_10bps_by_symbol.csv",index=False)
 pd.DataFrame(trade_diagnostics).to_csv(out/"trade_feature_diagnostics_10bps.csv",index=False)
 pd.DataFrame(filter_replay_rows).to_csv(out/"entry_filter_oos_replay_2022plus.csv",index=False)
 shared_portfolio=None;shared_portfolios={}
 if use_universe_filter and prepared:
  for slip in SLIPPAGE_BPS:
   portfolio=run_shared_capital_portfolio(prepared,uni,slippage_bps=slip,initial_capital=INITIAL_CAPITAL)
   pd.DataFrame(portfolio.get("trades",[])).to_csv(out/f"shared_capital_trades_{slip}bps.csv",index=False)
   pd.DataFrame(portfolio.get("equity_curve",[])).to_csv(out/f"shared_capital_equity_{slip}bps.csv",index=False)
   if portfolio.get("equity_curve"):
    pd.DataFrame(annual_metrics(portfolio["equity_curve"],portfolio["trades"],INITIAL_CAPITAL)).to_csv(out/f"shared_capital_annual_{slip}bps.csv",index=False)
   shared_portfolios[f"{slip}bps"]={k:v for k,v in portfolio.items() if k not in {"trades","equity_curve"}}
  shared_portfolio=shared_portfolios.get("10bps")
  # ATR trailing-stop overlay sensitivity on the same point-in-time top-ten
  # universe and single shared account; baseline portfolio remains untouched.
  atr_portfolio_rows=[]
  for slip in SLIPPAGE_BPS:
   for activation_pct in (0.05,0.10,0.15,0.20):
    for atr_multiple in (2.5,3.0,3.5):
     overlay=run_shared_capital_portfolio(prepared,uni,slippage_bps=slip,
       initial_capital=INITIAL_CAPITAL,trailing_atr_multiple=atr_multiple,
       atr_activation_profit_pct=activation_pct)
     metric={k:v for k,v in overlay.items() if k not in {"trades","equity_curve"}}
     atr_portfolio_rows.append({"variant":"shared_capital_profit_gated_atr_trailing_stop",
       "slippage_bps":slip,**metric})
   for peak_dd in (0.05,0.08,0.10,0.12,0.15):
    overlay=run_shared_capital_portfolio(prepared,uni,slippage_bps=slip,
      initial_capital=INITIAL_CAPITAL,atr_activation_profit_pct=0.10,
      trailing_peak_drawdown_pct=peak_dd)
    metric={k:v for k,v in overlay.items() if k not in {"trades","equity_curve"}}
    atr_portfolio_rows.append({"variant":"shared_capital_profit_gated_peak_drawdown_stop",
      "slippage_bps":slip,**metric})
    if slip==10 and peak_dd==0.10:
     pd.DataFrame(overlay.get("trades",[])).to_csv(out/"shared_capital_peak10_activation10_trades_10bps.csv",index=False)
     pd.DataFrame(overlay.get("equity_curve",[])).to_csv(out/"shared_capital_peak10_activation10_equity_10bps.csv",index=False)
  pd.DataFrame(atr_portfolio_rows).to_csv(out/"shared_capital_exit_overlay_metrics.csv",index=False)
 target_assessment=None
 if shared_portfolio:
  target_assessment={
   "cagr_50pct_pass":bool(shared_portfolio.get("cagr_mark_to_market") is not None and shared_portfolio["cagr_mark_to_market"]>=.50),
   "win_rate_86_9565pct_pass":bool(shared_portfolio.get("win_rate_completed_only") is not None and shared_portfolio["win_rate_completed_only"]>=.8695652173913043),
   "max_drawdown_25pct_pass":bool(shared_portfolio.get("max_drawdown") is not None and shared_portfolio["max_drawdown"]>=-.25),
   "universe_coverage_99pct_pass":bool(shared_portfolio.get("complete_universe_day_ratio",0)>=.99),
   "tsmc_baseline_reproduction_pass":bool(tsmc_reproduction and tsmc_reproduction.get("matches_benchmark_within_tolerance"))}
  target_assessment["all_targets_and_coverage_pass"]=all(target_assessment.values())
 data_provenance={}
 if universe_path.exists():
  mp=universe_path.parent/"manifest.json"
  if mp.exists():data_provenance["daily_top10_universe"]=json.loads(mp.read_text(encoding="utf-8"))
 for label,filename in (("twse_twt49u","manifest.json"),("twse_tpex_special_events","special_events_manifest.json")):
  mp=Path(a.actions)/filename
  if mp.exists():data_provenance[label]=json.loads(mp.read_text(encoding="utf-8"))
 result={"classification":"FIXED_TSMC_RULE_TRANSFER_AUDIT_DAILY_TOP10_VOLUME_UNIVERSE",
  "period":"2015 onward through last official raw bar","symbols_expected":len(symbols),
  "symbols_completed":len(set(x["symbol"] for x in rows)),"blocked":blocked,
  "costs":{"commission_per_side":.001425,"minimum_commission_ntd":20,"sell_tax":.003,"slippage_bps":list(SLIPPAGE_BPS)},
  "target":{"cagr_min":.50,"win_rate_min":.8695652173913043,"max_drawdown_floor":-.25},
  "status":"COMPLETE" if len(set(x["symbol"] for x in rows))==len(symbols) and not blocked else "BLOCKED_OR_PARTIAL",
  "data_provenance":data_provenance,"cross_validation":{"parameters_frozen_from_TSMC":True,"no_reoptimization_on_transfer_universe":True,"annual_performance_decomposition":"annual_metrics_10bps_by_symbol.csv","interpretation":"Year-by-year stability report using unchanged frozen rules; this is not a retuned annual OOF strategy."},"metrics":rows,"entry_filter_oos_replay_2022plus":filter_replay_rows,"entry_filter_threshold_provenance":{"training_period":"2015-01-01 through 2021-12-31","test_period":"2022-01-01 through final official raw bar","sma50_slope_10_min":0.006062,"distance_sma_200_min":0.073873,"thresholds_selected_from_training_trades_only":True},"tsmc_2330_universe_free_reproduction":tsmc_reproduction,"shared_capital_portfolio":shared_portfolio,"shared_capital_portfolios_by_slippage":shared_portfolios,"target_assessment":target_assessment,"rules":RULES,
  "notes":["No per-symbol parameter tuning; frozen TSMC rule family only.",
   "Entry eligibility uses the official same-day closing traded-share top ten; signals are known after close and orders execute no earlier than the next session.",
   "Raw official TWSE MI_INDEX and TPEx afterTrading/otc OHLC used for fills; exchange-specific corporate-action back-adjustment used for signal indicators.",
   "TWSE TWT49U covers ex-rights/ex-dividend events from 2010 and TPEx exDailyQ from 2014 (the required 200-session warmup window); TWSE TWTAUU/TWTB8U and TPEx revivt cover special resumption events from 2011/2013; performance measurement begins 2015.",
   "Any missing action data, missing limit reference, or OHLC-limit conflict blocks that symbol. The shared-capital portfolio also skips new entries when any daily top-ten constituent is unaudited.",
  "Post-2015 stock dividends are modeled as share multipliers derived from official prior close, cash dividend, and ex-right reference price; fractional share entitlements are rounded down conservatively. Post-2015 loss-recovery capital reductions and par-value changes use official reference-price-derived share ratios. Documented Yageo cash-refund reductions use issuer-notice share ratios and refunds, reconciled to TWSE reference prices; unsupported cash-refund events remain blocked.",
   "Single-symbol results are independent NT$100,000 per-stock transfer tests; shared_capital_portfolio is a separate one-position shared-NT$100,000 account."]}
 (out/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
 pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
 print(json.dumps({k:v for k,v in result.items() if k!="metrics"},ensure_ascii=False,indent=2,default=float))
if __name__=="__main__":main()
