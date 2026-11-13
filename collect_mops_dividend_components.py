"""Collect official MOPS dividend components and reconcile them to TWSE TWT49U.

MOPS t108sb27 is queried by ROC event year for listed companies. Cash and stock
components are matched by the exact ex-dividend/ex-rights date, then accepted
only when the computed reference price matches official TWSE TWT49U.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import re
import time
from urllib.parse import urlencode
from pathlib import Path
import pandas as pd
import requests

URL="https://mopsov.twse.com.tw/mops/web/ajax_t108sb27"

def rocdate(value):
    if value is None or pd.isna(value): return None
    s=str(value).strip()
    if not s: return None
    m=re.search(r"(\d{2,3})[./-](\d{1,2})[./-](\d{1,2})",s)
    if not m: return None
    year=int(m.group(1))+1911
    return f"{year:04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

def number(value):
    if value is None or pd.isna(value): return 0.0
    m=re.search(r"[-+]?\d+(?:\.\d+)?",str(value).replace(",",""))
    return float(m.group(0)) if m else 0.0

def par_value(value):
    x=number(value)
    return x if x>0 else None

def last_level(column):
    if isinstance(column,tuple): return str(column[-1]).strip()
    return str(column).strip()

def fetch_year(session, roc_year, symbols, raw_dir, raw_cache_dir=None, retries=3):
    params={
      "step":"1","firstin":"true","off":"1","TYPEK":"sii",
      "co_id_1":"","co_id_2":"","year":str(roc_year),"month":"",
      "b_date":"","e_date":"","type":"1","rulesubmit2":"搜尋"
    }
    last_error=None
    cache_path=(Path(raw_cache_dir)/f"mops_t108sb27_sii_{roc_year}.html") if raw_cache_dir else None
    cached_html=cache_path.read_text(encoding="utf-8") if cache_path and cache_path.exists() else None
    for attempt in range(retries):
        try:
            if cached_html is not None:
                html=cached_html
                source_url=URL+"?"+urlencode(params)
            else:
                response=session.get(URL,params=params,timeout=(10,90))
                response.raise_for_status()
                response.encoding="utf-8"
                html=response.text
                source_url=response.url
            if "FOR SECURITY REASONS" in html or "因為安全性考量" in html:
                raise RuntimeError("MOPS security interstitial")
            tables=pd.read_html(io.StringIO(html), encoding="utf-8")
            frame=next((t for t in tables if any(last_level(c)=="公司代號" for c in t.columns)),None)
            if frame is None: raise RuntimeError(f"MOPS t108sb27 data table missing for ROC year {roc_year}")
            raw_dir.mkdir(parents=True,exist_ok=True)
            (raw_dir/f"mops_t108sb27_sii_{roc_year}.html").write_text(html,encoding="utf-8")
            def col(label,required=True):
                matches=[c for c in frame.columns if last_level(c)==label]
                if matches:return matches[0]
                if required:raise RuntimeError(f"MOPS year {roc_year} missing column {label}; got {[last_level(c) for c in frame.columns]}")
                return None
            code_c=col("公司代號")
            right_c=col("除權交易日",False)
            div_c=col("除息交易日",False)
            rights_ratio_c=col("現金增資認股比率(%)",False)
            rights_price_c=col("現金增資認購價(元/股)",False)
            rights_shares_c=col("現金增資總股數(股)",False)
            stock_c1=col("盈餘轉增資配股(元/股)",False)
            stock_c2=col("法定盈餘公積、資本公積轉增資配股(元/股)",False)
            cash_c1=col("盈餘分配之股東現金股利(元/股)",False)
            cash_c2=col("法定盈餘公積、資本公積發放之現金(元/股)",False)
            par_c=col("普通股每股面額",False)
            year_c=col("股利所屬年度",False)
            out=[]
            for _,row in frame.iterrows():
                symbol=str(row[code_c]).strip()
                if symbol.endswith(".0"):symbol=symbol[:-2]
                if symbol not in symbols:continue
                stock=number(row[stock_c1] if stock_c1 else 0)+number(row[stock_c2] if stock_c2 else 0)
                cash=number(row[cash_c1] if cash_c1 else 0)+number(row[cash_c2] if cash_c2 else 0)
                rights_pct=number(row[rights_ratio_c]) if rights_ratio_c else 0.0
                rights_price=number(row[rights_price_c]) if rights_price_c else 0.0
                rights_shares=number(row[rights_shares_c]) if rights_shares_c else 0.0
                par=par_value(row[par_c]) if par_c else None
                ex_right=rocdate(row[right_c]) if right_c else None
                ex_div=rocdate(row[div_c]) if div_c else None
                dividend_year=str(row[year_c]) if year_c else ""
                common={"symbol":symbol,"par_value_per_share":par,"dividend_year":dividend_year,
                  "mops_year":roc_year,"ex_right_date":ex_right,"ex_dividend_date":ex_div,
                  "rights_issue_ratio_pct":rights_pct,"rights_subscription_price":rights_price if rights_pct>0 else None,
                  "rights_issue_total_shares":rights_shares,"source_url":source_url}
                if ex_right and ex_div and ex_right==ex_div:
                    if cash>0 or stock>0 or rights_pct>0:
                        out.append({**common,"event_date":ex_right,"cash_dividend_per_share":cash,
                          "stock_dividend_per_share":stock})
                else:
                    if ex_right and (stock>0 or rights_pct>0):
                        out.append({**common,"event_date":ex_right,"cash_dividend_per_share":0.0,
                          "stock_dividend_per_share":stock})
                    if ex_div and cash>0:
                        out.append({**common,"event_date":ex_div,"cash_dividend_per_share":cash,
                          "stock_dividend_per_share":0.0,"rights_issue_ratio_pct":0.0,
                          "rights_subscription_price":None,"rights_issue_total_shares":0.0})
            return out,{"roc_year":roc_year,"url":source_url,"table_rows":len(frame),"matched_symbols":int(frame[code_c].astype(str).isin(symbols).sum()),"component_event_rows":len(out)}
        except Exception as exc:
            last_error=exc
            if attempt+1<retries:time.sleep(2**attempt+1)
    raise RuntimeError(f"MOPS t108sb27 ROC year {roc_year}: {last_error}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--actions-dir",required=True)
    p.add_argument("--overrides",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--raw-dir",default=None,help="Reuse previously downloaded official MOPS HTML pages")
    p.add_argument("--pause",type=float,default=0.5)
    a=p.parse_args()
    root=Path(a.actions_dir)
    twt=pd.read_csv(root/"twt49u_all.csv",dtype={"symbol":str})
    twt["date"]=pd.to_datetime(twt.date,errors="coerce").dt.strftime("%Y-%m-%d")
    if "cash_dividend_status" not in twt:raise RuntimeError("TWT49U file missing cash_dividend_status")
    unresolved=twt[(twt.cash_dividend_status.astype(str)!="resolved") & twt.date.notna() & (twt.date>="2015-01-01")].copy()
    if unresolved.empty:raise RuntimeError("No unresolved 2015+ TWT49U events found")
    unresolved["event_year"]=pd.to_datetime(unresolved.date).dt.year
    symbols=set(unresolved.symbol.astype(str).str.strip())
    years=sorted({int(y)-1911 for y in unresolved.event_year})
    raw_dir=Path(a.output_dir)/"raw_html"
    session=requests.Session()
    session.headers.update({
      "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
      "Referer":"https://mopsov.twse.com.tw/mops/web/index",
      "Origin":"https://mopsov.twse.com.tw",
    })
    components=[];year_meta=[]
    for i,year in enumerate(years,1):
        rows,meta=fetch_year(session,year,symbols,raw_dir,raw_cache_dir=a.raw_dir)
        components.extend(rows);year_meta.append(meta)
        print(f"MOPS ROC {year}: table_rows={meta['table_rows']} components={len(rows)}",flush=True)
        if i<len(years):time.sleep(a.pause)
    comp=pd.DataFrame(components)
    if comp.empty:raise RuntimeError("MOPS returned no matching dividend component rows")
    # Collapse multiple MOPS distributions for the same issuer and ex-date.
    grouped=[]
    for (symbol,event_date),g in comp.groupby(["symbol","event_date"],sort=True):
        pars=sorted({round(float(x),6) for x in g.par_value_per_share.dropna() if float(x)>0})
        rights_pct=float(pd.to_numeric(g.rights_issue_ratio_pct,errors="coerce").fillna(0).sum())
        rights_weight=pd.to_numeric(g.rights_issue_ratio_pct,errors="coerce").fillna(0)
        rights_prices=pd.to_numeric(g.rights_subscription_price,errors="coerce")
        weighted_den=float(rights_weight.sum())
        rights_price=(float((rights_weight*rights_prices.fillna(0)).sum()/weighted_den)
                      if weighted_den>0 and rights_prices[rights_weight>0].notna().all() else None)
        if len(pars)!=1:
            grouped.append({"symbol":symbol,"event_date":event_date,"cash_dividend_per_share":float(g.cash_dividend_per_share.sum()),
              "stock_dividend_per_share":float(g.stock_dividend_per_share.sum()),"par_value_per_share":None,
              "rights_issue_ratio_pct":rights_pct,"rights_subscription_price":rights_price,
              "rights_issue_total_shares":float(pd.to_numeric(g.rights_issue_total_shares,errors="coerce").fillna(0).sum()),
              "mops_rows":len(g),"par_value_conflict":True,"source_url":";".join(sorted(set(g.source_url.astype(str))))})
        else:
            grouped.append({"symbol":symbol,"event_date":event_date,"cash_dividend_per_share":float(g.cash_dividend_per_share.sum()),
              "stock_dividend_per_share":float(g.stock_dividend_per_share.sum()),"par_value_per_share":pars[0],
              "rights_issue_ratio_pct":rights_pct,"rights_subscription_price":rights_price,
              "rights_issue_total_shares":float(pd.to_numeric(g.rights_issue_total_shares,errors="coerce").fillna(0).sum()),
              "mops_rows":len(g),"par_value_conflict":False,"source_url":";".join(sorted(set(g.source_url.astype(str))))})
    comp=pd.DataFrame(grouped)
    comp.to_csv(Path(a.output_dir)/"mops_dividend_components.csv",index=False)
    existing=pd.read_csv(a.overrides,dtype={"symbol":str}) if Path(a.overrides).exists() else pd.DataFrame(columns=[
      "symbol","event_date","cash_dividend_per_share","stock_dividend_per_share","par_value_per_share","source_type","source_url","source_note"])
    existing["symbol"]=existing.symbol.astype(str).str.strip()
    existing["event_date"]=pd.to_datetime(existing.event_date,errors="coerce").dt.strftime("%Y-%m-%d")
    additions=[];unmatched=[]
    lookup={(str(r.symbol),str(r.event_date)):r for r in comp.itertuples(index=False)}
    for r in unresolved.itertuples(index=False):
        key=(str(r.symbol).strip(),str(r.date))
        candidate=lookup.get(key)
        if candidate is None:
            unmatched.append({"symbol":key[0],"event_date":key[1],"reason":"no exact MOPS ex-right/ex-dividend date match"})
            continue
        if candidate.par_value_conflict or candidate.par_value_per_share is None or candidate.par_value_per_share<=0:
            unmatched.append({"symbol":key[0],"event_date":key[1],"reason":"missing or conflicting MOPS par value"})
            continue
        cash=float(candidate.cash_dividend_per_share)
        stock=float(candidate.stock_dividend_per_share)
        par=float(candidate.par_value_per_share)
        prev=float(r.previous_close);ref=float(r.reference_price)
        if not math.isfinite(prev) or not math.isfinite(ref) or prev<=0 or ref<=0:
            unmatched.append({"symbol":key[0],"event_date":key[1],"reason":"missing or invalid official TWT49U previous close/reference price"})
            continue
        rights_pct=float(candidate.rights_issue_ratio_pct or 0.0)
        rights_ratio=rights_pct/100.0
        rights_price=float(candidate.rights_subscription_price) if candidate.rights_subscription_price is not None and pd.notna(candidate.rights_subscription_price) else 0.0
        if rights_ratio>0 and rights_price<=0:
            unmatched.append({"symbol":key[0],"event_date":key[1],"reason":"MOPS cash-rights issue ratio has no valid subscription price"})
            continue
        expected=(prev-cash+rights_ratio*rights_price)/(1.0+stock/par+rights_ratio)
        diff=abs(expected-ref)
        if not math.isfinite(expected) or expected<=0 or not math.isfinite(diff) or diff>0.03:
            unmatched.append({"symbol":key[0],"event_date":key[1],"reason":"MOPS components do not reconcile to official TWT49U reference price",
              "previous_close":prev,"official_reference_price":ref,"expected_reference_price":expected,"absolute_difference":diff,
              "cash_dividend":cash,"stock_dividend":stock,"par_value":par})
            continue
        additions.append({"symbol":key[0],"event_date":key[1],"cash_dividend_per_share":cash,
          "stock_dividend_per_share":stock,"par_value_per_share":par,
          "rights_issue_ratio_pct":rights_pct,"rights_subscription_price":rights_price if rights_ratio>0 else None,
          "rights_issue_total_shares":float(candidate.rights_issue_total_shares or 0.0),
          "source_type":"MOPS_official_t108sb27_reconciled_to_TWSE_TWT49U",
          "source_url":candidate.source_url,
          "source_note":f"MOPS official t108sb27 {candidate.mops_rows} row(s); cash={cash}; stock={stock}; par={par}; rights_ratio_pct={rights_pct}; rights_subscription_price={rights_price}; TWSE previous_close={prev}; official_reference={ref}; expected_reference={expected:.6f}; abs_diff={diff:.6f}"})
    additions_df=pd.DataFrame(additions)
    if not additions_df.empty:
        existing=existing[~existing.set_index(["symbol","event_date"]).index.isin(additions_df.set_index(["symbol","event_date"]).index)]
        existing=pd.concat([existing,additions_df],ignore_index=True)
    existing=existing.drop_duplicates(["symbol","event_date"],keep="last").sort_values(["symbol","event_date"])
    Path(a.overrides).parent.mkdir(parents=True,exist_ok=True)
    existing.to_csv(a.overrides,index=False)
    unresolved_df=pd.DataFrame(unmatched)
    unresolved_df.to_csv(Path(a.output_dir)/"mops_unresolved.csv",index=False)
    manifest={
      "source":"Official MOPS t108sb27",
      "query_type":"listed companies (TYPEK=sii)",
      "roc_years_queried":years,
      "twt49u_unresolved_events":int(len(unresolved)),
      "mops_component_event_rows":int(len(comp)),
      "new_overrides_reconciled":int(len(additions)),
      "existing_override_rows":int(len(existing)),
      "unresolved_after_mops":int(len(unmatched)),
      "mops_year_metadata":year_meta,
      "unresolved_examples":unmatched[:100],
      "overrides_sha256":hashlib.sha256(Path(a.overrides).read_bytes()).hexdigest(),
      "complete":True,
      "acceptance_rule":"A MOPS component row is accepted only when exact issuer/date match and cash/stock/par/cash-rights formula reproduces the official TWSE TWT49U reference price within NT$0.03.",
      "rights_issue_terms_included":True,
      "raw_html_reused":bool(a.raw_dir)
    }
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
