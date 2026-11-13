"""Collect official TWSE TWT49U historical ex-rights/ex-dividend results.

Public TWSE endpoint, no paid token. TWT49U has historical results since
2003-05-05. Requests are split by calendar year to avoid large-range timeouts.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import requests

URL="https://www.twse.com.tw/exchangeReport/TWT49U"
SYMBOLS={"2330":"台積電","2454":"聯發科","1303":"南亞","2327":"國巨","2408":"南亞科","2308":"台達電","3037":"欣興","3008":"大立光","2383":"台光電","2303":"聯電"}

def rocdate(v):
 s=str(v).strip().replace("年","/").replace("月","/").replace("日","")
 p=s.split("/")
 if len(p)==3:
  y=int(p[0]);y+=1911 if y<1911 else 0
  return f"{y:04d}-{int(p[1]):02d}-{int(p[2]):02d}"
 if len(s)==8 and s.isdigit():
  y=int(s[:4]);return f"{int(s[:4]):04d}-{int(s[4:6]):02d}-{int(s[6:]):02d}"
 raise ValueError(f"unrecognized TWT49U date {v!r}")

def num(v):
 s=str(v).strip().replace(",","")
 if s in {"","--","-","X","nan","None"}:return None
 try:return float(s)
 except:return None

def index(headers,*terms,required=True):
 for i,h in enumerate(headers):
  hh=str(h).replace(" ","").strip()
  if any(t in hh for t in terms):return i
 if required:raise RuntimeError(f"TWT49U missing field {terms}; schema={headers}")
 return None

def fetch_year(session,year,start_date,end_date):
 lo=pd.Timestamp(start_date);hi=pd.Timestamp(end_date)
 candidates=[
  {"response":"json","strDate":lo.strftime("%Y%m%d"),"endDate":hi.strftime("%Y%m%d")},
  {"response":"json","strDate":lo.strftime("%Y%m%d"),"endDate":hi.strftime("%Y%m%d"),"lang":"zh"},
  {"response":"json","strDate":f"{lo.year-1911:03d}{lo.strftime('%m%d')}","endDate":f"{hi.year-1911:03d}{hi.strftime('%m%d')}"},
  {"response":"json","startDate":lo.strftime("%Y%m%d"),"endDate":hi.strftime("%Y%m%d")},
 ]
 errors=[]
 for params in candidates:
  try:
   r=session.get(URL,params=params,timeout=(15,150),headers={"User-Agent":"Mozilla/5.0"})
   r.raise_for_status();payload=r.json()
   if not isinstance(payload,dict) or not isinstance(payload.get("fields"),list) or not isinstance(payload.get("data"),list):
    errors.append({"url":r.url,"payload":str(payload)[:300]});continue
   h=[str(x).strip() for x in payload["fields"]]
   di=next((i for i,x in enumerate(h) if any(t in x for t in ("資料日期","除權息日期","除權日期","日期"))),None)
   if di is None:errors.append({"url":r.url,"fields":h});continue
   sample=[]
   for row in payload["data"][:5]:
    try:sample.append(rocdate(row[di]))
    except:pass
   if not sample or not any(start_date<=d<=end_date for d in sample):
    errors.append({"url":r.url,"stat":payload.get("stat"),"sample_dates":sample,"row_count":len(payload["data"])});continue
   return payload,r.url
  except Exception as e:errors.append({"params":params,"error":str(e)})
 raise RuntimeError(f"TWT49U {year} failed; attempts={errors}")

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--start-date",default="2010-01-04")
 p.add_argument("--end-date",default="2026-11-12")
 p.add_argument("--output",default="data/top10_actions_twt49u")
 a=p.parse_args()
 start=pd.Timestamp(a.start_date);end=pd.Timestamp(a.end_date)
 if start>end:raise ValueError("start date after end date")
 s=requests.Session();rows=[];schemas=[];urls=[]
 for year in range(start.year,end.year+1):
  lo=max(start,pd.Timestamp(year=year,month=1,day=1)).strftime("%Y-%m-%d")
  hi=min(end,pd.Timestamp(year=year,month=12,day=31)).strftime("%Y-%m-%d")
  payload,url=fetch_year(s,year,lo,hi);urls.append(url)
  h=[str(x).strip() for x in payload["fields"]];schemas.append({"year":year,"fields":h})
  date_i=index(h,"資料日期","除權息日期","除權日期","日期")
  sym_i=index(h,"股票代號","證券代號")
  name_i=index(h,"股票名稱","證券名稱")
  prev_i=index(h,"除權息前收盤價","前一日收盤價")
  ref_i=index(h,"除權息參考價","除權參考價","參考價")
  up_i=index(h,"漲停價格","漲停價");down_i=index(h,"跌停價格","跌停價")
  type_i=index(h,"權/息","權息",required=False)
  cash_i=index(h,"息值","現金股利",required=False)
  stock_i=index(h,"權值","股票股利",required=False)
  total_i=index(h,"權值+息值","權息",required=False)
  if cash_i is None and total_i is None:raise RuntimeError(f"{year}: neither cash dividend nor combined amount field")
  for raw in payload["data"]:
   if len(raw)<=max(x for x in [date_i,sym_i,name_i,prev_i,ref_i,up_i,down_i] if x is not None):continue
   sym=str(raw[sym_i]).strip()
   try:day=rocdate(raw[date_i])
   except:continue
   if not a.start_date<=day<=a.end_date:continue
   kind=str(raw[type_i]).strip() if type_i is not None and len(raw)>type_i else ""
   if cash_i is not None:
    cash=num(raw[cash_i]) or 0.0
    stock=(num(raw[stock_i]) or 0.0) if stock_i is not None else None
    cash_status="resolved" if stock is not None else "unresolved_stock_component"
   else:
    total=num(raw[total_i]) or 0.0
    cash=total if kind=="息" else (0.0 if kind=="權" else None)
    stock=total if kind=="權" else (0.0 if kind=="息" else None)
    cash_status="resolved" if kind in {"息","權"} else "unresolved_mixed_action"
   vals={"symbol":sym,"stock_id":sym,"name":str(raw[name_i]).strip() or SYMBOLS.get(sym,sym),
    "date":day,"previous_close":num(raw[prev_i]),"reference_price":num(raw[ref_i]),
    "limit_up":num(raw[up_i]),"limit_down":num(raw[down_i]),
    "cash_dividend":cash,"stock_dividend":stock,"action_type":kind,
    "cash_dividend_status":cash_status,"source":"TWSE TWT49U"}
   if any(vals[k] is None for k in ("previous_close","reference_price","limit_up","limit_down")):
    raise RuntimeError(f"Missing official event reference/limits: {vals}")
   if vals["previous_close"]<=0 or vals["reference_price"]<=0 or vals["limit_up"]<=0 or vals["limit_down"]<=0:
    raise RuntimeError(f"Invalid official event values: {vals}")
   vals["adjustment_ratio"]=vals["reference_price"]/vals["previous_close"]
   rows.append(vals)
  print(f"[YEAR DONE] {year} rows_for_basket={sum(1 for x in rows if x['date'].startswith(str(year)))}",flush=True)
 frame=pd.DataFrame(rows)
 if frame.empty:raise RuntimeError("No TWT49U rows for requested ten symbols")
 dup=frame.groupby(["symbol","date"]).agg({"reference_price":"nunique","limit_up":"nunique","limit_down":"nunique"})
 bad=dup[(dup>1).any(axis=1)]
 if len(bad):raise RuntimeError(f"Conflicting TWT49U duplicate symbol/date rows: {bad.head().to_dict()}")
 frame=frame.drop_duplicates(["symbol","date"]).sort_values(["symbol","date"])
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 frame.to_csv(out/"twt49u_all.csv",index=False)
 events=frame[["symbol","date","adjustment_ratio","source"]].copy()
 events["sources"]="TWSE TWT49U";events["source_count"]=1;events["conflict"]=False
 events.to_csv(out/"events.csv",index=False)
 for sym in sorted(frame.symbol.astype(str).unique()):frame[frame.symbol.astype(str)==sym].to_csv(out/f"{sym}.csv",index=False)
 manifest={"source":"TWSE official TWT49U","url":URL,"queried_urls":urls,
  "collected_at_taipei":datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
  "requested_period":[a.start_date,a.end_date],"rows":len(frame),
  "symbols_with_events":int(frame.symbol.nunique()),
  "symbols_without_events":[s for s in SYMBOLS if s not in set(frame.symbol)],
  "unresolved_cash_dividend_events":int((frame.cash_dividend_status!="resolved").sum()),
  "complete":True,"schemas":schemas,
  "sha256":hashlib.sha256((out/"twt49u_all.csv").read_bytes()).hexdigest(),
  "limitations":["TWT49U excludes certain cash capital-reduction and split-reduction events; separate TWTAUU/TWTB8U event feeds are required.",
   "Mixed combined rights/dividend rows without separately reported cash amount are explicitly unresolved and block exact cash-dividend accounting.",
   "Any raw OHLC outside reconstructed daily limits or any special corporate-action marker must block affected symbol/date."]}
 (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps({k:v for k,v in manifest.items() if k not in {"schemas","queried_urls"}},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
