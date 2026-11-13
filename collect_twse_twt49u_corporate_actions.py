"""Collect official TWSE TWT49U historical ex-rights/ex-dividend results.

Public TWSE RWD endpoint, no paid token. TWT49U has historical result rows
from 2003-05-05; the parser is header-driven because legacy schema differs.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json
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
  y=int(s[:4]);return f"{y:04d}-{int(s[4:6]):02d}-{int(s[6:]):02d}"
 raise ValueError(f"unrecognized TWT49U date {v!r}")

def num(v):
 s=str(v).strip().replace(",","")
 if s in {"","--","-","X","nan","None"}:return None
 try:return float(s)
 except:return None

def pick(headers,*terms):
 for i,h in enumerate(headers):
  hh=str(h).replace(" ","").strip()
  if any(t in hh for t in terms):return i
 raise RuntimeError(f"TWT49U missing expected field among {terms}; headers={headers}")

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--start-date",default="2010-01-04")
 p.add_argument("--end-date",default="2026-11-12")
 p.add_argument("--output",default="data/top10_actions_twt49u")
 a=p.parse_args()
 params={"response":"json","strDate":pd.Timestamp(a.start_date).strftime("%Y%m%d"),"endDate":pd.Timestamp(a.end_date).strftime("%Y%m%d")}
 r=requests.get(URL,params=params,timeout=(15,120),headers={"User-Agent":"Mozilla/5.0"})
 r.raise_for_status()
 payload=r.json()
 if not isinstance(payload,dict) or not isinstance(payload.get("fields"),list) or not isinstance(payload.get("data"),list):
  raise RuntimeError(f"Invalid official TWT49U JSON response: {str(payload)[:1000]}")
 headers=[str(x).strip() for x in payload["fields"]]
 table=[headers]+payload["data"]
 date_i=pick(headers,"除權息日期","除權日期","資料日期","日期")
 sym_i=pick(headers,"股票代號","證券代號")
 name_i=pick(headers,"股票名稱","證券名稱")
 prev_i=pick(headers,"前一日收盤價","除權息前收盤價")
 ref_i=pick(headers,"除權息參考價","除權參考價","參考價")
 up_i=pick(headers,"漲停價","漲停價格");down_i=pick(headers,"跌停價","跌停價格")
 total_i=pick(headers,"權值+息值","權息","息值","現金股利")
 type_i=next((i for i,h in enumerate(headers) if "權/息" in h or "權息" in h),None)
 rows=[]
 for arow in table[1:]:
  if len(arow)<=max(date_i,sym_i,name_i,prev_i,ref_i,up_i,down_i,total_i):continue
  sym=str(arow[sym_i]).strip()
  if sym not in SYMBOLS:continue
  try:day=rocdate(arow[date_i])
  except:continue
  if not a.start_date<=day<=a.end_date:continue
  kind=str(arow[type_i]).strip() if type_i is not None and len(arow)>type_i else ""
  total=num(arow[total_i]) or 0.0
  cash=total if kind=="息" else (0.0 if kind=="權" else None)
  stock=total if kind=="權" else (0.0 if kind=="息" else None)
  vals={"symbol":sym,"stock_id":sym,"name":str(arow[name_i]).strip() or SYMBOLS[sym],
   "date":day,"previous_close":num(arow[prev_i]),"reference_price":num(arow[ref_i]),
   "limit_up":num(arow[up_i]),"limit_down":num(arow[down_i]),
   "cash_dividend":cash,"stock_dividend":stock,"action_type":kind,
   "cash_dividend_status":"resolved" if kind in {"息","權"} else "unresolved_mixed_action",
   "source":"TWSE TWT49U"}
  if vals["reference_price"] is None or vals["limit_up"] is None or vals["limit_down"] is None:
   raise RuntimeError(f"Missing official reference/limits for {sym} {day}: {vals}")
  if vals["previous_close"] is None or vals["previous_close"]<=0 or vals["reference_price"]<=0:
   raise RuntimeError(f"Invalid reference prices: {vals}")
  vals["adjustment_ratio"]=vals["reference_price"]/vals["previous_close"]
  rows.append(vals)
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
 for sym in SYMBOLS:
  sub=frame[frame.symbol==sym].copy()
  sub.to_csv(out/f"{sym}.csv",index=False)
 manifest={"source":"TWSE official RWD TWT49U","url":r.url,"collected_at_taipei":datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
  "requested_period":[a.start_date,a.end_date],"rows":len(frame),"symbols_with_events":int(frame.symbol.nunique()),
  "symbols_without_events":[s for s in SYMBOLS if s not in set(frame.symbol)],
  "complete":True,"schema":headers,"sha256":hashlib.sha256((out/"twt49u_all.csv").read_bytes()).hexdigest(),
  "limitations":["TWT49U covers ex-rights/ex-dividend calculation results; it does not replace separate capital-reduction/resumption event feeds.",
   "Any raw OHLC outside reconstructed daily limits or any special corporate-action marker must block affected symbol/date."]}
 (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps({k:v for k,v in manifest.items() if k!="schema"},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
