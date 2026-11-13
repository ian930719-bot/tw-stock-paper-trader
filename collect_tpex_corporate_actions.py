"""Collect official TPEx ex-rights/dividend and capital-reduction events."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd,requests
from bs4 import BeautifulSoup

BASE="https://www.tpex.org.tw/www/zh-tw/"
def clean(x):return re.sub(r"\s+","",BeautifulSoup(str(x if x is not None else ""),"html.parser").get_text(" ",strip=True))
def num(x):
 s=BeautifulSoup(str(x if x is not None else ""),"html.parser").get_text(" ",strip=True).replace(",","").strip()
 if s in {"","--","-","X","nan","None","除權息"}:return None
 try:return float(s)
 except:return None
def rocdate(x):
 s=str(x).strip()
 digits=re.sub(r"\D","",s)
 if len(digits)==7:y=int(digits[:3])+1911;m=int(digits[3:5]);d=int(digits[5:])
 elif len(digits)==8:y=int(digits[:4]);m=int(digits[4:6]);d=int(digits[6:])
 else:raise ValueError(f"bad TPEx ROC date {x!r}")
 return f"{y:04d}-{m:02d}-{d:02d}"
def idx(fields,*names,required=True):
 normalized=[clean(x) for x in fields]
 for name in names:
  n=clean(name)
  if n in normalized:return normalized.index(n)
 if required:raise RuntimeError(f"Missing TPEx field {names}; got {normalized}")
 return None
def get_table(session,action,start,end):
 url=BASE+action
 params={"startDate":start.replace("-","/"),"endDate":end.replace("-","/"),"response":"json"}
 r=session.get(url,params=params,timeout=(15,150),headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"})
 r.raise_for_status();j=r.json()
 tables=j.get("tables") or []
 for t in tables:
  if t.get("fields") and isinstance(t.get("data"),list):return t,r.url,j
 if j.get("fields") and isinstance(j.get("data"),list):return {"fields":j["fields"],"data":j["data"]},r.url,j
 raise RuntimeError(f"TPEx {action} returned no populated table: {str(j)[:500]}")
def collect_exdailyq(session,start,end):
 t,url,payload=get_table(session,"bulletin/exDailyQ",start,end)
 f=t["fields"];rows=t["data"]
 di=idx(f,"除權息日期");si=idx(f,"代號","股票代號","證券代號");ni=idx(f,"名稱","股票名稱",required=False)
 pi=idx(f,"除權息前收盤價");ri=idx(f,"除權息參考價")
 ti=idx(f,"權/息",required=False);cash_i=idx(f,"現金股利","息值",required=False)
 stock_i=idx(f,"每仟股無償配股","權值",required=False)
 rights_fields=[idx(f,n,required=False) for n in ("現金增資股數","公開承銷股數","員工認購股數","原股東認購股數","按持股比例仟股認購")]
 ui=idx(f,"漲停價","漲停價格",required=False);li=idx(f,"跌停價","跌停價格",required=False)
 out=[]
 for raw in rows:
  sym=clean(raw[si])
  try:day=rocdate(raw[di])
  except:continue
  if not start<=day<=end:continue
  prev=num(raw[pi]);ref=num(raw[ri])
  if not prev or not ref or prev<=0 or ref<=0:raise RuntimeError(f"Invalid TPEx exDailyQ reference {sym} {day}: {raw}")
  cash=num(raw[cash_i]) if cash_i is not None else None
  stock=num(raw[stock_i]) if stock_i is not None else None
  rights_shares=sum((num(raw[k]) or 0.0) for k in rights_fields if k is not None)
  kind=clean(raw[ti]) if ti is not None else ""
  status="resolved"
  if cash is None and ("息" in kind or kind not in {"權"}):status="unresolved_cash_component"
  if stock is None and "權" in kind:status="unresolved_stock_component"
  if cash is None and kind=="權":cash=0.0
  if stock is None and kind=="息":stock=0.0
  out.append({"market":"TPEX","symbol":sym,"date":day,"action_type":kind,
   "previous_close":prev,"reference_price":ref,"adjustment_ratio":ref/prev,
   "cash_dividend":cash,"stock_dividend":stock,"rights_issue_shares":rights_shares,"cash_dividend_status":status,
   "limit_up":num(raw[ui]) if ui is not None else None,
   "limit_down":num(raw[li]) if li is not None else None,
   "source":"TPEx bulletin/exDailyQ"})
 if not out:raise RuntimeError("TPEx exDailyQ returned zero dated events; coverage cannot be verified")
 return out,{"url":url,"fields":[clean(x) for x in f],"rows":len(out)}
def collect_revivt(session,start,end):
 t,url,payload=get_table(session,"bulletin/revivt",start,end)
 f=t["fields"];rows=t["data"]
 di=idx(f,"恢復買賣日期","恢復買賣日");si=idx(f,"股票代號","代號","證券代號")
 pi=idx(f,"最後交易日之收盤價格","最後交易日收盤價格")
 ri=idx(f,"減資恢復買賣開始日參考價格","減資恢復買賣開始日參考價")
 ui=idx(f,"漲停價格","漲停價",required=False);li=idx(f,"跌停價格","跌停價",required=False)
 reason_i=idx(f,"減資原因","原因",required=False)
 out=[]
 for raw in rows:
  sym=clean(raw[si])
  try:day=rocdate(raw[di])
  except:continue
  if not start<=day<=end:continue
  prev=num(raw[pi]);ref=num(raw[ri])
  if not prev or not ref or prev<=0 or ref<=0:raise RuntimeError(f"Invalid TPEx revivt reference {sym} {day}: {raw}")
  out.append({"market":"TPEX","symbol":sym,"date":day,"event_type":"capital_reduction",
   "previous_close":prev,"reference_price":ref,
   "limit_up":num(raw[ui]) if ui is not None else None,
   "limit_down":num(raw[li]) if li is not None else None,
   "reason":clean(raw[reason_i]) if reason_i is not None else "",
   "source":"TPEx bulletin/revivt"})
 if not out:raise RuntimeError("TPEx revivt returned zero dated capital-reduction events; coverage cannot be verified")
 return out,{"url":url,"fields":[clean(x) for x in f],"rows":len(out)}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--start-date",default="2010-01-01")
 p.add_argument("--end-date",default="2026-11-12")
 p.add_argument("--special-start-date",default="2013-01-01")
 p.add_argument("--output",default="data/top10_actions_twt49u")
 a=p.parse_args()
 s=requests.Session()
 ex,exmeta=collect_exdailyq(s,a.start_date,a.end_date)
 rev,revmeta=collect_revivt(s,a.special_start_date,a.end_date)
 ex_years=sorted({int(r["date"][:4]) for r in ex})
 rev_years=sorted({int(r["date"][:4]) for r in rev})
 required_ex_years={2015,2020,2025}
 if not required_ex_years.issubset(set(ex_years)):
  raise RuntimeError(f"TPEx exDailyQ historical coverage missing required years: {required_ex_years-set(ex_years)}; got {ex_years}")
 if 2025 not in rev_years:
  raise RuntimeError(f"TPEx revivt 2025 control events missing; years={rev_years}")
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(ex).to_csv(out/"tpex_exdailyq_all.csv",index=False)
 special_path=out/"special_events.csv"
 new=pd.DataFrame(rev)
 if special_path.exists():
  old=pd.read_csv(special_path,dtype={"symbol":str})
  if "market" not in old:old["market"]="TWSE"
  combined=pd.concat([old,new],ignore_index=True)
 else:combined=new
 combined=combined.drop_duplicates(["market","symbol","date","event_type"],keep="last")
 combined.to_csv(special_path,index=False)
 previous={}
 mp=out/"special_events_manifest.json"
 if mp.exists():
  try:previous=json.loads(mp.read_text(encoding="utf-8"))
  except:previous={}
 tpex_meta={"exDailyQ":exmeta,"revivt":revmeta,"exDailyQ_rows":len(ex),"revivt_rows":len(rev),"exDailyQ_years":ex_years,"revivt_years":rev_years}
 complete=bool(previous.get("complete",False)) and len(ex)>0 and len(rev)>0
 manifest={**previous,"complete":complete,"tpex":tpex_meta,
  "sources":list(set(previous.get("sources",[])+["TPEx bulletin/exDailyQ","TPEx bulletin/revivt"])),
  "coverage":{"tpex_exdailyq":[a.start_date,a.end_date],"tpex_revivt":[a.special_start_date,a.end_date]},
  "note":"TPEx exDailyQ supplies official reference prices, cash/stock dividend components and event-day limits. TPEx revivt supplies capital-reduction resumption events; affected symbols are blocked until share/cash-refund mechanics are modeled."}
 mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps({"exDailyQ_rows":len(ex),"revivt_rows":len(rev),"special_manifest_complete":complete},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
