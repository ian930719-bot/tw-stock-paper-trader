"""Collect TWSE official capital-reduction and par-value resumption events."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd,requests


BASE="https://www.twse.com.tw/exchangeReport/"
def rocdate(v):
 s=str(v).strip().replace("年","/").replace("月","/").replace("日","")
 p=s.split("/")
 if len(p)==3:
  y=int(p[0]);y+=1911 if y<1911 else 0
  return f"{y:04d}-{int(p[1]):02d}-{int(p[2]):02d}"
 if len(s)==8 and s.isdigit():return f"{s[:4]}-{s[4:6]}-{s[6:]}"
 raise ValueError(v)
def num(v):
 try:return float(str(v).replace(",","").strip())
 except:return None
def ix(h,*terms,required=True):
 for i,x in enumerate(h):
  x=str(x).replace(" ","")
  if any(t in x for t in terms):return i
 if required:raise RuntimeError(f"missing {terms} in {h}")
 return None
def get(session,dataset,year,start,end):
 url=BASE+dataset
 lo=pd.Timestamp(start);hi=pd.Timestamp(end)
 candidates=[
  {"response":"json","strDate":lo.strftime("%Y%m%d"),"endDate":hi.strftime("%Y%m%d")},
  {"response":"json","strDate":f"{lo.year-1911:03d}{lo.strftime('%m%d')}","endDate":f"{hi.year-1911:03d}{hi.strftime('%m%d')}"},
  {"response":"json","startDate":lo.strftime("%Y%m%d"),"endDate":hi.strftime("%Y%m%d")},
  {"response":"json","strDate":hi.strftime("%Y%m%d"),"endDate":lo.strftime("%Y%m%d")},
 ]
 errors=[]
 for params in candidates:
  try:
   r=session.get(url,params=params,timeout=(15,120),headers={"User-Agent":"Mozilla/5.0"});r.raise_for_status()
   j=r.json()
   if isinstance(j,dict) and "stat" in j and ("查無資料" in str(j.get("stat")) or "沒有符合條件" in str(j.get("stat"))):
    return {"fields":[],"data":[],"stat":j.get("stat")}
   if not isinstance(j,dict) or not isinstance(j.get("fields"),list) or not isinstance(j.get("data"),list):
    errors.append({"url":r.url,"payload":str(j)[:300]});continue
   if not j["fields"] and not j["data"]:
    return j
   if j["data"]:
    h=[str(x).strip() for x in j["fields"]]
    di=ix(h,"恢復買賣日期","恢復買賣日","資料日期","日期")
    found=False
    for row in j["data"]:
     try:
      day=rocdate(row[di])
      if start<=day<=end:found=True;break
     except:pass
    if not found:
     errors.append({"url":r.url,"sample":j["data"][:1]});continue
   return j
  except Exception as e:errors.append({"params":params,"error":str(e)})
 raise RuntimeError(f"{dataset} {year} query failed; attempts={errors}")

def main():
 p=argparse.ArgumentParser();p.add_argument("--start-date",default="2011-01-01");p.add_argument("--end-date",default="2026-11-12");p.add_argument("--output",default="data/top10_actions_twt49u");a=p.parse_args()
 s=requests.Session();allrows=[];schemas={};skipped_periods=[]
 for dataset,kind in [("TWTAUU","capital_reduction"),("TWTB8U","par_value_change")]:
  rows=[];schema=[]
  periods=[(a.start_date,a.end_date)]
  try:
   payload=get(s,dataset,"full_range",a.start_date,a.end_date)
   batches=[(payload,a.start_date,a.end_date)]
  except Exception as full_error:
   print(f"[FULL RANGE FALLBACK] {dataset}: {full_error}",flush=True)
   batches=[]
   for y in range(pd.Timestamp(a.start_date).year,pd.Timestamp(a.end_date).year+1):
    lo=max(pd.Timestamp(a.start_date),pd.Timestamp(y,1,1)).strftime("%Y-%m-%d")
    hi=min(pd.Timestamp(a.end_date),pd.Timestamp(y,12,31)).strftime("%Y-%m-%d")
    try:batches.append((get(s,dataset,y,lo,hi),lo,hi))
    except Exception as year_error:
     if y==2011:
      print(f"[NO 2011 ROWSET] {dataset}: {year_error}",flush=True)
      skipped_periods.append({"dataset":dataset,"year":y,"reason":str(year_error)})
      continue
     raise
  for j,lo,hi in batches:
   h=[str(x).strip() for x in j["fields"]];schema.append({"period":[lo,hi],"fields":h,"rows":len(j["data"])})
   if not h and not j["data"]:
    print(f"[PERIOD EMPTY] {dataset} {lo}..{hi}",flush=True)
    continue
   di=ix(h,"恢復買賣日期","恢復買賣日","資料日期","日期")
   si=ix(h,"股票代號","證券代號");ni=ix(h,"股票名稱","證券名稱",required=False)
   pi=ix(h,"停止買賣前收盤價格","停止買賣前價格","停止買賣前收盤價","變更前收盤價")
   ri=ix(h,"恢復買賣參考價格","恢復買賣參考價","變更後恢復買賣參考價","恢復交易參考價格")
   ui=ix(h,"漲停價格","漲停價",required=False);li=ix(h,"跌停價格","跌停價",required=False)
   reason_i=ix(h,"減資原因","變更原因","原因",required=False)
   for raw in j["data"]:
    if len(raw)<=max(di,si,pi,ri):continue
    sym=str(raw[si]).strip()
    try:day=rocdate(raw[di])
    except:continue
    if not lo<=day<=hi:continue
    prev=num(raw[pi]);ref=num(raw[ri])
    if not prev or not ref or prev<=0 or ref<=0:raise RuntimeError(f"{dataset} invalid reference {sym} {day}: {raw}")
    rows.append({"market":"TWSE","symbol":sym,"date":day,"event_type":kind,"previous_close":prev,"reference_price":ref,
      "limit_up":num(raw[ui]) if ui is not None else None,"limit_down":num(raw[li]) if li is not None else None,
      "reason":str(raw[reason_i]).strip() if reason_i is not None else "",
      "source":"TWSE "+dataset})
  schemas[dataset]=schema
  allrows.extend(rows)
  print(f"[DONE] {dataset}: {len(rows)} special events",flush=True)
 if not any(r["symbol"]=="2327" and r["date"]=="2025-08-25" for r in allrows):
  raise RuntimeError("Known 2327 2025-08-25 TWSE special resumption marker was not reconciled to TWTAUU/TWTB8U.")
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 frame=pd.DataFrame(allrows,columns=["market","symbol","date","event_type","previous_close","reference_price","limit_up","limit_down","reason","source"])
 frame.to_csv(out/"special_events.csv",index=False)
 m={"source":"TWSE official TWTAUU/TWTB8U","collected_at_taipei":datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
  "requested_period":[a.start_date,a.end_date],"rows":len(frame),"events_by_type":frame.event_type.value_counts().to_dict() if len(frame) else {},
  "complete":len(skipped_periods)==0,"skipped_periods":skipped_periods,"schemas":schemas,"note":"Empty response for a year is accepted only when endpoint returns a valid JSON schema and empty data list."}
 (out/"special_events_manifest.json").write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps({k:v for k,v in m.items() if k!="schemas"},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
