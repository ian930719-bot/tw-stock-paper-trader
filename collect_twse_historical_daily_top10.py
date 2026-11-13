"""Reconstruct point-in-time daily top-10 TWSE stocks by official turnover.

Uses official TWSE MI_INDEX daily market-wide close reports. It requests only
dates present in the official raw-price collection's trading calendar. No
current-survivor filtering is applied to the ranking source.
"""
from __future__ import annotations
import argparse,csv,json,random,time,hashlib
from pathlib import Path
import pandas as pd
import requests

URL="https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
def normnum(x):
 s=str(x).strip().replace(",","")
 if s in {"","--","-","X","nan","None"}: return None
 try:return float(s)
 except:return None

def fetch_day(session,day,retries=6):
 params={"date":pd.Timestamp(day).strftime("%Y%m%d"),"type":"ALLBUT0999","response":"json"}
 err=None
 for k in range(retries):
  try:
   r=session.get(URL,params=params,timeout=(10,45),headers={"User-Agent":"Mozilla/5.0"})
   r.raise_for_status(); j=r.json()
   if j.get("stat")!="OK": return []
   # Newer TWSE responses contain a table named "每日收盤行情" with fields/data.
   tables=j.get("tables",[])
   table=next((t for t in tables if "每日收盤行情" in str(t.get("title",""))),None)
   if table is None:
    table=next((t for t in tables if "證券代號" in t.get("fields",[]) and "成交金額" in t.get("fields",[])),None)
   if table is None: raise RuntimeError("MI_INDEX daily quote table not found")
   fields=table.get("fields",[])
   def idx(*names):
    for n in names:
     if n in fields:return fields.index(n)
    raise RuntimeError("Missing MI_INDEX field: "+"/".join(names))
   code=idx("證券代號"); name=idx("證券名稱"); vol=idx("成交股數"); money=idx("成交金額"); op=idx("開盤價"); hi=idx("最高價"); lo=idx("最低價"); cl=idx("收盤價")
   rows=[]
   for a in table.get("data",[]):
    sym=str(a[code]).strip()
    if not (len(sym)==4 and sym.isdigit()) or sym.startswith("0") or sym.startswith("91"): continue
    v=normnum(a[vol]); m=normnum(a[money])
    if v is None or m is None: continue
    rows.append({"date":pd.Timestamp(day).strftime("%Y-%m-%d"),"symbol":sym,"name":str(a[name]).strip(),
                 "volume_shares":v,"turnover_ntd":m,"open":normnum(a[op]),"high":normnum(a[hi]),
                 "low":normnum(a[lo]),"close":normnum(a[cl]),"rank_by_turnover":None,"source":"TWSE MI_INDEX"})
   if not rows: raise RuntimeError("No market-wide equity rows parsed")
   rows.sort(key=lambda z:(-z["turnover_ntd"],z["symbol"]))
   for n,z in enumerate(rows,1):z["rank_by_turnover"]=n
   return rows
  except Exception as e:
   err=e
   if k+1<retries:time.sleep(min(25,2**k)+random.random())
 raise RuntimeError(f"{day}: {err}")

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--raw-bars",default="data/top10_official_raw/bars")
 p.add_argument("--output",default="data/twse_marketwide_turnover")
 p.add_argument("--pause",type=float,default=.25)
 a=p.parse_args()
 files=sorted(Path(a.raw_bars).glob("*.csv"))
 if len(files)!=10:raise RuntimeError(f"Expected 10 official raw bar files, got {len(files)}")
 dates=sorted(set().union(*[set(pd.read_csv(f,usecols=["date"]).date.astype(str)) for f in files]))
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 daily=out/"daily";daily.mkdir(exist_ok=True)
 s=requests.Session();summary=[];combined=[]
 for i,d in enumerate(dates,1):
  path=daily/(d+".csv")
  if path.exists() and path.stat().st_size>100:
   frame=pd.read_csv(path); rows=frame.to_dict("records")
  else:
   rows=fetch_day(s,d)
   pd.DataFrame(rows).to_csv(path,index=False)
  if not rows:raise RuntimeError(f"No market-wide TWSE data for {d}")
  top=rows[:10]
  combined.extend(top)
  summary.append({"date":d,"universe_rows":len(rows),"top10_turnover_ntd":sum(x["turnover_ntd"] for x in top)})
  if i%100==0:print(f"[PROGRESS] {i}/{len(dates)} dates; latest={d}",flush=True)
  if a.pause:time.sleep(a.pause)
 frame=pd.DataFrame(combined)
 frame=frame.sort_values(["date","rank_by_turnover"])
 frame.to_csv(out/"historical_daily_top10_by_turnover.csv",index=False)
 # Extract complete daily histories for every symbol that ever entered the
 # point-in-time top ten; one pass over the saved market-wide daily files.
 union=sorted(frame.symbol.astype(str).unique())
 bars_dir=out/"symbol_bars";bars_dir.mkdir(exist_ok=True)
 handles={}
 try:
  for sym in union:
   fh=(bars_dir/f"{sym}.csv").open("w",newline="",encoding="utf-8")
   wr=csv.writer(fh);wr.writerow(["symbol","name","date","open","high","low","close","volume","turnover","note","source"])
   handles[sym]=(fh,wr)
  for path in sorted(daily.glob("*.csv")):
   dayframe=pd.read_csv(path,dtype={"symbol":str})
   for r in dayframe.itertuples(index=False):
    sym=str(r.symbol)
    if sym not in handles:continue
    vals=[r.open,r.high,r.low,r.close]
    if any(pd.isna(x) for x in vals):continue
    handles[sym][1].writerow([sym,r.name,r.date,r.open,r.high,r.low,r.close,
      r.volume_shares,r.turnover_ntd,"","TWSE MI_INDEX"])
 finally:
  for fh,wr in handles.values():fh.close()
 man={"source":"TWSE official MI_INDEX daily market-wide closing report","endpoint":URL,
      "dates_requested":len(dates),"dates_completed":len(summary),"complete":len(summary)==len(dates),
      "first_date":dates[0],"last_date":dates[-1],"rows":len(frame),"point_in_time_union_symbols":len(union),"symbol_bar_files":len(union),
      "sha256":hashlib.sha256((out/"historical_daily_top10_by_turnover.csv").read_bytes()).hexdigest(),
      "method":"rank all listed common-stock rows by official daily turnover NT dollars; top 10 per date",
      "limitations":["Market-wide endpoint includes securities that must be classified; script filters four-digit numeric codes only.",
                     "Historical universe categories and special security types require separate classification audit.",
                     "Daily top ten is ranked on same-day close turnover and can only be used to choose next-session trades, never same-day entries."]}
 (out/"manifest.json").write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(man,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
