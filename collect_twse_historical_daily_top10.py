"""Point-in-time daily top ten Taiwan common stocks by official traded shares.

TWSE uses MI_INDEX; TPEx uses its dated afterTrading/otc JSON endpoint.
The daily close on t determines eligibility for a signal at close t; orders
execute no earlier than the next session. ETF, TDR and warrant-like codes are
excluded by a conservative code-class filter, not a current-survivor master.
"""
from __future__ import annotations
import argparse,csv,json,random,time,hashlib,re
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

TWSE_URL="https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TPEX_URL="https://www.tpex.org.tw/www/zh-tw/afterTrading/otc"

def clean(x):
 return BeautifulSoup(str(x if x is not None else ""),"html.parser").get_text(" ",strip=True).replace(",","").strip()
def normnum(x):
 s=clean(x).replace("−","-").replace("－","-")
 if s in {"","--","-","X","nan","None","除權息"}:return None
 try:return float(s)
 except:return None
def normfield(x):
 return re.sub(r"\s+","",BeautifulSoup(str(x),"html.parser").get_text(" ",strip=True))
def common_code(sym):
 s=str(sym).strip()
 return bool(re.fullmatch(r"\d{4}",s)) and not s.startswith(("00","91"))

def rank_rows(rows):
 rows.sort(key=lambda z:(-z["volume_shares"],-z["turnover_ntd"],z["symbol"]))
 for n,z in enumerate(rows,1):z["rank_by_volume"]=n
 for n,z in enumerate(sorted(rows,key=lambda z:(-z["turnover_ntd"],-z["volume_shares"],z["symbol"])),1):z["rank_by_turnover"]=n
 return rows

def fetch_twse_day(session,day,retries=6):
 params={"date":pd.Timestamp(day).strftime("%Y%m%d"),"type":"ALLBUT0999","response":"json"}
 err=None
 for k in range(retries):
  try:
   r=session.get(TWSE_URL,params=params,timeout=(10,45),headers={"User-Agent":"Mozilla/5.0"})
   r.raise_for_status();j=r.json()
   if j.get("stat")!="OK":raise RuntimeError("TWSE stat="+str(j.get("stat")))
   tables=j.get("tables",[])
   table=next((t for t in tables if "每日收盤行情" in str(t.get("title",""))),None)
   if table is None:table=next((t for t in tables if "證券代號" in t.get("fields",[]) and "成交金額" in t.get("fields",[])),None)
   if table is None:raise RuntimeError("TWSE MI_INDEX daily quote table not found")
   fields=[normfield(x) for x in table.get("fields",[])]
   def idx(*names,required=True):
    for n in names:
     nn=normfield(n)
     if nn in fields:return fields.index(nn)
    if required:raise RuntimeError("Missing TWSE field "+"/".join(names)+"; got "+repr(fields))
    return None
   code=idx("證券代號");name=idx("證券名稱");vol=idx("成交股數");money=idx("成交金額")
   op=idx("開盤價");hi=idx("最高價");lo=idx("最低價");cl=idx("收盤價")
   note=idx("註記",required=False)
   rows=[]
   for a in table.get("data",[]):
    sym=clean(a[code])
    if not common_code(sym):continue
    v=normnum(a[vol]);m=normnum(a[money])
    if v is None or m is None:continue
    rows.append({"date":pd.Timestamp(day).strftime("%Y-%m-%d"),"symbol":"TWSE_"+sym,"stock_id":sym,"market":"TWSE",
     "name":clean(a[name]),"volume_shares":v,"turnover_ntd":m,"open":normnum(a[op]),"high":normnum(a[hi]),
     "low":normnum(a[lo]),"close":normnum(a[cl]),"note":clean(a[note]) if note is not None else "",
     "rank_by_volume":None,"rank_by_turnover":None,"source":"TWSE MI_INDEX"})
   if len(rows)<100:raise RuntimeError(f"TWSE suspiciously few common stocks: {len(rows)}")
   return rows
  except Exception as e:
   err=e
   if k+1<retries:time.sleep(min(25,2**k)+random.random())
 raise RuntimeError(f"TWSE {day}: {err}")

def fetch_tpex_day(session,day,retries=6):
 day=pd.Timestamp(day)
 roc=f"{day.year-1911}/{day.month:02d}/{day.day:02d}"
 params={"date":roc,"type":"EW","response":"json"}
 err=None
 for k in range(retries):
  try:
   r=session.get(TPEX_URL,params=params,timeout=(10,45),headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"})
   r.raise_for_status();j=r.json()
   tables=j.get("tables") or []
   table=None
   for t in tables:
    fields=[normfield(x) for x in t.get("fields",[])]
    if any(x in fields for x in ("代號","股票代號","證券代號")) and any(x in fields for x in ("成交股數","成交量")):
     table=t;break
   if table is None and isinstance(j.get("fields"),list):table={"fields":j["fields"],"data":j.get("data",[])}
   if table is None:raise RuntimeError("TPEx afterTrading/otc quote table not found: "+str(j.get("stat","")))
   fields=[normfield(x) for x in table.get("fields",[])]
   def idx(*names,required=True):
    for n in names:
     nn=normfield(n)
     if nn in fields:return fields.index(nn)
    if required:raise RuntimeError("Missing TPEx field "+"/".join(names)+"; got "+repr(fields))
    return None
   code=idx("代號","股票代號","證券代號");name=idx("名稱","股票名稱","證券名稱")
   vol=idx("成交股數","成交量");money=idx("成交金額","成交金額(元)","成交值")
   op=idx("開盤","開盤價");hi=idx("最高","最高價");lo=idx("最低","最低價");cl=idx("收盤","收盤價")
   note=idx("註記","備註",required=False)
   rows=[]
   for a in table.get("data",[]):
    sym=clean(a[code])
    if not common_code(sym):continue
    v=normnum(a[vol]);m=normnum(a[money])
    if v is None or m is None:continue
    rows.append({"date":day.strftime("%Y-%m-%d"),"symbol":"TPEX_"+sym,"stock_id":sym,"market":"TPEX",
     "name":clean(a[name]),"volume_shares":v,"turnover_ntd":m,"open":normnum(a[op]),"high":normnum(a[hi]),
     "low":normnum(a[lo]),"close":normnum(a[cl]),"note":clean(a[note]) if note is not None else "",
     "rank_by_volume":None,"rank_by_turnover":None,"source":"TPEx afterTrading/otc"})
   if len(rows)<100:raise RuntimeError(f"TPEx suspiciously few common stocks: {len(rows)}")
   return rows
  except Exception as e:
   err=e
   if k+1<retries:time.sleep(min(25,2**k)+random.random())
 raise RuntimeError(f"TPEx {day}: {err}")

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--raw-bars",default="data/top10_official_raw/bars")
 p.add_argument("--output",default="data/twse_marketwide_turnover")
 p.add_argument("--pause",type=float,default=.35)
 a=p.parse_args()
 files=sorted(Path(a.raw_bars).glob("*.csv"))
 if len(files)!=10:raise RuntimeError(f"Expected 10 official TWSE raw calendar files, got {len(files)}")
 dates=sorted(set().union(*[set(pd.read_csv(f,usecols=["date"]).date.astype(str)) for f in files]))
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 daily=out/"daily";daily.mkdir(exist_ok=True)
 session=requests.Session();summary=[];combined=[]
 for i,d in enumerate(dates,1):
  path=daily/(d+".csv")
  cached=pd.read_csv(path,dtype={"symbol":str,"stock_id":str}) if path.exists() and path.stat().st_size>100 else pd.DataFrame()
  if not cached.empty and {"market","stock_id","rank_by_volume"}.issubset(cached.columns) and {"TWSE","TPEX"}.issubset(set(cached.market.astype(str))):
   rows=cached.to_dict("records")
  else:
   twse_rows=fetch_twse_day(session,d)
   tpex_rows=fetch_tpex_day(session,d)
   rows=rank_rows(twse_rows+tpex_rows)
   pd.DataFrame(rows).to_csv(path,index=False)
  if not rows:raise RuntimeError(f"No combined TWSE/TPEx market-wide data for {d}")
  # Cached daily rows were already ranked on volume when written.
  rows=sorted(rows,key=lambda z:int(z.get("rank_by_volume") or 999999))
  top=rows[:10];combined.extend(top)
  summary.append({"date":d,"universe_rows":len(rows),"twse_rows":sum(x["market"]=="TWSE" for x in rows),
                  "tpex_rows":sum(x["market"]=="TPEX" for x in rows),
                  "top10_volume_shares":sum(x["volume_shares"] for x in top)})
  if i%100==0:print(f"[PROGRESS] {i}/{len(dates)} dates; latest={d}",flush=True)
  if a.pause:time.sleep(a.pause)
 frame=pd.DataFrame(combined).sort_values(["date","rank_by_volume"])
 out_file=out/"historical_daily_top10_by_volume.csv"
 frame.to_csv(out_file,index=False)
 union=sorted(frame.symbol.astype(str).unique())
 bars_dir=out/"symbol_bars";bars_dir.mkdir(exist_ok=True)
 handles={}
 try:
  for sym in union:
   fh=(bars_dir/f"{sym}.csv").open("w",newline="",encoding="utf-8")
   wr=csv.writer(fh);wr.writerow(["symbol","stock_id","market","name","date","open","high","low","close","volume","turnover","note","source"])
   handles[sym]=(fh,wr)
  for path in sorted(daily.glob("*.csv")):
   dayframe=pd.read_csv(path,dtype={"symbol":str,"stock_id":str})
   for r in dayframe.itertuples(index=False):
    sym=str(r.symbol)
    if sym not in handles:continue
    handles[sym][1].writerow([sym,r.stock_id,r.market,r.name,r.date,r.open,r.high,r.low,r.close,
      r.volume_shares,r.turnover_ntd,getattr(r,"note",""),r.source])
 finally:
  for fh,wr in handles.values():fh.close()
 market_counts=frame.market.value_counts().to_dict()
 man={"source":["TWSE official MI_INDEX","TPEx official afterTrading/otc"],
      "endpoints":[TWSE_URL,TPEX_URL],"markets_included":["TWSE","TPEX"],
      "dates_requested":len(dates),"dates_completed":len(summary),"complete":len(summary)==len(dates),
      "first_date":dates[0],"last_date":dates[-1],"top10_rows":len(frame),
      "point_in_time_union_symbols":len(union),"symbol_bar_files":len(union),"top10_rows_by_market":market_counts,
      "sha256":hashlib.sha256(out_file.read_bytes()).hexdigest(),
      "method":"rank TWSE and TPEx ordinary common-stock rows together by official daily traded shares; top 10 per date; turnover retained as secondary field",
      "limitations":["Common-stock filter keeps four-digit numeric codes and excludes 00xx ETF and 91xx TDR classes; six-digit warrants are excluded. Residual special-security classifications should be checked.",
       "The seed-stock trading calendar is used to enumerate market sessions; TPEx must return at least 100 ordinary-stock rows for every requested session.",
       "Daily top ten is known at the close and is used only for next-session order eligibility."]}
 (out/"manifest.json").write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(man,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
