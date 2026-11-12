"""Collect official TWSE raw daily OHLCV for the fixed ten-stock research basket.

All ten selected symbols are TWSE-listed. This collector uses the TWSE RWD
monthly STOCK_DAY endpoint (no paid token). Prices are unadjusted execution
bars; volume and turnover are official exchange-reported values. It does not
infer price limits, dividends, or historical universe membership.
"""
from __future__ import annotations
import argparse, hashlib, json, random, time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import requests

BASE = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
STOCKS = {
    "2330": "台積電", "2454": "聯發科", "1303": "南亞", "2327": "國巨",
    "2408": "南亞科", "2308": "台達電", "3037": "欣興", "3008": "大立光",
    "2383": "台光電", "2303": "聯電",
}
FIELDS = ["date","volume","turnover","open","high","low","close","change","trades","note"]

def roc_date(value: str) -> str:
    s=str(value).strip().replace("年","/").replace("月","/").replace("日","")
    parts=s.split("/")
    if len(parts)!=3: raise ValueError(f"Unexpected TWSE date: {value!r}")
    y=int(parts[0]); y += 1911 if y < 1911 else 0
    return date(y,int(parts[1]),int(parts[2])).isoformat()

def number(value):
    s=str(value).strip().replace(",","")
    if s in {"","--","-","X","nan","None"}: return None
    return float(s)

def fetch_month(session, symbol, month, retries=6):
    params={"date":month.strftime("%Y%m01"),"stockNo":symbol,"response":"json"}
    last=None
    for attempt in range(retries):
        try:
            r=session.get(BASE,params=params,timeout=(10,45),
                          headers={"User-Agent":"Mozilla/5.0 research-data-collector"})
            r.raise_for_status()
            payload=r.json()
            if payload.get("stat") not in {"OK","很抱歉，沒有符合條件的資料!"}:
                raise RuntimeError(f"TWSE status={payload.get('stat')}")
            if payload.get("stat")!="OK": return []
            fields=payload.get("fields",[])
            rows=[]
            for raw in payload.get("data",[]):
                item=dict(zip(fields,raw))
                rows.append({
                    "symbol":symbol,"name":STOCKS[symbol],
                    "date":roc_date(item.get("日期")),
                    "volume":number(item.get("成交股數")),
                    "turnover":number(item.get("成交金額")),
                    "open":number(item.get("開盤價")),
                    "high":number(item.get("最高價")),
                    "low":number(item.get("最低價")),
                    "close":number(item.get("收盤價")),
                    "change":number(item.get("漲跌價差")),
                    "trades":number(item.get("成交筆數")),
                    "note":str(item.get("註記","")).strip(),
                    "source":"TWSE RWD STOCK_DAY",
                    "source_month":month.strftime("%Y-%m"),
                })
            return rows
        except Exception as e:
            last=e
            if attempt+1<retries:
                time.sleep(min(30,2**attempt)+random.random())
    raise RuntimeError(f"{symbol} {month:%Y-%m} failed after {retries} attempts: {last}")

def collect(output: Path, start: str, end: str, pause: float):
    start_d=date.fromisoformat(start); end_d=date.fromisoformat(end)
    if start_d>end_d: raise ValueError("start must be <= end")
    months=pd.date_range(start_d.replace(day=1),end_d.replace(day=1),freq="MS")
    output.mkdir(parents=True,exist_ok=True)
    bars_dir=output/"bars"; bars_dir.mkdir(exist_ok=True)
    session=requests.Session()
    statuses=[]
    for n,(symbol,name) in enumerate(STOCKS.items(),1):
        path=bars_dir/f"{symbol}.csv"
        old=pd.read_csv(path,dtype={"symbol":str}) if path.exists() else pd.DataFrame()
        existing=set(old.date.astype(str)) if not old.empty and "date" in old else set()
        rows=[]
        for i,m in enumerate(months,1):
            got=fetch_month(session,symbol,m)
            rows.extend(got)
            if pause: time.sleep(pause)
            if i%24==0:
                print(f"[PROGRESS] {symbol} {i}/{len(months)} months",flush=True)
        fresh=pd.DataFrame(rows)
        if not fresh.empty:
            fresh=fresh.drop_duplicates(["symbol","date"],keep="last")
            fresh=fresh[(fresh.date>=start)&(fresh.date<=end)]
            if not old.empty:
                fresh=pd.concat([old[~old.date.astype(str).isin(set(fresh.date.astype(str)))],fresh],ignore_index=True)
        elif not old.empty:
            fresh=old
        if fresh.empty: raise RuntimeError(f"No TWSE raw data for {symbol}")
        fresh["date"]=pd.to_datetime(fresh.date,errors="coerce").dt.strftime("%Y-%m-%d")
        for c in ["volume","turnover","open","high","low","close"]:
            fresh[c]=pd.to_numeric(fresh[c],errors="coerce")
        fresh=fresh.dropna(subset=["date","volume","turnover","open","high","low","close"])
        valid=(fresh[["open","high","low","close"]].gt(0).all(axis=1)
               &(fresh.volume>=0)&(fresh.turnover>=0)
               &(fresh.high>=fresh[["open","close","low"]].max(axis=1))
               &(fresh.low<=fresh[["open","close","high"]].min(axis=1)))
        fresh=fresh.loc[valid].drop_duplicates(["symbol","date"]).sort_values("date").reset_index(drop=True)
        fresh.to_csv(path,index=False,float_format="%.6f")
        statuses.append({"symbol":symbol,"name":name,"rows":len(fresh),
                         "first_date":fresh.date.min(),"last_date":fresh.date.max(),
                         "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                         "source":"TWSE RWD STOCK_DAY"})
        print(f"[DONE] {n}/10 {symbol} {name}: {len(fresh)} rows {fresh.date.min()}..{fresh.date.max()}",flush=True)
    manifest={
        "source":"Taiwan Stock Exchange official RWD historical monthly STOCK_DAY",
        "endpoint":BASE,"collected_at_taipei":datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
        "start_date":start,"end_date":end,"symbols_expected":10,
        "symbols_collected":len(statuses),"collection_complete":len(statuses)==10,
        "price_basis":"raw unadjusted exchange OHLC; execution-price candidate",
        "volume_turnover":"official TWSE reported daily shares and traded value",
        "limitations":[
          "The ten symbols are the preselected recent-liquidity basket, not a point-in-time top-ten universe for every historical date.",
          "This artifact does not include historical official daily price limits, reference prices, dividends, stock dividends, or split factors.",
          "No backtest should execute until price-limit and corporate-action datasets are independently reconciled."
        ],
        "status_rows":statuses
    }
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="data/top10_official_raw")
    p.add_argument("--start-date",default="2010-01-04")
    p.add_argument("--end-date",default=date.today().isoformat())
    p.add_argument("--pause-seconds",type=float,default=.35)
    a=p.parse_args()
    print(json.dumps(collect(Path(a.output),a.start_date,a.end_date,a.pause_seconds),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
