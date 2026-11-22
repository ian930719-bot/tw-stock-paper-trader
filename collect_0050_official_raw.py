"""Collect official TWSE raw daily OHLCV for 0050 only.

Raw prices are preserved exactly as traded. This dataset is NOT split-adjusted
and must not be used for cross-split returns until the 2025-06-18 4-for-1 ETF
split is independently reconciled. Corporate actions are stored separately.
"""
from __future__ import annotations
import argparse, hashlib, json, random, time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import requests

URL = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
SYMBOL = "0050"
NAME = "元大台灣50"
FIELDS = ["date","volume","turnover","open","high","low","close","change","trades","note"]

def roc_date(value: str) -> str:
    s = str(value).strip().replace("年","/").replace("月","/").replace("日","")
    p = s.split("/")
    if len(p) != 3:
        raise ValueError(f"Unexpected TWSE date: {value!r}")
    y = int(p[0]); y += 1911 if y < 1911 else 0
    return date(y, int(p[1]), int(p[2])).isoformat()

def number(value):
    s = str(value).strip().replace(",","")
    if s in {"","--","-","X","nan","None"} or s.startswith("X"):
        return None
    return float(s)

def fetch_month(session, month, retries=6):
    params = {"date":month.strftime("%Y%m01"),"stockNo":SYMBOL,"response":"json"}
    last = None
    for attempt in range(retries):
        try:
            r = session.get(URL, params=params, timeout=(10,45),
                            headers={"User-Agent":"Mozilla/5.0 0050-official-research"})
            r.raise_for_status()
            payload = r.json()
            if payload.get("stat") == "很抱歉，沒有符合條件的資料!":
                return []
            if payload.get("stat") != "OK":
                raise RuntimeError(f"TWSE status={payload.get('stat')}")
            fields = payload.get("fields", [])
            rows = []
            for raw in payload.get("data", []):
                item = dict(zip(fields, raw))
                rows.append({
                    "symbol":SYMBOL, "name":NAME,
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
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(min(30, 2**attempt) + random.random())
    raise RuntimeError(f"{SYMBOL} {month:%Y-%m} failed after {retries} attempts: {last}")

def collect(output: Path, start: str, end: str, pause: float):
    lo, hi = date.fromisoformat(start), date.fromisoformat(end)
    if lo > hi:
        raise ValueError("start date must be <= end date")
    months = pd.date_range(lo.replace(day=1), hi.replace(day=1), freq="MS")
    output.mkdir(parents=True, exist_ok=True)
    bars_dir = output / "bars"; bars_dir.mkdir(exist_ok=True)
    path = bars_dir / f"{SYMBOL}.csv"
    old = pd.read_csv(path, dtype={"symbol":str}) if path.exists() else pd.DataFrame()
    rows = []
    session = requests.Session()
    for i, month in enumerate(months, 1):
        rows.extend(fetch_month(session, month))
        if pause:
            time.sleep(pause)
        if i % 24 == 0:
            print(f"[PROGRESS] {SYMBOL} {i}/{len(months)} months", flush=True)
    fresh = pd.DataFrame(rows)
    if not fresh.empty:
        fresh = fresh.drop_duplicates(["symbol","date"], keep="last")
        fresh = fresh[(fresh.date >= start) & (fresh.date <= end)]
        if not old.empty:
            fresh = pd.concat([old[~old.date.astype(str).isin(set(fresh.date.astype(str)))], fresh], ignore_index=True)
    elif not old.empty:
        fresh = old
    if fresh.empty:
        raise RuntimeError("No official TWSE 0050 daily data returned")
    fresh["date"] = pd.to_datetime(fresh.date, errors="coerce").dt.strftime("%Y-%m-%d")
    for c in ["volume","turnover","open","high","low","close"]:
        fresh[c] = pd.to_numeric(fresh[c], errors="coerce")
    valid = (fresh[["open","high","low","close"]].gt(0).all(axis=1)
             & (fresh.volume >= 0) & (fresh.turnover >= 0)
             & (fresh.high >= fresh[["open","close","low"]].max(axis=1))
             & (fresh.low <= fresh[["open","close","high"]].min(axis=1)))
    fresh = fresh.loc[valid].drop_duplicates(["symbol","date"]).sort_values("date").reset_index(drop=True)
    fresh.to_csv(path, index=False, float_format="%.6f")
    manifest = {
        "source":"Taiwan Stock Exchange official RWD monthly STOCK_DAY",
        "endpoint":URL, "symbol":SYMBOL, "name":NAME,
        "collected_at_taipei":datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
        "requested_period":[start,end], "rows":int(len(fresh)),
        "first_date":str(fresh.date.min()), "last_date":str(fresh.date.max()),
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "collection_complete":bool(len(fresh) > 1000),
        "price_basis":"raw unadjusted official exchange OHLC; execution-price candidate only",
        "volume_turnover":"official TWSE reported daily shares and traded value",
        "critical_corporate_action":"0050 completed a 4-for-1 split and resumed trading 2025-06-18; do not concatenate raw pre/post split prices for return calculations without applying verified split factor.",
        "limitations":[
            "This artifact contains raw daily prices only; no dividends, split-adjusted indicators, official price limits, or total-return series.",
            "The 2025-06-18 4-for-1 split must be independently reconciled before cross-split return or strategy metrics are calculated.",
            "Raw execution prices must remain unchanged; corporate-action-adjusted signal prices must be stored in separate columns."
        ]
    }
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    if not manifest["collection_complete"]:
        raise RuntimeError("0050 history unexpectedly short; collection not accepted")
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="data/0050_official_raw")
    p.add_argument("--start-date",default="2010-01-04")
    p.add_argument("--end-date",default=date.today().isoformat())
    p.add_argument("--pause-seconds",type=float,default=.35)
    a=p.parse_args()
    print(json.dumps(collect(Path(a.output),a.start_date,a.end_date,a.pause_seconds),ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
