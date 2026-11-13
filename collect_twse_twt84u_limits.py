"""Collect TWSE official daily price-limit rows for point-in-time top-ten symbols.

Uses the date-parameterized official exchangeReport/TWT84U endpoint, not
reconstructed previous-close formulas.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import random
import re
import time
from pathlib import Path
import pandas as pd
import requests

URL = "https://www.twse.com.tw/exchangeReport/TWT84U"

def clean(value):
    return re.sub(r"\s+", "", html.unescape(re.sub(r"<[^>]*>", "", str(value or ""))))

def number(value):
    s = clean(value).replace(",", "").replace("−", "-").replace("－", "-")
    if s in {"", "--", "---", "----", "X", "nan", "None"}:
        return None
    try:
        x = float(s)
        return x if x > 0 else None
    except (TypeError, ValueError):
        return None

def fetch_day(session, day, targets, retries=8):
    params = {"date": pd.Timestamp(day).strftime("%Y%m%d"), "response": "json"}
    last_error = None
    for attempt in range(retries):
        try:
            response = session.get(
                URL, params=params, timeout=(10, 45),
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json",
                         "Referer": "https://www.twse.com.tw/"}
            )
            response.raise_for_status()
            payload = response.json()
            if str(payload.get("stat", "")).upper() != "OK":
                raise RuntimeError(f"TWSE TWT84U stat={payload.get('stat')}")
            fields = [clean(x) for x in payload.get("fields", [])]
            def index(*names):
                for name in names:
                    if clean(name) in fields:
                        return fields.index(clean(name))
                raise RuntimeError(f"Missing official TWSE TWT84U field {names}; got {fields}")
            code_i = index("證券代號", "證券代碼", "股票代號")
            up_i = index("漲停價")
            ref_i = index("開盤競價基準")
            down_i = index("跌停價")
            last_i = next((i for i,x in enumerate(fields) if x=="最近成交日"),None)
            out = {}
            for row in payload.get("data", []):
                if len(row) <= max(code_i,up_i,ref_i,down_i):
                    continue
                symbol = clean(row[code_i])
                if symbol not in targets:
                    continue
                if symbol in out:
                    raise RuntimeError(f"Duplicate TWSE TWT84U code {symbol} on {day}")
                out[symbol] = {
                    "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                    "stock_id": symbol,
                    "official_limit_up": number(row[up_i]),
                    "official_reference_price": number(row[ref_i]),
                    "official_limit_down": number(row[down_i]),
                    "last_trading_day": clean(row[last_i]) if last_i is not None else None,
                    "source": "TWSE official exchangeReport/TWT84U",
                    "source_url": response.url,
                }
            return out
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(45,2**attempt)+random.uniform(0.25,1.25))
    raise RuntimeError(f"TWSE TWT84U {day}: {last_error}")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--daily-dir",required=True)
    parser.add_argument("--universe-file",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--pause",type=float,default=0.30)
    args=parser.parse_args()
    daily_dir=Path(args.daily_dir)
    universe=pd.read_csv(args.universe_file,dtype={"symbol":str,"stock_id":str})
    target_ids=set(universe.loc[universe.market.astype(str).str.upper()=="TWSE","stock_id"].astype(str))
    if not target_ids: raise RuntimeError("No TWSE symbols in point-in-time top-ten universe")
    daily_files=sorted(daily_dir.glob("*.csv"))
    if not daily_files: raise RuntimeError(f"No official daily marketwide files in {daily_dir}")
    output_path=Path(args.output);output_path.parent.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    records=[];missing=[];queried=[]
    for i,path in enumerate(daily_files,1):
        day=path.stem
        market=pd.read_csv(path,dtype={"stock_id":str})
        expected=set(market.loc[
            (market.market.astype(str).str.upper()=="TWSE")
            & market.stock_id.astype(str).isin(target_ids),"stock_id"
        ].astype(str))
        if not expected: continue
        rows=fetch_day(session,day,expected)
        queried.append(day)
        for symbol in sorted(expected):
            row=rows.get(symbol)
            if row is None:
                missing.append({"date":day,"stock_id":symbol,"reason":"symbol missing from official TWT84U response"})
                records.append({"date":day,"stock_id":symbol,"official_limit_up":None,
                    "official_reference_price":None,"official_limit_down":None,
                    "last_trading_day":None,"source":"TWSE official TWT84U; symbol row missing","source_url":None})
            else:
                if any(row[k] is None for k in ("official_limit_up","official_reference_price","official_limit_down")):
                    missing.append({"date":day,"stock_id":symbol,"reason":"official TWT84U price-limit/reference field blank"})
                records.append(row)
        if i%25==0 or i==len(daily_files):
            pd.DataFrame(records).to_csv(output_path,index=False)
            print(f"PROGRESS files={i}/{len(daily_files)} dates={len(queried)} rows={len(records)} missing={len(missing)}",flush=True)
        time.sleep(args.pause)
    frame=pd.DataFrame(records)
    if frame.empty: raise RuntimeError("No TWSE TWT84U official limits collected")
    if frame.duplicated(["date","stock_id"]).any(): raise RuntimeError("Duplicate TWSE date/symbol rows")
    frame.to_csv(output_path,index=False)
    manifest={
      "source":"TWSE official exchangeReport/TWT84U",
      "fields":["漲停價","開盤競價基準","跌停價"],
      "first_date":str(frame.date.min()),"last_date":str(frame.date.max()),
      "queried_dates":len(queried),"rows":len(frame),"symbols":int(frame.stock_id.nunique()),
      "missing_symbol_or_limit_rows":len(missing),"missing_examples":missing[:100],
      "sha256":hashlib.sha256(output_path.read_bytes()).hexdigest(),
      "complete":len(missing)==0,
      "note":"Official date-specific upper/lower limits and opening reference prices; no reconstructed price-limit formulas."
    }
    output_path.with_name("manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
