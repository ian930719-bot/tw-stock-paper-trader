"""Fallback verification for missing TPEx next-session price-limit rows.

Source is the official TPEx historical Daily Stock Quotes page. It exposes
next-day reference price, next-day upper limit and next-day lower limit.
This script only fills an explicitly listed missing key; it does not calculate
or reconstruct any limit prices.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import requests

CASES=[{"date":"2015-11-23","roc_date":"104/11/23","stock_id":"4147"}]
URL="https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php"

def main():
    session=requests.Session()
    session.headers.update({
      "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
      "Accept":"application/json,text/plain,*/*",
      "Referer":"https://www.tpex.org.tw/zh-tw/mainboard/trading/info/pricing.html"
    })
    rows=[];raw={}
    for case in CASES:
        params={"l":"zh-tw","o":"json","d":case["roc_date"],"s":"0,asc,0"}
        response=session.get(URL,params=params,timeout=(15,90))
        response.raise_for_status()
        payload=response.json()
        raw[case["date"]]=payload
        data=payload.get("aaData")
        if not isinstance(data,list):
            raise RuntimeError(f"TPEx historical daily quotes missing aaData for {case['date']}: {str(payload)[:1000]}")
        matches=[r for r in data if len(r)>=19 and str(r[0]).strip()==case["stock_id"]]
        if len(matches)!=1:
            raise RuntimeError(f"Expected one TPEx official daily quote row for {case['stock_id']} on {case['date']}; got {len(matches)}")
        r=matches[0]
        ref,up,down=[float(str(r[i]).replace(",","")) for i in (16,17,18)]
        if min(ref,up,down)<=0:
            raise RuntimeError(f"Invalid official TPEx next-day price fields for {case}: {r}")
        rows.append({
          "source_date":case["date"],"stock_id":case["stock_id"],
          "next_day_reference_price":ref,"next_day_limit_up":up,"next_day_limit_down":down,
          "source":"TPEx official historical Daily Stock Quotes (stk_quote_result.php)",
          "source_url":response.url
        })
    import pandas as pd
    out=Path("artifacts/tpex_missing_limit_fallback");out.mkdir(parents=True,exist_ok=True)
    csv=out/"tpex_missing_limit_fallback.csv"
    pd.DataFrame(rows).to_csv(csv,index=False)
    (out/"raw_probe.json").write_text(json.dumps(raw,ensure_ascii=False,indent=2),encoding="utf-8")
    manifest={
      "source":"TPEx official historical Daily Stock Quotes",
      "endpoint":URL,"rows":len(rows),"cases":CASES,
      "complete":len(rows)==len(CASES),
      "sha256":hashlib.sha256(csv.read_bytes()).hexdigest(),
      "note":"Exact official next-day quote fields; no formula-derived price limits."
    }
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"manifest":manifest,"rows":rows},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
