"""Probe official TPEx historical daily-quote routes for a missing limit row."""
from __future__ import annotations
import json,hashlib
from pathlib import Path
import requests,pandas as pd

CASES=[{"date":"2015-11-23","roc_date":"104/11/23","stock_id":"4147"}]
LEGACY="https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php"
ROUTES=[
 ("legacy_roc_d",LEGACY,{"l":"zh-tw","o":"json","d":"104/11/23","s":"0,asc,0"}),
 ("legacy_ad_d",LEGACY,{"l":"zh-tw","o":"json","d":"2015/11/23","s":"0,asc,0"}),
 ("legacy_compact_d",LEGACY,{"l":"zh-tw","o":"json","d":"20151123","s":"0,asc,0"}),
 ("legacy_date",LEGACY,{"l":"zh-tw","o":"json","date":"2015/11/23","s":"0,asc,0"}),
 ("new_daily_quotes","https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes",{"date":"2015/11/23","type":"EW","response":"json"}),
 ("openapi_iso","https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",{"date":"2015-11-23"}),
 ("openapi_compact","https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",{"date":"20151123"}),
 ("openapi_roc","https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",{"date":"104/11/23"}),
]

def extract(payload,code):
    candidates=[]
    if isinstance(payload,dict):
        if isinstance(payload.get("aaData"),list): candidates.extend(payload["aaData"])
        for table in payload.get("tables",[]) if isinstance(payload.get("tables"),list) else []:
            if isinstance(table,dict) and isinstance(table.get("data"),list):
                candidates.extend(table["data"])
        if isinstance(payload.get("data"),list): candidates.extend(payload["data"])
    elif isinstance(payload,list):
        candidates.extend(payload)
    out=[]
    for row in candidates:
        if isinstance(row,dict):
            codev=str(row.get("代號",row.get("Code",row.get("stock_id","")))).strip()
            if codev==code:out.append(row)
        elif isinstance(row,(list,tuple)) and len(row)>=17 and str(row[0]).strip()==code:
            out.append(row)
    return out

def main():
    s=requests.Session()
    s.headers.update({"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
      "Accept":"application/json,text/plain,*/*","Referer":"https://www.tpex.org.tw/zh-tw/mainboard/trading/info/pricing.html"})
    probes=[];resolved=[]
    for case in CASES:
      for name,url,params in ROUTES:
        try:
          r=s.get(url,params=params,timeout=(10,45))
          r.raise_for_status()
          try:payload=r.json()
          except Exception:payload={"text":r.text[:2000]}
          matches=extract(payload,case["stock_id"])
          fields=[]
          if isinstance(payload,dict):
            if payload.get("tables") and isinstance(payload["tables"],list) and isinstance(payload["tables"][0],dict):
              fields=payload["tables"][0].get("fields",[])
            else:fields=payload.get("fields",[])
          record={"case":case,"route":name,"url":r.url,"status":r.status_code,
             "payload_keys":list(payload.keys()) if isinstance(payload,dict) else None,
             "payload_date":payload.get("date") if isinstance(payload,dict) else None,
             "table_fields":fields,"match_count":len(matches),
             "sample_match":matches[0] if matches else None}
          probes.append(record)
          if len(matches)==1:
            row=matches[0]
            if isinstance(row,dict):
              def getv(*keys):
                for k in keys:
                  if k in row:return row[k]
                return None
              ref=getv("次日 參考價","次日參考價","next_day_reference_price")
              up=getv("次日 漲停價","次日漲停價","next_day_limit_up")
              down=getv("次日 跌停價","次日跌停價","next_day_limit_down")
            else:
              norm=[str(v).replace(" ","").strip() for v in fields]
              if all(k in norm for k in ["次日參考價","次日漲停價","次日跌停價"]):
                ref,up,down=[row[norm.index(k)] for k in ["次日參考價","次日漲停價","次日跌停價"]]
              elif len(row)>=19:
                ref,up,down=row[16],row[17],row[18]
              else:
                ref,up,down=row[14],row[15],row[16]
            def num(v):
              return float(str(v).replace(",","").strip()) if v not in (None,"","--","-") else None
            ref,up,down=map(num,(ref,up,down))
            if all(v is not None and v>0 for v in (ref,up,down)):
              resolved.append({"source_date":case["date"],"stock_id":case["stock_id"],
                "next_day_reference_price":ref,"next_day_limit_up":up,"next_day_limit_down":down,
                "source":f"TPEx official historical daily quote fallback ({name})","source_url":r.url})
        except Exception as e:
          probes.append({"case":case,"route":name,"url":url,"error":str(e)})
    out=Path("artifacts/tpex_missing_limit_fallback");out.mkdir(parents=True,exist_ok=True)
    (out/"probe_results.json").write_text(json.dumps(probes,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame(resolved,columns=["source_date","stock_id","next_day_reference_price","next_day_limit_up","next_day_limit_down","source","source_url"]).drop_duplicates(["source_date","stock_id"]).to_csv(out/"tpex_missing_limit_fallback.csv",index=False)
    manifest={"source":"TPEx official historical daily quotes","cases":CASES,"probe_attempts":len(probes),
      "resolved_rows":len(resolved),"complete":len(resolved)==len(CASES),
      "sha256":hashlib.sha256((out/"tpex_missing_limit_fallback.csv").read_bytes()).hexdigest(),
      "note":"Only official published next-day quote fields are accepted; no price-limit formula reconstruction.",
      "probes":probes}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"complete":manifest["complete"],"resolved_rows":resolved,
      "probes":[{k:v for k,v in p.items() if k not in {"sample_match"}} for p in probes]},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
