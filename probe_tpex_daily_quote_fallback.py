"""Probe official TPEx historical daily-quote routes for a missing limit row."""
from __future__ import annotations
import json,hashlib
from pathlib import Path
import requests,pandas as pd

CASES=[
 {"date":"2015-11-23","roc_date":"104/11/23","stock_id":"4147"},
 {"date":"2015-11-24","roc_date":"104/11/24","stock_id":"4147"},
 {"date":"2015-11-25","roc_date":"104/11/25","stock_id":"4147"},
 {"date":"2015-11-26","roc_date":"104/11/26","stock_id":"4147"},
 {"date":"2015-11-27","roc_date":"104/11/27","stock_id":"4147"},
]
ROUTES=[
 ("new_daily_quotes","https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes",{"type":"EW","response":"json"}),
 ("no_limit_page","https://www.tpex.org.tw/www/zh-tw/afterTrading/noLimit",{"type":"EW","response":"json"}),
 ("ipo_no_limit_page","https://www.tpex.org.tw/www/zh-tw/afterTrading/ipoNoLimit",{"type":"EW","response":"json"}),
 ("openapi_no_limit_iso","https://www.tpex.org.tw/openapi/v1/tpex_ipo_no_limit",{"date":"2015-11-23"}),
 ("openapi_no_limit_compact","https://www.tpex.org.tw/openapi/v1/tpex_ipo_no_limit",{"date":"20151123"}),
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
          route_params=dict(params)
          if name=="new_daily_quotes" or "no_limit" in name:
            route_params["date"]=case["roc_date"] if name in {"new_daily_quotes","no_limit_page","ipo_no_limit_page"} else params.get("date")
          r=s.get(url,params=route_params,timeout=(10,45))
          r.raise_for_status()
          try:payload=r.json()
          except Exception:payload={"text":r.text[:2000]}
          matches=extract(payload,case["stock_id"])
          if isinstance(payload,dict) and payload.get("date") and str(payload.get("date"))!=case["date"].replace("-",""):
            matches=[]
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
          if len(matches)==1 and name=="new_daily_quotes":
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
              no_limit=(up>=9995 and down<=0.01)
              resolved.append({"source_date":case["date"],"stock_id":case["stock_id"],
                "next_day_reference_price":ref,
                "next_day_limit_up":None if no_limit else up,
                "next_day_limit_down":None if no_limit else down,
                "next_day_limit_status":"no_limit" if no_limit else "limited",
                "source":f"TPEx official historical daily quote fallback ({name})","source_url":r.url})
        except Exception as e:
          probes.append({"case":case,"route":name,"url":url,"error":str(e)})
    out=Path("artifacts/tpex_missing_limit_fallback");out.mkdir(parents=True,exist_ok=True)
    (out/"probe_results.json").write_text(json.dumps(probes,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame(resolved,columns=["source_date","stock_id","next_day_reference_price","next_day_limit_up","next_day_limit_down","next_day_limit_status","source","source_url"]).drop_duplicates(["source_date","stock_id"]).to_csv(out/"tpex_missing_limit_fallback.csv",index=False)
    manifest={"source":"TPEx official historical daily quotes","cases":CASES,"probe_attempts":len(probes),
      "resolved_rows":len(resolved),"complete":len(resolved)==len(CASES),
      "no_limit_rows":sum(1 for x in resolved if x.get("next_day_limit_status")=="no_limit"),
      "sha256":hashlib.sha256((out/"tpex_missing_limit_fallback.csv").read_bytes()).hexdigest(),
      "note":"Only official published next-day quote fields are accepted; no price-limit formula reconstruction.",
      "probes":probes}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"complete":manifest["complete"],"resolved_rows":resolved,
      "probes":[{k:v for k,v in p.items() if k not in {"sample_match"}} for p in probes]},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
