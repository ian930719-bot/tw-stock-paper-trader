"""Probe MOPS full CSV export behind t108sb27 HTML pagination."""
from __future__ import annotations
import json,io,hashlib
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

YEARS=[104,105,107,110,114]
TARGETS={"2888","2892","5880","3454","2823"}

def main():
 s=requests.Session()
 s.headers.update({"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
  "Referer":"https://mopsov.twse.com.tw/mops/web/index","Origin":"https://mopsov.twse.com.tw"})
 out=Path("artifacts/mops_full_csv_export_probe");out.mkdir(parents=True,exist_ok=True)
 reports=[]
 for year in YEARS:
  params={"step":"1","firstin":"true","off":"1","TYPEK":"sii","co_id_1":"","co_id_2":"",
   "year":str(year),"month":"","b_date":"","e_date":"","type":"1","rulesubmit2":"搜尋"}
  r=s.get("https://mopsov.twse.com.tw/mops/web/ajax_t108sb27",params=params,timeout=(15,90))
  r.raise_for_status();r.encoding="utf-8"
  soup=BeautifulSoup(r.text,"html.parser")
  node=soup.find("input",{"name":"filename"})
  filename=node.get("value") if node else None
  report={"roc_year":year,"html_url":r.url,"html_length":len(r.text),
   "html_table_rows":[len(t.find_all("tr")) for t in soup.find_all("table")],
   "filename":filename}
  if not filename:
   report["csv_error"]="No hidden filename input in MOPS HTML"
   reports.append(report);continue
  csv_response=s.post("https://mopsov.twse.com.tw/server-java/t105sb02",
   data={"firstin":"true","step":"10","filename":filename},timeout=(15,120),
   headers={"Referer":r.url})
  report.update({"csv_url":csv_response.url,"csv_status":csv_response.status_code,
   "csv_content_type":csv_response.headers.get("content-type"),
   "csv_length":len(csv_response.content),"csv_start":csv_response.content[:300].decode("utf-8",errors="replace")})
  if csv_response.status_code==200 and len(csv_response.content)>100 and b"FOR SECURITY REASONS" not in csv_response.content:
   path=out/f"mops_t108sb27_sii_{year}_full.csv"
   path.write_bytes(csv_response.content)
   report["csv_sha256"]=hashlib.sha256(csv_response.content).hexdigest()
   try:
    frame=pd.read_csv(io.BytesIO(csv_response.content),dtype={"公司代號":str},encoding="utf-8-sig")
    report["parsed_shape"]=list(frame.shape)
    report["columns"]=list(frame.columns)
    code=next((c for c in frame.columns if str(c).strip()=="公司代號"),None)
    if code:
     report["target_rows"]=frame[frame[code].astype(str).str.strip().isin(TARGETS)].fillna("").to_dict("records")
    else:
     # Fallback: parse without headers and locate target code rows.
     raw=pd.read_csv(io.BytesIO(csv_response.content),header=None,dtype=str,encoding="utf-8-sig")
     report["raw_shape"]=list(raw.shape)
     report["target_rows_raw"]=raw[raw.astype(str).apply(lambda c:c.str.strip().isin(TARGETS)).any(axis=1)].fillna("").values.tolist()[:100]
   except Exception as e:
    report["parse_error"]=str(e)
  reports.append(report)
  print(json.dumps(report,ensure_ascii=False)[:10000],flush=True)
 (out/"manifest.json").write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(reports,ensure_ascii=False,indent=2)[:30000])

if __name__=="__main__":main()
