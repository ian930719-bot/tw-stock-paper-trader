"""Probe official MOPS issuer announcements for historical TWSE capital-reduction terms."""
from __future__ import annotations
import json,time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

CASES=[
 {"symbol":"8261","roc_year":"104","event_date":"2015-09-11"},
 {"symbol":"2327","roc_year":"105","event_date":"2016-08-15"},
 {"symbol":"2327","roc_year":"106","event_date":"2017-08-18"},
 {"symbol":"2327","roc_year":"111","event_date":"2022-10-31"},
 {"symbol":"2327","roc_year":"114","event_date":"2025-08-25"},
 {"symbol":"2412","roc_year":"100","event_date":"2011-01-25"},
]
HOSTS=["https://mops.twse.com.tw","https://mopsov.twse.com.tw"]

def parse_html(text):
 soup=BeautifulSoup(text,"html.parser")
 tables=[]
 for table in soup.find_all("table")[:20]:
  rows=[]
  for tr in table.find_all("tr"):
   cells=[c.get_text(" ",strip=True) for c in tr.find_all(["th","td"])]
   links=[{"text":a.get_text(" ",strip=True),"href":a.get("href"),"onclick":a.get("onclick")} for a in tr.find_all("a")]
   if cells or links: rows.append({"cells":cells,"links":links})
  if rows:tables.append(rows[:200])
 links=[{"text":a.get_text(" ",strip=True),"href":a.get("href"),"onclick":a.get("onclick")} for a in soup.find_all("a")]
 return {"title":soup.title.get_text(" ",strip=True) if soup.title else None,"tables":tables,"links":links[:300]}

def main():
 s=requests.Session()
 s.headers.update({"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
  "Referer":"https://mops.twse.com.tw/mops/web/index","Origin":"https://mops.twse.com.tw"})
 out=Path("artifacts/twse_reduction_detail_probe");out.mkdir(parents=True,exist_ok=True)
 results=[]
 for case in CASES:
  for host in HOSTS:
   url=host+"/mops/web/t05st01"
   params={"firstin":"true","co_id":case["symbol"],"year":case["roc_year"]}
   try:
    r=s.get(url,params=params,timeout=(15,75));r.raise_for_status();r.encoding="utf-8"
    parsed=parse_html(r.text)
    relevant=[]
    for table in parsed["tables"]:
     for row in table:
      joined=" ".join(row["cells"])
      if any(k in joined for k in ["減資","換發","退還股款","新股"]):
       relevant.append(row)
    results.append({"case":case,"host":host,"url":r.url,"status":r.status_code,"length":len(r.text),
      "title":parsed["title"],"relevant_rows":relevant,"links":parsed["links"][:100]})
    (out/f"mops_t05st01_{case['symbol']}_{case['roc_year']}_{'ov' if 'mopsov' in host else 'legacy'}.html").write_text(r.text,encoding="utf-8")
    break
   except Exception as e:
    results.append({"case":case,"host":host,"url":url,"error":str(e)})
  time.sleep(0.5)
 # Probe the historical TWSE reduction table HTML for anchor/detail structure.
 for label,start,end in [("2015","20150801","20150930"),("2016","20160801","20160831"),("2025","20250801","20250831")]:
  url="https://www.twse.com.tw/exchangeReport/TWTAUU"
  params={"response":"html","startDate":start,"endDate":end}
  try:
   r=s.get(url,params=params,timeout=(15,90));r.raise_for_status();r.encoding="utf-8"
   parsed=parse_html(r.text)
   results.append({"case":{"label":label,"start":start,"end":end},"host":"TWSE","url":r.url,"status":r.status_code,
    "length":len(r.text),"title":parsed["title"],"tables":parsed["tables"][:5],"links":parsed["links"][:100]})
   (out/f"twse_twta uu_{label}.html".replace(" ","")).write_text(r.text,encoding="utf-8")
  except Exception as e:results.append({"case":{"label":label},"host":"TWSE","url":url,"error":str(e)})
  time.sleep(0.5)
 (out/"probe.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(results,ensure_ascii=False,indent=2)[:30000])

if __name__=="__main__":main()
