"""V18: audit yearly return and daily P&L concentration of saved strategy equity curves."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

INITIAL=1_000_000.0
STRATEGIES=("mom20_breadth60","mom20_breadth50","mom20","annual_top20")

def audit(path,name):
    d=pd.read_csv(path,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if not {"date","equity"}.issubset(d.columns): raise ValueError(f"{path} must contain date,equity")
    d["equity"]=pd.to_numeric(d.equity,errors="coerce")
    d=d.dropna(subset=["date","equity"])
    d["pnl"]=d.equity.diff()
    d.loc[0,"pnl"]=d.equity.iloc[0]-INITIAL
    rows=[]; previous=INITIAL
    for year,g in d.groupby(d.date.dt.year,sort=True):
        ending=float(g.equity.iloc[-1])
        rows.append({"year":int(year),"return_pct":(ending/previous-1)*100,
                     "pnl":ending-previous,"daily_rows":len(g)})
        previous=ending
    total=float(d.equity.iloc[-1]-INITIAL)
    top5=float(d.pnl.nlargest(5).sum()/total*100) if total else None
    top10=float(d.pnl.nlargest(10).sum()/total*100) if total else None
    through=d[d.date<=pd.Timestamp("2025-12-31")]
    days=(through.date.iloc[-1]-through.date.iloc[0]).days
    cagr25=((float(through.equity.iloc[-1])/INITIAL)**(365.25/days)-1)*100 if days>0 else None
    last_year=int(d.date.dt.year.max())
    year_start=d[d.date.dt.year==last_year]
    start_equity=float(d[d.date.dt.year<last_year].equity.iloc[-1]) if (d.date.dt.year<last_year).any() else INITIAL
    ytd=(float(year_start.equity.iloc[-1])/start_equity-1)*100
    return {"strategy":name,"data_start":str(d.date.iloc[0].date()),"data_end":str(d.date.iloc[-1].date()),
            "final_equity":float(d.equity.iloc[-1]),"cagr_through_2025_pct":cagr25,
            "latest_partial_year":last_year,"latest_partial_year_return_pct":ytd,
            "top5_daily_pnl_share_pct":top5,"top10_daily_pnl_share_pct":top10,"yearly":rows}

def main(curves_dir,output):
    root=Path(curves_dir);results=[]
    for name in STRATEGIES:
        p=root/f"test_equity_{name}.csv"
        if p.exists():results.append(audit(p,name))
    if not results:raise FileNotFoundError(f"No expected equity curves found in {root}")
    summary={"version":"V18","initial_capital":INITIAL,"results":results,
      "decision_rule":["Do not select a strategy based on a single high-CAGR test interval.",
      "Require positive net CAGR in research, validation and test intervals.",
      "Report each calendar year's return and the latest partial year's contribution.",
      "Report top-five and top-ten daily P&L contribution; a high concentration is a fragility warning.",
      "A historical pass is not sufficient for deployment; require forward paper trading and data-integrity signoff."],
      "limitations":["This audits saved equity curves; it does not validate source price accuracy or fills.",
      "Daily P&L concentration is descriptive and does not replace trade-level attribution.",
      "Existing 2022-2026 test window has been examined and is not a pristine holdout."]}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--curves-dir",default="artifacts/top20_v16");p.add_argument("--output",default="artifacts/top20_v18_return_concentration/summary.json")
    a=p.parse_args();main(a.curves_dir,a.output)
