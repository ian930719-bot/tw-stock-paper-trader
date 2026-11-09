"""Per-security historical rule mining with a chronological holdout.

This is a discovery screen, not a portfolio backtest or a promise of profitability.
Signals are known at close t, entry is next-session open, and exits are close t+h.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

COST = 2 * 0.001425 + 0.003 + 2 * 0.001
HORIZONS = (5, 20, 60)
FEATURES = ("ret_20", "ret_60", "ret_120", "reversal_5", "dist_sma50",
            "dist_sma200", "breakout20", "breakout55", "relvol20",
            "atr_pct14", "vol_compress", "close_location")

def make_features(d):
    d = d.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    c = pd.to_numeric(d.close, errors="coerce").to_numpy(float)
    o = pd.to_numeric(d.open, errors="coerce").to_numpy(float)
    h = pd.to_numeric(d.high, errors="coerce").to_numpy(float)
    l = pd.to_numeric(d.low, errors="coerce").to_numpy(float)
    v = pd.to_numeric(d.volume, errors="coerce").to_numpy(float)
    s = pd.Series(c)
    f = pd.DataFrame(index=np.arange(len(d)))
    f["ret_20"] = s.pct_change(20).to_numpy()
    f["ret_60"] = s.pct_change(60).to_numpy()
    f["ret_120"] = s.pct_change(120).to_numpy()
    f["reversal_5"] = -s.pct_change(5).to_numpy()
    for n, name in ((50, "dist_sma50"), (200, "dist_sma200")):
        f[name] = c / s.rolling(n, min_periods=n).mean().to_numpy() - 1
    for n, name in ((20, "breakout20"), (55, "breakout55")):
        prior = pd.Series(h).shift(1).rolling(n, min_periods=n).max().to_numpy()
        f[name] = c / prior - 1
    f["relvol20"] = v / pd.Series(v).shift(1).rolling(20, min_periods=20).mean().to_numpy()
    prev = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h-l, np.abs(h-prev), np.abs(l-prev)]), axis=0)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().to_numpy()
    f["atr_pct14"] = atr / c
    rv20 = s.pct_change().rolling(20, min_periods=20).std(ddof=0).to_numpy()
    rv60 = s.pct_change().rolling(60, min_periods=60).std(ddof=0).to_numpy()
    f["vol_compress"] = rv20 / rv60
    f["close_location"] = (c-l) / np.where(h != l, h-l, np.nan)
    y = {}
    for n in HORIZONS:
        arr = np.full(len(c), np.nan)
        if len(c) > n:
            arr[:len(c)-n] = c[n:] / o[1:len(c)-n+1] - 1
        y[n] = arr
    valid = np.isfinite(f[list(FEATURES)].to_numpy()).all(axis=1)
    for n in HORIZONS:
        valid &= np.isfinite(y[n])
    return f.loc[valid].reset_index(drop=True), {n:a[valid] for n,a in y.items()}

def run(input_dir: Path, output_dir: Path, split: float = .70):
    meta_path = input_dir / "candidate_symbols.csv"
    meta = (pd.read_csv(meta_path, dtype={"symbol":str}).fillna("")
            if meta_path.exists() else pd.DataFrame(columns=["symbol","name","industry_category","candidate_source"]))
    meta = meta.drop_duplicates("symbol").set_index("symbol")
    bar_dir = input_dir / "bars" if (input_dir/"bars").exists() else input_dir
    paths = sorted(bar_dir.glob("*.csv"))
    rows, coverage = [], []
    for ix, path in enumerate(paths, 1):
        symbol = path.stem
        try:
            d = pd.read_csv(path)
            if len(d) < 500:
                coverage.append({"symbol":symbol,"rows":len(d),"status":"insufficient_history"})
                continue
            f, y = make_features(d)
            if len(f) < 450:
                coverage.append({"symbol":symbol,"rows":len(f),"status":"insufficient_valid_rows"})
                continue
            cut = int(len(f)*split)
            x = f.to_numpy(float)
            for j, feat in enumerate(FEATURES):
                a, b = x[:cut,j], x[cut:,j]
                for tail, q in (("high",.80),("low",.20)):
                    threshold = float(np.quantile(a,q))
                    mt = a >= threshold if tail == "high" else a <= threshold
                    mv = b >= threshold if tail == "high" else b <= threshold
                    if mt.sum() < 30 or mv.sum() < 15:
                        continue
                    for horizon in HORIZONS:
                        tr, te = y[horizon][:cut][mt]-COST, y[horizon][cut:][mv]-COST
                        neg = abs(np.sum(te[te<0]))
                        rows.append({
                            "symbol":symbol,"factor":feat,"tail":tail,"horizon_days":horizon,
                            "threshold_train":threshold,"train_signals":int(len(tr)),
                            "train_mean_net":float(np.mean(tr)),"train_win_rate":float(np.mean(tr>0)),
                            "test_signals":int(len(te)),"test_mean_net":float(np.mean(te)),
                            "test_median_net":float(np.median(te)),"test_win_rate":float(np.mean(te>0)),
                            "test_profit_factor":float(np.sum(te[te>0])/neg) if neg else None,
                            "test_compounded_signal_return":float(np.prod(1+te)-1)})
            coverage.append({"symbol":symbol,"rows":len(f),"status":"analysed"})
        except Exception as exc:
            coverage.append({"symbol":symbol,"rows":0,"status":"error:"+type(exc).__name__})
        if ix % 100 == 0:
            print(f"processed {ix}/{len(paths)}", flush=True)
    all_rules, cov = pd.DataFrame(rows), pd.DataFrame(coverage)
    if not all_rules.empty:
        all_rules["train_score"] = all_rules.train_mean_net * np.sqrt(all_rules.train_signals)
        # Select only on training data; never sort or select using holdout performance.
        top = (all_rules.sort_values(["symbol","train_score"],ascending=[True,False])
               .groupby("symbol",as_index=False).head(3))
        top = top.merge(meta.reset_index(),on="symbol",how="left")
        industry = (top.groupby("industry_category",dropna=False)
            .agg(stocks=("symbol","nunique"),rules=("factor","size"),
                 avg_test_net=("test_mean_net","mean"),median_test_net=("test_mean_net","median"),
                 positive_test_rule_rate=("test_mean_net",lambda z:float(np.mean(z>0))),
                 avg_test_win_rate=("test_win_rate","mean"))
            .reset_index().sort_values("avg_test_net",ascending=False))
    else:
        top, industry = pd.DataFrame(), pd.DataFrame()
    output_dir.mkdir(parents=True,exist_ok=True)
    all_rules.to_csv(output_dir/"all_stock_factor_rules.csv",index=False)
    top.to_csv(output_dir/"per_stock_top_rules.csv",index=False)
    industry.to_csv(output_dir/"industry_rule_summary.csv",index=False)
    cov.to_csv(output_dir/"stock_coverage.csv",index=False)
    manifest = {
        "candidate_symbols":int(len(meta)),
        "analysed_symbols":int((cov.status=="analysed").sum()) if not cov.empty else 0,
        "not_analysed_symbols":int((cov.status!="analysed").sum()) if not cov.empty else 0,
        "features":list(FEATURES),"horizons_days":list(HORIZONS),
        "split":"chronological 70% train / 30% holdout per symbol",
        "thresholds":"per-stock training-only 20th/80th percentile",
        "execution":"close signal, next-session open entry, close exit after horizon",
        "estimated_round_trip_cost":COST,
        "selection":"top 3 per symbol selected using training mean net return * sqrt(training signal count)",
        "limitations":["not a portfolio simulation; forward labels overlap",
          "industry label is latest available, not historical point-in-time classification",
          "candidate universe is not complete point-in-time history; delisted coverage incomplete",
          "corporate actions and exchange calendars not fully reconciled",
          "multiple-testing risk; holdout is a first screen, not proof of durable edge"]}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",default="artifacts/per_stock_rule_mining")
    parser.add_argument("--split",type=float,default=.70)
    args=parser.parse_args()
    if not .5 <= args.split <= .85: raise SystemExit("--split must be between 0.5 and 0.85")
    run(Path(args.input),Path(args.output),args.split)
