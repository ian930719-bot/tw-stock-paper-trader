"""TSMC 2330 retrospective swing episode path study (descriptive, not a trading system).

Uses future-confirmed 10% zigzag pivots only to define historical episodes. For each
pivot, measures indicators at 1/3/5/10/20 sessions BEFORE the pivot and compares
them with one same-calendar-year control date matched on prior 60-day trend and ATR.
All features at each snapshot use data available by that snapshot date.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

LAGS = (20, 10, 5, 3, 1)
FEATURES = ("rsi", "macdh_pct", "atrp", "ret5", "ret10", "ret20", "ret40",
            "dd20", "dd40", "ma20dev", "ma60dev", "vol5", "vol20", "bbpct",
            "bbwidth", "rsi_chg5", "macdh_chg5", "vol_chg5", "atr_chg5")


def load_features(path):
    d = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    c, h, l, v = d.close, d.high, d.low, d.volume
    ch = c.diff()
    up = ch.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-ch.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    d["macdh_pct"] = hist / c
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d["atrp"] = tr.rolling(14).mean() / c
    for n in (5, 10, 20, 40, 60):
        d[f"ret{n}"] = c.pct_change(n)
        d[f"ma{n}dev"] = c / c.rolling(n).mean() - 1
        d[f"vol{n}"] = v / v.rolling(n).mean()
    d["dd20"] = c / h.rolling(20).max() - 1
    d["dd40"] = c / h.rolling(40).max() - 1
    d["bbpct"] = (c-c.rolling(20).mean()) / (2*c.rolling(20).std())
    d["bbwidth"] = 4*c.rolling(20).std() / c.rolling(20).mean()
    d["rsi_chg5"] = d.rsi.diff(5)
    d["macdh_chg5"] = d.macdh_pct.diff(5)
    d["vol_chg5"] = d.vol5.diff(5)
    d["atr_chg5"] = d.atrp.pct_change(5)
    return d


def zigzag_pivots(d, reversal=0.10):
    """Retrospective pivot labels: close confirms reversal, pivot is prior intraday extreme."""
    hi, lo, close = d.high.to_numpy(float), d.low.to_numpy(float), d.close.to_numpy(float)
    state, extreme, ix, out = "seek_high", float(hi[0]), 0, []
    for i in range(1, len(d)):
        if state == "seek_high":
            if hi[i] > extreme:
                extreme, ix = hi[i], i
            if close[i] <= extreme * (1-reversal):
                out.append({"idx": ix, "type": "high", "price": extreme, "confirmed_idx": i})
                state, extreme, ix = "seek_low", float(lo[i]), i
        else:
            if lo[i] < extreme:
                extreme, ix = lo[i], i
            if close[i] >= extreme * (1+reversal):
                out.append({"idx": ix, "type": "low", "price": extreme, "confirmed_idx": i})
                state, extreme, ix = "seek_high", float(hi[i]), i
    return out


def run(input_csv, output_dir):
    d = load_features(input_csv)
    pivots = [p for p in zigzag_pivots(d, .10) if p["idx"] >= 65]
    pivot_indices = np.array([p["idx"] for p in pivots], dtype=int)
    control_mask = np.ones(len(d), dtype=bool)
    for ix in pivot_indices:
        control_mask[max(0, ix-35):min(len(d), ix+36)] = False
    controls = np.flatnonzero(control_mask)
    # One unique control per pivot, same year, matched on pre-event 60d trend and ATR.
    used, pairs = set(), []
    for p in pivots:
        ix = p["idx"]
        year = d.date.iloc[ix].year
        cand = controls[d.date.iloc[controls].dt.year.to_numpy() == year]
        cand = np.array([j for j in cand if j not in used and
                         pd.notna(d.loc[j, "ma60dev"]) and pd.notna(d.loc[j, "atrp"])])
        if not len(cand):
            continue
        dist = ((d.loc[cand, "ma60dev"].to_numpy()-d.loc[ix, "ma60dev"])/.08)**2
        dist += ((d.loc[cand, "atrp"].to_numpy()-d.loc[ix, "atrp"])/.01)**2
        cj = int(cand[np.argmin(dist)])
        used.add(cj)
        pairs.append({"pivot": p["type"], "pivot_date": str(d.date.iloc[ix].date()),
                      "control_date": str(d.date.iloc[cj].date()), "pivot_idx": ix,
                      "control_idx": cj, "year": year})
    long = []
    for pair in pairs:
        for lag in LAGS:
            for feat in FEATURES:
                ei, ci = pair["pivot_idx"]-lag, pair["control_idx"]-lag
                if ei >= 0 and ci >= 0 and pd.notna(d.loc[ei, feat]) and pd.notna(d.loc[ci, feat]):
                    long.append({"pivot": pair["pivot"], "lag": lag, "feature": feat,
                                 "event_value": float(d.loc[ei, feat]),
                                 "control_value": float(d.loc[ci, feat]),
                                 "pivot_date": pair["pivot_date"], "control_date": pair["control_date"]})
    values = pd.DataFrame(long)
    results = []
    for (typ, lag, feat), g in values.groupby(["pivot", "lag", "feature"]):
        a, b = g.event_value.to_numpy(), g.control_value.to_numpy()
        diff = a-b
        try:
            pval = float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
        except ValueError:
            pval = 1.0
        median_diff = float(np.median(diff))
        pair_auc = float(np.mean(diff > 0) + .5*np.mean(diff == 0))
        if median_diff < 0:
            pair_auc = 1-pair_auc
        results.append({"pivot": typ, "lag": int(lag), "feature": feat, "pairs": len(g),
                        "event_median": float(np.median(a)), "control_median": float(np.median(b)),
                        "median_paired_diff": median_diff, "directional_pair_auc": pair_auc,
                        "wilcoxon_p": pval})
    stats = pd.DataFrame(results)
    stats["q_bh"] = np.nan
    for _, group in stats.groupby(["pivot", "lag"]):
        pvals = group.wilcoxon_p.to_numpy()
        order = np.argsort(pvals)
        q = np.empty(len(pvals)); running = 1.0
        for rank in range(len(order)-1, -1, -1):
            k = order[rank]
            running = min(running, pvals[k]*len(pvals)/(rank+1))
            q[k] = running
        stats.loc[group.index, "q_bh"] = q
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out/"matched_feature_path_effects.csv", index=False)
    values.to_csv(out/"pivot_feature_paths.csv", index=False)
    pd.DataFrame(pairs).to_csv(out/"matched_episode_controls.csv", index=False)
    summary = {"symbol": "2330", "rows": len(d), "data_start": str(d.date.min().date()),
               "data_end": str(d.date.max().date()), "source": "Yahoo Finance auto_adjust=True",
               "pivot_definition": "10% zigzag close confirmation; pivot date is prior intraday extreme",
               "pivots": {k: sum(x["type"] == k for x in pivots) for k in ("low", "high")},
               "matched_pairs": {k: sum(x["pivot"] == k for x in pairs) for k in ("low", "high")},
               "lags": list(LAGS), "features": list(FEATURES),
               "warning": "Retrospective pivots are labels, not signals. Controls are matched by year, MA60 deviation and ATR%; only 17 matched pairs per side in this sample. Multiple testing corrected within each pivot/lag family using BH. Adjusted Yahoo data is not official execution data.",
               "top_effects": {}}
    for typ in ("low", "high"):
        for lag in LAGS:
            z = stats[(stats.pivot == typ) & (stats.lag == lag)].sort_values(["q_bh", "directional_pair_auc"], ascending=[True, False]).head(8)
            summary["top_effects"][f"{typ}_{lag}d"] = z.to_dict("records")
    (out/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/top10_adjusted_prices/bars/2330.csv")
    parser.add_argument("--output", default="artifacts/tsmc_swing_v4")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), ensure_ascii=False, indent=2))
