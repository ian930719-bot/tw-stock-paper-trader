"""Exploratory TSMC swing event study: indicators, divergences and regime stability.

Labels use future prices only to score events. Features are strictly causal through
the signal date. This is a research diagnostic, not an executable strategy.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["rsi", "macdh", "macdh_chg3", "atrp", "ret5", "ret10", "ret20",
            "ret40", "ddhi20", "ddhi40", "ma20dev", "ma60dev", "vol5",
            "vol20", "bull_div", "bear_div", "vol_climax_contract", "bb_pct",
            "bb_width", "bb_width_chg5"]
PERIODS = {"train": ("2010-01-01", "2020-12-31"),
           "validation": ("2021-01-01", "2023-12-31"),
           "historical_test": ("2024-01-01", "2026-11-10")}


def build_frame(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    c, h, l, v = d.close, d.high, d.low, d.volume
    change = c.diff()
    up = change.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-change.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / down.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    d["macdh"] = hist
    d["macdh_chg3"] = hist.diff(3)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d["atrp"] = tr.rolling(14).mean() / c
    for n in (5, 10, 20, 40, 60):
        d[f"ret{n}"] = c.pct_change(n)
        d[f"ddhi{n}"] = c / h.rolling(n).max() - 1
        d[f"ddlo{n}"] = c / l.rolling(n).min() - 1
        d[f"ma{n}dev"] = c / c.rolling(n).mean() - 1
        d[f"vol{n}"] = v / v.rolling(n).mean()
    # Bullish divergence: close breaks the prior 20-session low, RSI exceeds
    # RSI at that prior low by at least 2 points.
    rsi_at_low = np.full(len(d), np.nan)
    for i in range(20, len(d)):
        j = i - 1 - int(np.argmin(l.iloc[i-20:i].to_numpy()))
        rsi_at_low[i] = d.rsi.iloc[j]
    d["bull_div"] = ((c < l.shift(1).rolling(20).min()) &
                     (d.rsi > rsi_at_low + 2)).astype(int)
    # Bearish divergence: price breaks prior 20-session high but RSI fails
    # to exceed its prior 20-session maximum by 2 points.
    d["bear_div"] = ((c > h.shift(1).rolling(20).max()) &
                     (d.rsi < d.rsi.shift(1).rolling(20).max() - 2)).astype(int)
    d["vol_climax_contract"] = ((d.vol5.shift(3) > 1.8) & (d.vol5 < .8)).astype(int)
    d["bb_pct"] = (c - c.rolling(20).mean()) / (2 * c.rolling(20).std())
    d["bb_width"] = 4 * c.rolling(20).std() / c.rolling(20).mean()
    d["bb_width_chg5"] = d.bb_width.pct_change(5)

    # Outcome labels: in next 20 sessions, find the lowest low; within the
    # following 20 sessions price must rebound at least 10%. High label is
    # symmetric. The label is future-looking by design; features are not.
    n = len(d)
    low_y = np.zeros(n, dtype=int)
    high_y = np.zeros(n, dtype=int)
    for i in range(n - 41):
        lo_idx = i + 1 + int(np.argmin(l.iloc[i+1:i+21].to_numpy()))
        if h.iloc[lo_idx+1:min(n, lo_idx+21)].max() / l.iloc[lo_idx] - 1 >= .10:
            low_y[i] = 1
        hi_idx = i + 1 + int(np.argmax(h.iloc[i+1:i+21].to_numpy()))
        if h.iloc[hi_idx] / l.iloc[hi_idx+1:min(n, hi_idx+21)].min() - 1 >= .10:
            high_y[i] = 1
    d["low_event"] = low_y
    d["high_event"] = high_y
    return d.iloc[:-41].dropna(subset=FEATURES).reset_index(drop=True)


def clustered_rule_rates(d: pd.DataFrame) -> list[dict]:
    rules = {
        "low_drop8_vol15": (d.ret20 < -.08) & (d.vol20 > 1.5),
        "low_drop8_rsi35_vol15": (d.ret20 < -.08) & (d.rsi < 35) & (d.vol20 > 1.5),
        "low_drop3atr_vol15": (d.ddhi20 < -3*d.atrp) & (d.vol20 > 1.5),
        "high_rsi65_bear_div": (d.rsi > 65) & (d.bear_div == 1),
        "high_ret8_rsi65": (d.ret20 > .08) & (d.rsi > 65),
        "high_ma20_bb": (d.ma20dev > .05) & (d.bb_pct > .8),
    }
    rows = []
    for rule, mask in rules.items():
        target = "low_event" if rule.startswith("low") else "high_event"
        for period, (start, end) in PERIODS.items():
            z = d[(d.date >= start) & (d.date <= end)]
            ids = np.flatnonzero(mask.loc[z.index].to_numpy())
            kept, last = [], -10_000
            for idx in ids:
                if idx - last >= 20:
                    kept.append(idx)
                    last = idx
            sample = z.iloc[kept]
            rows.append({"rule": rule, "period": period, "signals": len(sample),
                         "hit_rate": float(sample[target].mean()) if len(sample) else None,
                         "base_rate": float(z[target].mean())})
    return rows


def model_scores(d: pd.DataFrame) -> list[dict]:
    train = d[(d.date >= PERIODS["train"][0]) & (d.date <= PERIODS["train"][1])]
    val = d[(d.date >= PERIODS["validation"][0]) & (d.date <= PERIODS["validation"][1])]
    test = d[(d.date >= PERIODS["historical_test"][0]) & (d.date <= PERIODS["historical_test"][1])]
    results = []
    for target in ("low_event", "high_event"):
        models = [
            ("logistic", make_pipeline(SimpleImputer(), StandardScaler(),
                                       LogisticRegression(class_weight="balanced", max_iter=2000, C=.1))),
            ("random_forest", RandomForestClassifier(n_estimators=500, min_samples_leaf=20,
                             max_features=.8, class_weight="balanced_subsample", random_state=7)),
            ("extra_trees", ExtraTreesClassifier(n_estimators=500, min_samples_leaf=20,
                             max_features=.8, class_weight="balanced", random_state=7)),
        ]
        for model_name, model in models:
            model.fit(train[FEATURES], train[target])
            pv = model.predict_proba(val[FEATURES])[:, 1]
            candidates = []
            for q in (.70, .75, .80, .85, .90, .925, .95, .975):
                threshold = float(np.quantile(pv, q))
                selected = pv >= threshold
                if selected.sum() >= 8:
                    candidates.append((float(val.loc[selected, target].mean()),
                                       int(selected.sum()), threshold))
            if not candidates:
                continue
            # Choose only on validation; threshold is frozen before historical test.
            precision, count, threshold = max(candidates, key=lambda x: (x[0], x[1]))
            for period, frame in (("validation", val), ("historical_test", test)):
                prob = model.predict_proba(frame[FEATURES])[:, 1]
                chosen = prob >= threshold
                results.append({"target": target, "model": model_name, "period": period,
                                "auc": float(roc_auc_score(frame[target], prob)),
                                "base_rate": float(frame[target].mean()),
                                "threshold_frozen_on_validation": threshold,
                                "signals": int(chosen.sum()),
                                "precision": float(frame.loc[chosen, target].mean()) if chosen.any() else None,
                                "recall": float(chosen.sum() / max(1, frame[target].sum()))})
    return results


def run(input_csv: str, output_dir: str) -> dict:
    d = build_frame(input_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rules = clustered_rule_rates(d)
    models = model_scores(d)
    pd.DataFrame(rules).to_csv(out / "tsmc_swing_v2_rule_rates.csv", index=False)
    pd.DataFrame(models).to_csv(out / "tsmc_swing_v2_model_scores.csv", index=False)
    summary = {"symbol": "2330", "rows": len(d), "start": str(d.date.min().date()),
               "end": str(d.date.max().date()), "source": "Yahoo Finance auto_adjust=True",
               "event_definition": "future 20-session extremum then >=10% reversal in next 20 sessions",
               "feature_policy": "all indicators use data through signal date only",
               "signal_de_duplication": "rule rates only: suppress further alerts for 20 sessions",
               "caveats": ["future labels overlap and are not independent",
                           "test period has been examined in prior work, not a pristine holdout",
                           "adjusted OHLCV is not official raw execution data",
                           "model top-signal precision can be high while recall is very low",
                           "no model output is an executable buy/sell strategy"],
               "rule_rates": rules, "model_scores": models}
    (out / "tsmc_swing_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/top10_adjusted_prices/bars/2330.csv")
    parser.add_argument("--output", default="artifacts/tsmc_swing_v2")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), ensure_ascii=False, indent=2))
