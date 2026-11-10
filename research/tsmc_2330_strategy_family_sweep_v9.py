"""V9 single-stock strategy-family sweep for TSMC 2330.

Signals use only current/past adjusted daily OHLCV. Signal at close, fill next open.
Research/validation/test periods are fixed before evaluation; no test-period tuning.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

CAPITAL = 100_000.0
BUY_FEE, SELL_FEE_TAX, SLIPPAGE = .001425, .004425, .001
SPLITS = {
    "research": ("2010-01-04", "2017-12-29"),
    "validation": ("2018-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2026-11-10"),
}

def prepare(path):
    d = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    c, h, l = d.close, d.high, d.low
    for n in (20, 50, 100, 200):
        d[f"ma{n}"] = c.rolling(n).mean()
        d[f"ma{n}_slope20"] = d[f"ma{n}"].pct_change(20)
    for n in (20, 40, 55, 100):
        d[f"high{n}_prev"] = h.shift(1).rolling(n).max()
    for n in (20, 40, 60, 120):
        d[f"ret{n}"] = c.pct_change(n)
    change = c.diff()
    up = change.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-change.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / down.replace(0, np.nan))
    d["rsi_prev"] = d.rsi.shift(1)
    return d

def candidates():
    out = []
    for n in (20, 40, 55, 100):
        for trend in (50, 100, 200):
            for exit_ma in (20, 50, 100):
                if exit_ma <= trend:
                    out.append(("breakout", (n, trend, exit_ma)))
    for look in (20, 40, 60, 120):
        for threshold in (0, .05, .10, .15, .20, .30):
            for trend in (50, 100, 200):
                for exit_ma in (20, 50, 100):
                    if exit_ma <= trend:
                        out.append(("momentum", (look, threshold, trend, exit_ma)))
    for trend in (100, 200):
        for floor in (25, 30, 35, 40, 45):
            for exit_ma in (20, 50):
                out.append(("pullback", (trend, floor, exit_ma)))
    return out

def make_signals(d, family, p):
    if family == "breakout":
        n, trend, exit_ma = p
        buy = (d.close > d[f"high{n}_prev"]) & (d.close > d[f"ma{trend}"]) & (d[f"ma{trend}_slope20"] > 0)
    elif family == "momentum":
        look, threshold, trend, exit_ma = p
        buy = (d[f"ret{look}"] > threshold) & (d.close > d[f"ma{trend}"]) & (d[f"ma{trend}_slope20"] > 0)
    else:
        trend, floor, exit_ma = p
        buy = (d.close > d[f"ma{trend}"]) & (d.rsi_prev < floor) & (d.rsi >= floor)
    sell = d.close < d[f"ma{exit_ma}"]
    return buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy()

def simulate(d, buy, sell, start, end):
    cash, shares, pending, entry = CAPITAL, 0, None, 0.0
    curve, trades = [], []
    for i in range(start, end + 1):
        row = d.iloc[i]
        if pending:
            px = float(row.open) * (1 + SLIPPAGE if pending == "buy" else 1 - SLIPPAGE)
            if pending == "buy" and shares == 0:
                qty = int(cash / (px * (1 + BUY_FEE)))
                if qty:
                    cash -= qty * px * (1 + BUY_FEE)
                    shares, entry = qty, px
                    trades.append({"entry_date": str(row.date.date()), "entry_price": px, "shares": qty})
            elif pending == "sell" and shares:
                proceeds = shares * px * (1 - SELL_FEE_TAX)
                cash += proceeds
                trades[-1].update({"exit_date": str(row.date.date()), "exit_price": px,
                    "net_return": proceeds / (shares * entry * (1 + BUY_FEE)) - 1})
                shares, entry = 0, 0.0
            pending = None
        curve.append(cash + shares * float(row.close))
        if shares:
            if sell[i]:
                pending = "sell"
        elif buy[i]:
            pending = "buy"
    if shares:
        row = d.iloc[end]
        px = float(row.close) * (1 - SLIPPAGE)
        proceeds = shares * px * (1 - SELL_FEE_TAX)
        cash += proceeds
        trades[-1].update({"exit_date": str(row.date.date()), "exit_price": px,
            "net_return": proceeds / (shares * entry * (1 + BUY_FEE)) - 1})
        curve[-1] = cash
    curve = np.asarray(curve, dtype=float)
    years = (d.date.iloc[end] - d.date.iloc[start]).days / 365.25
    cagr = (curve[-1] / CAPITAL) ** (1 / years) - 1 if years > 0 else np.nan
    mdd = float(np.min(curve / np.maximum.accumulate(curve) - 1))
    closed = [t for t in trades if "exit_date" in t]
    return {"final_equity": float(curve[-1]), "cagr": float(cagr), "max_drawdown": mdd,
        "trades": len(closed), "win_rate": float(np.mean([t["net_return"] > 0 for t in closed])) if closed else None}, curve, closed

def benchmark(d, start, end):
    first_open = float(d.open.iloc[start]) * (1 + SLIPPAGE)
    qty = int(CAPITAL / (first_open * (1 + BUY_FEE)))
    cash = CAPITAL - qty * first_open * (1 + BUY_FEE)
    final = cash + qty * float(d.close.iloc[end]) * (1 - SLIPPAGE) * (1 - SELL_FEE_TAX)
    years = (d.date.iloc[end] - d.date.iloc[start]).days / 365.25
    return {"final_equity": final, "cagr": (final / CAPITAL) ** (1 / years) - 1, "max_drawdown": None}

def run(input_csv, output_dir):
    d = prepare(input_csv)
    ranges = {}
    for name, (lo, hi) in SPLITS.items():
        ix = d.index[(d.date >= lo) & (d.date <= hi)]
        if len(ix) == 0:
            raise ValueError(f"Missing data for split {name}: {lo}..{hi}")
        ranges[name] = (int(ix[0]), int(ix[-1]))
    rows, trades_out = [], []
    for family, params in candidates():
        buy, sell = make_signals(d, family, params)
        result = {"family": family, "parameters": json.dumps(params)}
        for split, (start, end) in ranges.items():
            metrics, _, trades = simulate(d, buy, sell, start, end)
            for key, value in metrics.items():
                result[f"{split}_{key}"] = value
            if split == "test":
                trades_out.extend([{"family": family, "parameters": result["parameters"], **t} for t in trades])
        rows.append(result)
    table = pd.DataFrame(rows).sort_values("research_cagr", ascending=False)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "strategy_family_sweep.csv", index=False)
    pd.DataFrame(trades_out).to_csv(out / "test_trades.csv", index=False)
    bench = {}
    for split, (start, end) in ranges.items():
        bench[split] = benchmark(d, start, end)
    summary = {
        "version": "V9", "symbol": "2330", "rows": len(d),
        "history_start": str(d.date.min().date()), "data_end": str(d.date.max().date()),
        "source": str(d.provider.iloc[0]) if "provider" in d else "user-supplied adjusted OHLCV",
        "splits": {k: {"start": str(d.date.iloc[v[0]].date()), "end": str(d.date.iloc[v[1]].date())} for k,v in ranges.items()},
        "candidate_count": len(rows), "buy_and_hold": bench,
        "top_10_by_research_cagr": table.head(10).replace({np.nan: None}).to_dict(orient="records"),
        "best_research_candidate": table.iloc[0].replace({np.nan: None}).to_dict(),
        "caveats": [
            "244 predeclared variants across breakout, momentum and trend-filtered RSI pullback families.",
            "Candidates are ranked only on 2010-2017 research CAGR; 2018-2021 validation and 2022-2026 test are not used for ranking.",
            "Daily adjusted Yahoo OHLCV is not official execution-grade exchange data.",
            "Long-only, integer shares, full cash, one position, no leverage; close signal and next-open fill.",
            "Fees and 0.10% slippage per side included; no market impact, queue, limit-up/down or partial-fill simulation.",
            "This sweep does not establish a 200% CAGR strategy; it is a screening experiment, not proof of a tradable edge."
        ]
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/top10_adjusted_prices/bars/2330.csv")
    parser.add_argument("--output", default="artifacts/tsmc_v9_strategy_family_sweep")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), ensure_ascii=False, indent=2, default=str))
