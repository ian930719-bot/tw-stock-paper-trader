"""V10: small-swing mean-reversion and pullback study for TSMC 2330.
All signals use close-known information and fill at next open. Candidates are
ranked on 2010-2017 only; validation/test are diagnostics and never ranked.
"""
from __future__ import annotations
import argparse, json
from itertools import product
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
    for n in (10, 20, 50, 200):
        d[f"ma{n}"] = c.rolling(n).mean()
    change = c.diff()
    up = change.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-change.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / down.replace(0, np.nan))
    d["rsi_prev"] = d.rsi.shift(1)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["bbp"] = (c - (mid - 2 * sd)) / (4 * sd)
    d["bbp_prev"] = d.bbp.shift(1)
    return d

def build_candidates(d):
    close = d.close
    signals = {}
    specs = []
    for kind in ("rsi", "bb", "ema"):
        thresholds = (20, 25, 30, 35, 40) if kind == "rsi" else (None,)
        for threshold, trend, target, stop, hold in product(
            thresholds, (0, 50, 200), (.03, .05, .08, .10, .15),
            (.03, .05, .08, .10), (5, 10, 15, 20)
        ):
            key = (kind, threshold, trend)
            if key not in signals:
                if kind == "rsi":
                    buy = ((d.rsi_prev < threshold) & (d.rsi >= threshold) &
                           (close > close.shift(1)))
                elif kind == "bb":
                    buy = ((d.bbp_prev < 0) & (d.bbp >= 0) &
                           (close > close.shift(1)))
                else:
                    buy = ((close > close.rolling(10).mean()) &
                           (close.shift(1) <= close.rolling(10).mean().shift(1)) &
                           (d.rsi < 65))
                if trend == 50:
                    buy &= close > d.ma50
                elif trend == 200:
                    buy &= close > d.ma200
                signals[key] = buy.fillna(False).to_numpy(bool)
            specs.append((kind, threshold, trend, target, stop, hold, key))
    return specs, signals

def simulate(d, buy, target, stop, max_hold, start, end, keep_trades=False):
    opens, closes = d.open.to_numpy(float), d.close.to_numpy(float)
    dates = d.date.to_numpy()
    cash, shares, entry, entry_i, pending = CAPITAL, 0, 0.0, -1, 0
    curve = np.empty(end - start + 1)
    trades = []
    wins = total_return = 0.0
    count = 0
    for j, i in enumerate(range(start, end + 1)):
        if pending:
            action = pending
            px = opens[i] * (1 + SLIPPAGE if action == 1 else 1 - SLIPPAGE)
            if action == 1 and shares == 0:
                qty = int(cash / (px * (1 + BUY_FEE)))
                if qty:
                    cash -= qty * px * (1 + BUY_FEE)
                    shares, entry, entry_i = qty, px, i
                    if keep_trades:
                        trades.append({"entry_date": str(pd.Timestamp(dates[i]).date()),
                                       "entry_price": px, "shares": qty})
            elif action == -1 and shares:
                proceeds = shares * px * (1 - SELL_FEE_TAX)
                net = proceeds / (shares * entry * (1 + BUY_FEE)) - 1
                cash += proceeds
                count += 1
                wins += net > 0
                total_return += net
                if keep_trades:
                    trades[-1].update({"exit_date": str(pd.Timestamp(dates[i]).date()),
                                       "exit_price": px, "net_return": net,
                                       "hold_sessions": i - entry_i})
                shares, entry, entry_i = 0, 0.0, -1
            pending = 0
        curve[j] = cash + shares * closes[i]
        if shares:
            ret = closes[i] / entry - 1
            if ret >= target or ret <= -stop or i - entry_i >= max_hold:
                pending = -1
        elif buy[i]:
            pending = 1
    if shares:
        px = closes[end] * (1 - SLIPPAGE)
        proceeds = shares * px * (1 - SELL_FEE_TAX)
        net = proceeds / (shares * entry * (1 + BUY_FEE)) - 1
        cash += proceeds
        count += 1
        wins += net > 0
        total_return += net
        curve[-1] = cash
        if keep_trades:
            trades[-1].update({"exit_date": str(pd.Timestamp(dates[end]).date()),
                               "exit_price": px, "net_return": net,
                               "hold_sessions": end - entry_i})
    years = (pd.Timestamp(dates[end]) - pd.Timestamp(dates[start])).days / 365.25
    cagr = (cash / CAPITAL) ** (1 / years) - 1
    mdd = float(np.min(curve / np.maximum.accumulate(curve) - 1))
    return {"final_equity": float(cash), "cagr": float(cagr),
            "max_drawdown": mdd, "trades": count,
            "win_rate": wins / count if count else None,
            "average_net_trade": total_return / count if count else None}, trades

def benchmark(d, start, end):
    o, c = d.open.to_numpy(float), d.close.to_numpy(float)
    dates = d.date.to_numpy()
    px = o[start] * (1 + SLIPPAGE)
    qty = int(CAPITAL / (px * (1 + BUY_FEE)))
    cash = CAPITAL - qty * px * (1 + BUY_FEE)
    curve = cash + qty * c[start:end+1]
    curve[-1] = cash + qty * c[end] * (1 - SLIPPAGE) * (1 - SELL_FEE_TAX)
    years = (pd.Timestamp(dates[end]) - pd.Timestamp(dates[start])).days / 365.25
    return {"final_equity": float(curve[-1]),
            "cagr": float((curve[-1] / CAPITAL) ** (1 / years) - 1),
            "max_drawdown": float(np.min(curve / np.maximum.accumulate(curve) - 1))}

def run(input_csv, output_dir):
    d = prepare(input_csv)
    ranges = {}
    for name, (lo, hi) in SPLITS.items():
        ix = d.index[(d.date >= lo) & (d.date <= hi)]
        if len(ix) == 0:
            raise ValueError(f"Missing data for {name}: {lo}..{hi}")
        ranges[name] = (int(ix[0]), int(ix[-1]))
    specs, signals = build_candidates(d)
    rows = []
    for kind, threshold, trend, target, stop, hold, key in specs:
        row = {"family": kind, "rsi_reclaim_level": threshold,
               "trend_filter_ma": trend or "none", "take_profit": target,
               "stop_loss": stop, "max_hold_sessions": hold}
        for split, (start, end) in ranges.items():
            m, _ = simulate(d, signals[key], target, stop, hold, start, end)
            for metric, value in m.items():
                row[f"{split}_{metric}"] = value
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("research_cagr", ascending=False).reset_index(drop=True)
    selected = table.iloc[0]
    test_best = table.sort_values("test_cagr", ascending=False).iloc[0]
    selected_key = (selected.family, None if pd.isna(selected.rsi_reclaim_level) else int(selected.rsi_reclaim_level),
                    0 if selected.trend_filter_ma == "none" else int(selected.trend_filter_ma))
    diagnostic_key = (test_best.family, None if pd.isna(test_best.rsi_reclaim_level) else int(test_best.rsi_reclaim_level),
                      0 if test_best.trend_filter_ma == "none" else int(test_best.trend_filter_ma))
    test_start, test_end = ranges["test"]
    _, selected_trades = simulate(d, signals[selected_key], selected.take_profit,
                                  selected.stop_loss, int(selected.max_hold_sessions),
                                  test_start, test_end, True)
    _, diagnostic_trades = simulate(d, signals[diagnostic_key], test_best.take_profit,
                                    test_best.stop_loss, int(test_best.max_hold_sessions),
                                    test_start, test_end, True)
    benchmarks = {name: benchmark(d, *span) for name, span in ranges.items()}
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "small_swing_candidate_results.csv", index=False)
    pd.DataFrame([{"selection": "research_ranked_candidate", **t} for t in selected_trades] +
                 [{"selection": "test_best_diagnostic_only", **t} for t in diagnostic_trades]
                 ).to_csv(out / "selected_test_trades.csv", index=False)
    summary = {
        "version": "V10", "symbol": "2330", "rows": len(d),
        "history_start": str(d.date.min().date()), "data_end": str(d.date.max().date()),
        "source": str(d.provider.iloc[0]) if "provider" in d else "adjusted OHLCV",
        "splits": {k: {"start": str(d.date.iloc[v[0]].date()),
                       "end": str(d.date.iloc[v[1]].date())} for k, v in ranges.items()},
        "candidate_count": len(rows), "buy_and_hold": benchmarks,
        "research_selected_candidate": selected.to_dict(),
        "test_best_diagnostic_only_not_selectable": test_best.to_dict(),
        "limitations": [
            "Candidates ranked only by research-period CAGR; validation/test do not influence selection.",
            "Test-best row is shown only to quantify selection bias and must not be adopted.",
            "Signals are close-based and filled next open; target/stop are checked at close, not intraday.",
            "No leverage, shorting, partial positions, market impact, limit-queue or partial-fill model.",
            "Yahoo adjusted OHLCV is not official execution-grade exchange data.",
            "A 2022-2026 test period is a historical regime, not a guarantee of future performance."
        ]
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/top10_adjusted_prices/bars/2330.csv")
    ap.add_argument("--output", default="artifacts/tsmc_v10_small_swing")
    args = ap.parse_args()
    print(json.dumps(run(args.input, args.output), ensure_ascii=False, indent=2, default=str))
