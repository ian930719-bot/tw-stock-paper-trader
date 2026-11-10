"""V11: TSMC 2330 squeeze, pullback re-entry and short-momentum study.
Signals are evaluated at the close and filled at the next session open.
Candidates are ranked only on the fixed 2010-2017 research interval.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

CAPITAL = 100_000.0
BUY_FEE, SELL_FEE_TAX, SLIPPAGE = 0.001425, 0.004425, 0.001
SPLITS = {
    "research": ("2010-01-04", "2017-12-29"),
    "validation": ("2018-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2026-11-10"),
}


def prepare(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    c, h, l, v = d.close, d.high, d.low, d.volume
    for n in (5, 10, 20, 50, 200):
        d[f"ma{n}"] = c.rolling(n).mean()
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    change = c.diff()
    up = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / down.replace(0, np.nan))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["bb_width"] = 4 * sd / mid
    d["bb_width_q25"] = d.bb_width.rolling(120, min_periods=60).quantile(0.25)
    d["volume_ratio"] = v / v.rolling(20).mean()
    d["drawdown20"] = c / c.rolling(20).max() - 1
    for n in (10, 20):
        d[f"prior_high{n}"] = h.shift(1).rolling(n).max()
    return d


def build_signals(d: pd.DataFrame) -> dict[str, np.ndarray]:
    c = d.close
    signals = {
        "squeeze_breakout": (
            (d.bb_width < d.bb_width_q25) & (c > d.prior_high20) &
            (c > d.ma50) & (d.ma50 > d.ma200) & (d.volume_ratio > 1.15)
        ),
        "trend_pullback_reentry": (
            (c > d.ema10) & (c.shift(1) <= d.ema10.shift(1)) &
            (d.ma50 > d.ma200) & (c > d.ma200) &
            d.drawdown20.between(-0.12, -0.025) & d.rsi.between(40, 65)
        ),
        "momentum_breakout": (
            (c > d.prior_high10) & (c > d.ma50) &
            (d.ma50.pct_change(20) > 0) & d.rsi.between(50, 75)
        ),
        "short_rebound": (
            (c > d.ema5) & (c.shift(1) <= d.ema5.shift(1)) &
            (d.ma50.pct_change(20) > 0) & d.rsi.between(35, 60) &
            (d.drawdown20 < -0.02)
        ),
    }
    return {name: s.fillna(False).to_numpy(dtype=bool) for name, s in signals.items()}


def simulate(d: pd.DataFrame, signal: np.ndarray, start: int, end: int,
             atr_multiple: float, exit_ma: int, max_hold: int,
             initial_stop: float, keep_trades: bool = False):
    opens = d.open.to_numpy(dtype=float)
    closes = d.close.to_numpy(dtype=float)
    atr = d.atr14.to_numpy(dtype=float)
    ma = d[f"ma{exit_ma}"].to_numpy(dtype=float)
    dates = d.date
    cash, shares, entry, entry_i, peak, pending = CAPITAL, 0, 0.0, -1, 0.0, 0
    curve = np.empty(end - start + 1, dtype=float)
    trades: list[dict] = []
    net_returns: list[float] = []
    for j, i in enumerate(range(start, end + 1)):
        if pending:
            action = pending
            px = opens[i] * (1 + SLIPPAGE if action == 1 else 1 - SLIPPAGE)
            if action == 1 and shares == 0:
                qty = int(cash / (px * (1 + BUY_FEE)))
                if qty:
                    cash -= qty * px * (1 + BUY_FEE)
                    shares, entry, entry_i, peak = qty, px, i, closes[i]
                    if keep_trades:
                        trades.append({"entry_date": str(dates.iloc[i].date()),
                                       "entry_price": px, "shares": qty})
            elif action == -1 and shares:
                proceeds = shares * px * (1 - SELL_FEE_TAX)
                net = proceeds / (shares * entry * (1 + BUY_FEE)) - 1
                cash += proceeds
                net_returns.append(net)
                if keep_trades:
                    trades[-1].update({"exit_date": str(dates.iloc[i].date()),
                                       "exit_price": px, "net_return": net,
                                       "hold_sessions": i - entry_i})
                shares, entry, entry_i, peak = 0, 0.0, -1, 0.0
            pending = 0
        curve[j] = cash + shares * closes[i]
        if shares:
            peak = max(peak, closes[i])
            atr_stop = np.isfinite(atr[i]) and atr[i] > 0 and closes[i] <= peak - atr_multiple * atr[i]
            fixed_stop = closes[i] <= entry * (1 - initial_stop)
            trend_exit = closes[i] < ma[i]
            time_exit = i - entry_i >= max_hold
            if atr_stop or fixed_stop or trend_exit or time_exit:
                pending = -1
        elif signal[i]:
            pending = 1
    if shares:
        px = closes[end] * (1 - SLIPPAGE)
        proceeds = shares * px * (1 - SELL_FEE_TAX)
        net = proceeds / (shares * entry * (1 + BUY_FEE)) - 1
        cash += proceeds
        net_returns.append(net)
        curve[-1] = cash
        if keep_trades:
            trades[-1].update({"exit_date": str(dates.iloc[end].date()),
                               "exit_price": px, "net_return": net,
                               "hold_sessions": end - entry_i})
    years = (dates.iloc[end] - dates.iloc[start]).days / 365.25
    cagr = (cash / CAPITAL) ** (1 / years) - 1 if years > 0 else np.nan
    mdd = float(np.min(curve / np.maximum.accumulate(curve) - 1))
    closed = [t for t in trades if "exit_date" in t]
    return {
        "final_equity": float(cash), "cagr_pct": float(cagr * 100),
        "max_drawdown_pct": mdd * 100, "closed_trades": len(net_returns),
        "win_rate_pct": float(np.mean(np.asarray(net_returns) > 0) * 100) if net_returns else None,
        "average_net_trade_pct": float(np.mean(net_returns) * 100) if net_returns else None,
    }, curve, closed


def benchmark(d: pd.DataFrame, start: int, end: int) -> dict:
    o, c = d.open.to_numpy(float), d.close.to_numpy(float)
    px = o[start] * (1 + SLIPPAGE)
    qty = int(CAPITAL / (px * (1 + BUY_FEE)))
    cash = CAPITAL - qty * px * (1 + BUY_FEE)
    curve = cash + qty * c[start:end + 1]
    curve[-1] = cash + qty * c[end] * (1 - SLIPPAGE) * (1 - SELL_FEE_TAX)
    years = (d.date.iloc[end] - d.date.iloc[start]).days / 365.25
    return {
        "final_equity": float(curve[-1]),
        "cagr_pct": float(((curve[-1] / CAPITAL) ** (1 / years) - 1) * 100),
        "max_drawdown_pct": float(np.min(curve / np.maximum.accumulate(curve) - 1) * 100),
    }


def run(input_csv: str, output_dir: str) -> dict:
    d = prepare(input_csv)
    ranges = {}
    for name, (lo, hi) in SPLITS.items():
        idx = d.index[(d.date >= lo) & (d.date <= hi)]
        if len(idx) == 0:
            raise ValueError(f"Missing data for split {name}: {lo}..{hi}")
        ranges[name] = (int(idx[0]), int(idx[-1]))
    signals = build_signals(d)
    rows = []
    for family, signal in signals.items():
        for atr_mult, exit_ma, max_hold, initial_stop in itertools.product(
            (2.0, 2.5, 3.0, 3.5), (10, 20), (20, 40, 60), (0.06, 0.08, 0.10)
        ):
            row = {"family": family, "atr_multiple": atr_mult,
                   "exit_ma": exit_ma, "max_hold_sessions": max_hold,
                   "initial_stop_pct": initial_stop * 100}
            for split, (start, end) in ranges.items():
                metrics, _, _ = simulate(d, signal, start, end, atr_mult,
                                         exit_ma, max_hold, initial_stop)
                row[split] = metrics
                for key, value in metrics.items():
                    row[f"{split}_{key}"] = value
            rows.append(row)
    table = pd.DataFrame(rows).sort_values(
        ["research_cagr_pct", "research_max_drawdown_pct"],
        ascending=[False, False]
    ).reset_index(drop=True)
    selected = table.iloc[0].to_dict()
    family = selected["family"]
    signal = signals[family]
    test_start, test_end = ranges["test"]
    _, curve, selected_trades = simulate(
        d, signal, test_start, test_end, float(selected["atr_multiple"]),
        int(selected["exit_ma"]), int(selected["max_hold_sessions"]),
        float(selected["initial_stop_pct"]) / 100, True
    )
    benchmarks = {name: benchmark(d, *span) for name, span in ranges.items()}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.drop(columns=["research", "validation", "test"]).to_csv(out / "candidate_results.csv", index=False)
    pd.DataFrame(selected_trades).to_csv(out / "selected_test_trades.csv", index=False)
    pd.DataFrame({"date": d.date.iloc[test_start:test_end + 1].reset_index(drop=True),
                  "equity": curve}).to_csv(out / "selected_test_equity.csv", index=False)
    summary = {
        "version": "V11", "symbol": "2330", "history_rows": len(d),
        "history_start": str(d.date.min().date()), "data_end": str(d.date.max().date()),
        "data_source": str(d.provider.iloc[0]) if "provider" in d else "adjusted OHLCV",
        "splits": {k: {"start": str(d.date.iloc[v[0]].date()),
                       "end": str(d.date.iloc[v[1]].date())} for k, v in ranges.items()},
        "candidate_count": len(rows), "buy_and_hold": benchmarks,
        "research_selected_candidate": selected,
        "top_10_research_candidates": table.head(10).drop(columns=["research", "validation", "test"]).to_dict("records"),
        "limitations": [
            "Candidates ranked only by 2010-2017 research CAGR; later periods were not used for selection.",
            "Single stock and historical data; no future return guarantee.",
            "Yahoo adjusted daily OHLCV is not official exchange execution data.",
            "Signals are close-known and filled next open; intraday stop execution is not modeled.",
            "No leverage, shorting, partial fills, market impact, limit queue or broker restrictions.",
            "V11 exit policy is a fixed exploratory rule family, not a validated production strategy.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/top10_adjusted_prices/bars/2330.csv")
    parser.add_argument("--output", default="artifacts/tsmc_v11_reentry")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), ensure_ascii=False, indent=2, default=str))
