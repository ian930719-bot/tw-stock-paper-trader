"""TSMC 2330 swing-low/high research: retrospective oracle vs causal confirmation.

Input is Yahoo auto_adjust=True adjusted OHLCV. This is a research proxy, not raw
exchange execution data. Signals confirm only after a specified reversal and
fills occur at the following session open. Retrospective pivot fills are an
explicitly impossible upper-bound diagnostic and must never be treated as
tradable performance.
"""
from __future__ import annotations
import argparse, itertools, json, math
from pathlib import Path
import numpy as np
import pandas as pd

BUY_FEE, SELL_FEE_TAX, SLIPPAGE = 0.001425, 0.004425, 0.001
INITIAL_CAPITAL = 100_000.0
PERIODS = {
    "train": ("2010-01-04", "2020-12-31"),
    "validation": ("2021-01-04", "2023-12-29"),
    "historical_test": ("2024-01-02", "2026-11-10"),
}

def load_bars(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["date", "open", "high", "low", "close"])
    valid = ((d[["open", "high", "low", "close"]] > 0).all(axis=1)
             & (d.high >= d[["open", "close", "low"]].max(axis=1))
             & (d.low <= d[["open", "close", "high"]].min(axis=1)))
    return d.loc[valid].drop_duplicates("date", keep="last").reset_index(drop=True)

def confirmed_events(d: pd.DataFrame, buy_rebound: float, sell_reversal: float):
    """Return confirmation-day buy/sell flags; no future bars are used."""
    n = len(d); buy = np.zeros(n, bool); sell = np.zeros(n, bool)
    hi = d.high.to_numpy(float); lo = d.low.to_numpy(float); cl = d.close.to_numpy(float)
    state = 0  # 0: seek low then confirm rebound; 1: seek high then confirm decline
    extreme = lo[0]
    for i in range(1, n):
        if state == 0:
            extreme = min(extreme, lo[i])
            if cl[i] >= extreme * (1 + buy_rebound):
                buy[i] = True; state = 1; extreme = hi[i]
        else:
            extreme = max(extreme, hi[i])
            if cl[i] <= extreme * (1 - sell_reversal):
                sell[i] = True; state = 0; extreme = lo[i]
    return buy, sell

def metrics(curve, trade_pnls, initial=INITIAL_CAPITAL):
    if not curve: return {}
    vals = np.asarray([x[1] for x in curve], float)
    days = int((curve[-1][0] - curve[0][0]).astype("timedelta64[D]").astype(int))
    years = max(days / 365.25, 1 / 365.25)
    dd = vals / np.maximum.accumulate(np.r_[initial, vals])[1:] - 1
    pnl = np.asarray(trade_pnls, float)
    loss = abs(pnl[pnl < 0].sum())
    return {
        "CAGR": float((vals[-1] / initial) ** (1 / years) - 1),
        "ending_equity": float(vals[-1]), "max_drawdown": float(dd.min()),
        "trades": int(len(pnl)),
        "profit_factor": float(pnl[pnl > 0].sum() / loss) if loss else (float("inf") if np.any(pnl > 0) else 0.0),
    }

def causal_backtest(d, buy_rebound, sell_reversal, start, end):
    buy, sell = confirmed_events(d, buy_rebound, sell_reversal)
    dates = d.date.to_numpy(); op = d.open.to_numpy(float); cl = d.close.to_numpy(float)
    idxs = np.flatnonzero((dates >= np.datetime64(start)) & (dates <= np.datetime64(end)))
    cash = INITIAL_CAPITAL; qty = 0; cost = 0.0; curve = []; pnls = []
    for i in idxs:
        if qty and i > 0 and sell[i-1]:
            fill = op[i] * (1 - SLIPPAGE)
            proceeds = qty * fill * (1 - SELL_FEE_TAX)
            pnls.append(proceeds - cost); cash += proceeds; qty = 0
        if not qty and i > 0 and buy[i-1]:
            fill = op[i] * (1 + SLIPPAGE)
            qty = math.floor(cash / (fill * (1 + BUY_FEE)))
            if qty:
                cost = qty * fill * (1 + BUY_FEE); cash -= cost
        curve.append((dates[i], cash + qty * cl[i]))
    if qty and curve:
        i = idxs[-1]; proceeds = qty * cl[i] * (1 - SLIPPAGE) * (1 - SELL_FEE_TAX)
        pnls.append(proceeds - cost); cash += proceeds; curve[-1] = (dates[i], cash)
    return metrics(curve, pnls)

def retrospective_oracle(d, reversal, start, end):
    """Impossible upper bound: fill at the eventual pivot extreme, known only later."""
    hi = d.high.to_numpy(float); lo = d.low.to_numpy(float); cl = d.close.to_numpy(float)
    dates = d.date.to_numpy(); state = 0; extreme = lo[0]; extreme_idx = 0; pivots = []
    for i in range(1, len(d)):
        if state in (0, -1):
            if lo[i] < extreme: extreme, extreme_idx = lo[i], i
            if cl[i] >= extreme * (1 + reversal):
                pivots.append((extreme_idx, "low", extreme)); state = 1; extreme, extreme_idx = hi[i], i
        else:
            if hi[i] > extreme: extreme, extreme_idx = hi[i], i
            if cl[i] <= extreme * (1 - reversal):
                pivots.append((extreme_idx, "high", extreme)); state = -1; extreme, extreme_idx = lo[i], i
    a, b = np.datetime64(start), np.datetime64(end)
    cash = INITIAL_CAPITAL; qty = 0; cost = 0.0; pnls = []
    for idx, typ, price in pivots:
        if dates[idx] < a or dates[idx] > b: continue
        if typ == "low" and qty == 0:
            fill = price * (1 + SLIPPAGE); qty = math.floor(cash / (fill * (1 + BUY_FEE)))
            cost = qty * fill * (1 + BUY_FEE); cash -= cost
        elif typ == "high" and qty:
            proceeds = qty * price * (1 - SLIPPAGE) * (1 - SELL_FEE_TAX)
            pnls.append(proceeds - cost); cash += proceeds; qty = 0
    if qty:
        idx = np.flatnonzero((dates >= a) & (dates <= b))[-1]
        proceeds = qty * cl[idx] * (1 - SLIPPAGE) * (1 - SELL_FEE_TAX)
        pnls.append(proceeds - cost); cash += proceeds
    period_dates = dates[(dates >= a) & (dates <= b)]
    years = max(int((period_dates[-1] - period_dates[0]).astype("timedelta64[D]").astype(int)) / 365.25, 1/365.25)
    return {"CAGR": float((cash / INITIAL_CAPITAL) ** (1 / years) - 1),
            "ending_equity": float(cash), "trades": len(pnls), "impossible_hindsight": True}

def run(input_csv, output_dir):
    d = load_bars(input_csv); out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    levels = [round(x, 2) for x in np.arange(.03, .201, .01)]
    rows = []
    for br, sr in itertools.product(levels, levels):
        val = causal_backtest(d, br, sr, *PERIODS["validation"].__str__().strip("()").split(", ")) if False else causal_backtest(d, br, sr, *PERIODS["validation"])
        if val["trades"] < 8: continue
        row = {"buy_rebound": br, "sell_reversal": sr}
        for name, period in PERIODS.items():
            row.update({f"{name}_{k}": v for k, v in causal_backtest(d, br, sr, *period).items()})
        rows.append(row)
    grid = pd.DataFrame(rows).sort_values("validation_CAGR", ascending=False)
    grid.to_csv(out / "tsmc_swing_parameter_grid.csv", index=False)
    oracle_rows = []
    for reversal in (.04, .05, .06, .08, .10, .12, .15, .20, .30):
        for name, period in PERIODS.items():
            oracle_rows.append({"reversal": reversal, "period": name, **retrospective_oracle(d, reversal, *period)})
    pd.DataFrame(oracle_rows).to_csv(out / "tsmc_retrospective_oracle.csv", index=False)
    single_rows = []
    for reversal in (.04, .05, .06, .08, .10, .12, .15, .20, .30):
        row = {"same_rebound_and_reversal": reversal}
        for name, period in PERIODS.items():
            row.update({f"{name}_{k}": v for k, v in causal_backtest(d, reversal, reversal, *period).items()})
        single_rows.append(row)
    pd.DataFrame(single_rows).to_csv(out / "tsmc_same_threshold_results.csv", index=False)
    report = {
        "symbol": "2330", "data_rows": len(d), "data_start": str(d.date.min().date()),
        "data_end": str(d.date.max().date()), "source": "Yahoo Finance auto_adjust=True",
        "execution": "close-confirmed signal then next available adjusted open; long-only, all-in integer shares, cash-only",
        "costs": {"buy_fee": BUY_FEE, "sell_fee_plus_tax": SELL_FEE_TAX, "slippage_each_side": SLIPPAGE},
        "parameter_pairs_tested": len(grid), "minimum_validation_trades": 8,
        "best_validation_pair": grid.iloc[0].to_dict() if len(grid) else {},
        "warning": "Adjusted prices are not raw execution prices. Oracle uses future-confirmed pivot extreme prices and is impossible to trade; it is an upper-bound illustration only.",
    }
    (out / "tsmc_swing_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/top10_adjusted_prices/bars/2330.csv")
    parser.add_argument("--output", default="artifacts/tsmc_swing")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), ensure_ascii=False, indent=2, default=str))
