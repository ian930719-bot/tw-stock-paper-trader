"""Robustness audit for the frozen TSMC 2330 capitulation/climax candidate.

This is a sensitivity audit, not parameter optimization. It keeps the candidate
rule fixed, reports chronological subperiods, execution delays, cost stress and
one-factor neighbouring thresholds. It never selects a new best parameter set.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

START_DATE = "2016-01-04"
INITIAL_CAPITAL = 100_000.0
FEE_RATE = 0.001425
SELL_TAX_RATE = 0.003
SLIPPAGE = 0.001
MIN_COMMISSION = 20


def commission(notional: float, fee_rate: float = FEE_RATE) -> int:
    return max(MIN_COMMISSION, int(math.ceil(notional * fee_rate))) if notional > 0 else 0


def build_candidate_signals(
    d: pd.DataFrame,
    *,
    low_relative_volume: float = 1.2,
    low_return: float = -0.01,
    low_stoch: float = 30.0,
    low_close_location: float = 0.25,
    high_distance_sma200: float = 0.10,
    high_return: float = 0.01,
    high_close_location: float = 0.70,
    high_intraday_range: float = 0.016,
    high_macd_change: float = 0.0,
    high_sma50_slope: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Frozen LOW capitulation + OBV and HIGH climax + extension/fade rule."""
    low = (
        (d.relative_volume20 >= low_relative_volume)
        & (d.return_1 <= low_return)
        & (d.stoch_k14 <= low_stoch)
        & (d.close_location <= low_close_location)
        & (d.obv_flow_ratio_5 <= 0)
    )
    high_climax = (
        (d.return_1 >= high_return)
        & (d.close_location >= high_close_location)
        & (d.intraday_range_pct >= high_intraday_range)
        & (d.distance_sma_200 >= high_distance_sma200)
    )
    high_extension = (
        (d.distance_sma_200 >= high_distance_sma200)
        & (d.macd_hist_change_3 <= high_macd_change)
        & (d.sma50_slope_10 > high_sma50_slope)
    )
    high = high_climax & high_extension
    action_window = (
        d.corporate_action_window.fillna(True).astype(bool)
        if "corporate_action_window" in d
        else pd.Series(False, index=d.index)
    )
    return low.fillna(False) & ~action_window, high.fillna(False) & ~action_window


def simulate(
    d: pd.DataFrame,
    low_signal: pd.Series,
    high_signal: pd.Series,
    *,
    start: str = START_DATE,
    end: str | None = None,
    slippage: float = SLIPPAGE,
    fee_rate: float = FEE_RATE,
    sell_tax_rate: float = SELL_TAX_RATE,
    delay_sessions: int = 0,
    capital: float = INITIAL_CAPITAL,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """All-in/all-out, next-session-open fills; no leverage or shorting."""
    if delay_sessions < 0:
        raise ValueError("delay_sessions must be non-negative")
    d = d.sort_values("date").reset_index(drop=True)
    lo = low_signal.reindex(d.index).fillna(False).astype(bool)
    hi = high_signal.reindex(d.index).fillna(False).astype(bool)
    if delay_sessions:
        lo = lo.shift(delay_sessions, fill_value=False)
        hi = hi.shift(delay_sessions, fill_value=False)
    end = end or d.date.max().date().isoformat()
    ix = np.flatnonzero(((d.date >= pd.Timestamp(start)) & (d.date <= pd.Timestamp(end))).to_numpy())
    if len(ix) < 2:
        raise ValueError("Insufficient rows in requested period")

    cash = float(capital)
    shares = 0
    entry_cost = 0.0
    entry_date = None
    equity_rows = []
    trade_rows = []
    conflicts = 0

    for k, i in enumerate(ix):
        row = d.iloc[i]
        if k > 0:
            prev = ix[k - 1]
            lo_now = bool(lo.iloc[prev])
            hi_now = bool(hi.iloc[prev])
            if lo_now and hi_now:
                conflicts += 1
                lo_now = hi_now = False
            if shares > 0 and hi_now:
                fill = float(row.raw_open) * (1 - slippage)
                notional = shares * fill
                proceeds = (
                    notional
                    - commission(notional, fee_rate)
                    - int(math.floor(notional * sell_tax_rate))
                )
                pnl = proceeds - entry_cost
                trade_rows.append({
                    "entry_date": entry_date,
                    "exit_date": row.date.date().isoformat(),
                    "shares": shares,
                    "entry_cost": entry_cost,
                    "exit_proceeds": proceeds,
                    "net_pnl": pnl,
                    "net_return": pnl / entry_cost,
                    "exit_reason": "high_signal_next_open",
                })
                cash += proceeds
                shares = 0
                entry_cost = 0.0
                entry_date = None
            elif shares == 0 and lo_now:
                fill = float(row.raw_open) * (1 + slippage)
                qty = int(cash // fill)
                while qty > 0 and qty * fill + commission(qty * fill, fee_rate) > cash + 1e-8:
                    qty -= 1
                if qty > 0:
                    entry_cost = qty * fill + commission(qty * fill, fee_rate)
                    cash -= entry_cost
                    shares = qty
                    entry_date = row.date.date().isoformat()

        equity_rows.append({
            "date": row.date.date().isoformat(),
            "equity": cash + shares * float(row.raw_close),
            "cash": cash,
            "shares": shares,
        })

    if shares:
        row = d.iloc[ix[-1]]
        fill = float(row.raw_close) * (1 - slippage)
        notional = shares * fill
        proceeds = notional - commission(notional, fee_rate) - int(math.floor(notional * sell_tax_rate))
        pnl = proceeds - entry_cost
        trade_rows.append({
            "entry_date": entry_date,
            "exit_date": row.date.date().isoformat(),
            "shares": shares,
            "entry_cost": entry_cost,
            "exit_proceeds": proceeds,
            "net_pnl": pnl,
            "net_return": pnl / entry_cost,
            "exit_reason": "end_of_test_close",
        })
        cash += proceeds
        equity_rows[-1].update({"equity": cash, "cash": cash, "shares": 0})

    curve = pd.DataFrame(equity_rows)
    curve["peak"] = curve.equity.cummax()
    curve["drawdown"] = curve.equity / curve.peak - 1
    trades = pd.DataFrame(trade_rows)
    elapsed = (pd.Timestamp(curve.date.iloc[-1]) - pd.Timestamp(curve.date.iloc[0])).days / 365.25
    final = float(curve.equity.iloc[-1])
    metrics = {
        "start": curve.date.iloc[0],
        "end": curve.date.iloc[-1],
        "initial_capital": capital,
        "final_equity": final,
        "total_return": final / capital - 1,
        "cagr": (final / capital) ** (1 / elapsed) - 1 if elapsed > 0 and final > 0 else np.nan,
        "max_drawdown": float(curve.drawdown.min()),
        "closed_trades": len(trades),
        "win_rate": float((trades.net_pnl > 0).mean()) if len(trades) else np.nan,
        "exposure": float((curve.shares > 0).mean()),
        "same_day_signal_conflicts_ignored": conflicts,
        "delay_sessions": delay_sessions,
        "slippage_each_side": slippage,
        "fee_rate_each_side": fee_rate,
        "sell_tax_rate": sell_tax_rate,
    }
    return metrics, trades, curve


def run(features_path: str, output_dir: str) -> dict:
    d = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    required = {
        "date", "raw_open", "raw_close", "relative_volume20", "return_1",
        "stoch_k14", "close_location", "obv_flow_ratio_5", "intraday_range_pct",
        "distance_sma_200", "macd_hist_change_3", "sma50_slope_10",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if d.date.duplicated().any():
        raise ValueError("Duplicate dates")

    low, high = build_candidate_signals(d)
    summary_rows = []
    trade_parts = []

    def add(label: str, **kwargs) -> None:
        metrics, trades, _ = simulate(d, low, high, **kwargs)
        metrics["test"] = label
        summary_rows.append(metrics)
        if len(trades):
            trades.insert(0, "test", label)
            trade_parts.append(trades)

    add("frozen_full_period")
    for label, start, end in (
        ("2016_2020", "2016-01-04", "2020-12-31"),
        ("2021_2026", "2021-01-01", "2026-11-09"),
        ("2023_2026_descriptive_only", "2023-01-01", "2026-11-09"),
    ):
        add(label, start=start, end=end)

    for delay in (1, 2):
        add(f"execution_delay_{delay}_session", delay_sessions=delay)
    for slip in (0.002, 0.003):
        add(f"slippage_{slip:.3f}_each_side", slippage=slip)

    neighbours = [
        ("low_relative_volume", (1.1, 1.2, 1.3)),
        ("low_return", (-0.008, -0.010, -0.012)),
        ("low_stoch", (25, 30, 35)),
        ("low_close_location", (0.20, 0.25, 0.30)),
        ("high_distance_sma200", (0.08, 0.10, 0.12)),
        ("high_return", (0.008, 0.010, 0.012)),
        ("high_close_location", (0.65, 0.70, 0.75)),
        ("high_intraday_range", (0.014, 0.016, 0.018)),
    ]
    neighbour_rows = []
    for name, values in neighbours:
        for value in values:
            kwargs = {name: value}
            lo, hi = build_candidate_signals(d, **kwargs)
            metrics, _, _ = simulate(d, lo, hi)
            neighbour_rows.append({
                "parameter": name,
                "value": value,
                "cagr": metrics["cagr"],
                "total_return": metrics["total_return"],
                "max_drawdown": metrics["max_drawdown"],
                "closed_trades": metrics["closed_trades"],
                "win_rate": metrics["win_rate"],
            })

    yearly_rows = []
    for year in range(2016, 2027):
        metrics, trades, _ = simulate(
            d, low, high, start=f"{year}-01-01", end=f"{year}-12-31"
        )
        metrics["year"] = year
        yearly_rows.append(metrics)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "tsmc_2330_rule_robustness_summary.csv", index=False)
    pd.DataFrame(neighbour_rows).to_csv(
        out / "tsmc_2330_rule_neighbour_sensitivity.csv", index=False
    )
    pd.DataFrame(yearly_rows).to_csv(
        out / "tsmc_2330_rule_yearly_returns.csv", index=False
    )
    if trade_parts:
        pd.concat(trade_parts, ignore_index=True).to_csv(
            out / "tsmc_2330_rule_robustness_trades.csv", index=False
        )
    meta = {
        "symbol": "2330",
        "candidate": "LOW capitulation + negative OBV; HIGH climax + extension/momentum fade",
        "period": [d.date.min().date().isoformat(), d.date.max().date().isoformat()],
        "execution": "all-in/all-out; next-session open; integer shares; no leverage or shorting",
        "costs": {
            "commission_each_side": FEE_RATE,
            "minimum_commission_ntd": MIN_COMMISSION,
            "sell_tax": SELL_TAX_RATE,
            "slippage_each_side": SLIPPAGE,
        },
        "sensitivity": "one-factor neighbouring thresholds only; no parameter selection",
        "limitations": [
            "Candidate thresholds were discovered after inspecting the same historical period; sensitivity is not independent validation.",
            "No cash dividends/corporate-action cashflows, official exchange price-limit lock/fill simulation, liquidity impact or broker-specific discounts.",
            "2026 is partial; the 2023-2026 segment is descriptive and not a clean untouched holdout.",
        ],
    }
    (out / "tsmc_2330_rule_robustness_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "summary": summary.to_dict(orient="records"),
        "neighbour_rows": len(neighbour_rows),
        "year_rows": len(yearly_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.features, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
