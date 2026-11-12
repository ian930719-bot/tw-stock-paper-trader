"""Execution-aware, single-symbol TSMC 2330 swing-rule backtest.

Signals are calculated at close t and executed no earlier than the next
available 2330 session open. Uses raw execution OHLC, FinMind daily exchange
price-limit fields, cash-dividend entitlement dates, standard commission,
sell-side securities transaction tax, and configurable per-side slippage.
No leverage, short selling, or intraday path assumptions are used.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

COMMISSION_RATE = 0.001425
SELL_TAX_RATE = 0.003
MIN_COMMISSION_NTD = 20
SLIPPAGE_BPS = (5, 10, 20, 30)
INITIAL_CAPITAL = 100_000.0
FEATURES_REQUIRED = {"date", "raw_open", "raw_high", "raw_low", "raw_close",
                     "corporate_action_window"}


def commission(notional: float) -> int:
    if notional <= 0:
        return 0
    return max(MIN_COMMISSION_NTD, int(math.ceil(notional * COMMISSION_RATE)))


def sell_tax(notional: float) -> int:
    if notional <= 0:
        return 0
    return int(math.floor(notional * SELL_TAX_RATE))


def is_limit_locked(side: str, open_price: float, high: float, low: float,
                    limit_up: float, limit_down: float) -> bool:
    """Fail closed when the opening auction is locked at the relevant limit."""
    values = (open_price, high, low, limit_up, limit_down)
    if not all(np.isfinite(v) for v in values):
        return True
    if side == "BUY":
        # Conservative: an opening auction at limit-up is treated as unfillable,
        # regardless of the later intraday low (which is not known at order time).
        return limit_up > 0 and open_price >= limit_up - 1e-8
    if side == "SELL":
        # Conservative: an opening auction at limit-down is treated as unfillable.
        return limit_down > 0 and open_price <= limit_down + 1e-8
    raise ValueError(f"Unknown side: {side}")


def trailing_drawdown_stop_hit(close: float, peak_close: float, drawdown_pct: float) -> bool:
    """Causal close-based trailing stop; execution is scheduled for next session open."""
    if not all(np.isfinite(v) for v in (close, peak_close, drawdown_pct)):
        return False
    if close <= 0 or peak_close <= 0 or not 0 < drawdown_pct < 1:
        return False
    return close <= peak_close * (1.0 - drawdown_pct)


def trailing_atr_stop_hit(close: float, peak_close: float, atr_pct: float,
                          multiple: float) -> bool:
    if not all(np.isfinite(v) for v in (close, peak_close, atr_pct, multiple)):
        return False
    if close <= 0 or peak_close <= 0 or atr_pct <= 0 or multiple <= 0:
        return False
    stop_level = peak_close - multiple * atr_pct * close
    return close <= stop_level


def buy_quantity(cash: float, execution_price: float) -> int:
    if cash <= 0 or execution_price <= 0:
        return 0
    qty = int(cash // execution_price)
    while qty > 0:
        notional = qty * execution_price
        if notional + commission(notional) <= cash + 1e-8:
            return qty
        qty -= 1
    return 0


def build_signals(features: pd.DataFrame, frozen_rules: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(features)
    low_signal = np.zeros(n, dtype=bool)
    high_signal = np.zeros(n, dtype=bool)
    for kind, target in (("LOW", low_signal), ("HIGH", high_signal)):
        rule = frozen_rules.get("rules", {}).get(kind, {})
        if not rule.get("selected"):
            continue
        conditions = []
        for name in rule["features"]:
            if name not in features or name not in rule["training_thresholds"]:
                raise ValueError(f"Frozen rule missing feature/threshold: {name}")
            values = pd.to_numeric(features[name], errors="coerce")
            cutoff = float(rule["training_thresholds"][name])
            condition = values.le(cutoff) if kind == "LOW" else values.ge(cutoff)
            conditions.append(condition.fillna(False).to_numpy())
        target[:] = np.logical_and.reduce(conditions)
    if "corporate_action_window" in features:
        action_window = features["corporate_action_window"].fillna(True).astype(bool).to_numpy()
        low_signal &= ~action_window
        high_signal &= ~action_window
    return low_signal, high_signal


def load_inputs(features_path: str, rules_path: str, limits_path: str,
                dividends_path: str) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict[str, float]]:
    bars = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if not FEATURES_REQUIRED.issubset(bars.columns):
        raise ValueError(f"Feature file missing raw execution columns: {sorted(FEATURES_REQUIRED - set(bars.columns))}")
    rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    limits = pd.read_csv(limits_path, dtype={"stock_id": str})
    required_limits = {"date", "stock_id", "reference_price", "limit_up", "limit_down"}
    if not required_limits.issubset(limits.columns):
        raise ValueError(f"Price-limit file missing columns: {sorted(required_limits - set(limits.columns))}")
    limits = limits.loc[limits["stock_id"].astype(str).str.strip() == "2330"].copy()
    limits["date"] = pd.to_datetime(limits["date"], errors="coerce").dt.normalize()
    for col in ("reference_price", "limit_up", "limit_down"):
        limits[col] = pd.to_numeric(limits[col], errors="coerce")
    limits = limits.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    if limits.empty:
        raise ValueError("No official daily price-limit rows for TSMC 2330")
    limit_lookup = {
        row.date.strftime("%Y-%m-%d"): (float(row.limit_up), float(row.limit_down))
        for row in limits.itertuples(index=False)
        if np.isfinite(row.limit_up) and np.isfinite(row.limit_down)
    }

    dividends = pd.read_csv(dividends_path)
    required_divs = {"date", "stock_or_cache_dividend", "stock_and_cache_dividend"}
    if not required_divs.issubset(dividends.columns):
        raise ValueError(f"Dividend file missing columns: {sorted(required_divs - set(dividends.columns))}")
    dividends = dividends.loc[dividends["stock_or_cache_dividend"].astype(str).str.strip() == "息"].copy()
    dividends["date"] = pd.to_datetime(dividends["date"], errors="coerce").dt.normalize()
    dividends["stock_and_cache_dividend"] = pd.to_numeric(
        dividends["stock_and_cache_dividend"], errors="coerce"
    )
    dividends = dividends.dropna(subset=["date", "stock_and_cache_dividend"])
    dividend_lookup = {
        row.date.strftime("%Y-%m-%d"): float(row.stock_and_cache_dividend)
        for row in dividends.itertuples(index=False)
    }
    return bars, rules, limits, dividend_lookup


def _simulate_period(bars: pd.DataFrame, low_signal: np.ndarray, high_signal: np.ndarray,
                     limit_lookup: dict, dividend_lookup: dict[str, float],
                     start_date: str, end_date: str, slippage_bps: int,
                     initial_capital: float, strategy: bool,
                     atr_stop_multiple: float | None = None,
                     atr_stop_mask: np.ndarray | None = None,
                     trailing_drawdown_pct: float | None = None) -> dict:
    dates = bars["date"].dt.normalize()
    period_idx = np.flatnonzero(
        (dates >= pd.Timestamp(start_date)).to_numpy()
        & (dates <= pd.Timestamp(end_date)).to_numpy()
    )
    if len(period_idx) < 2:
        raise ValueError(f"Insufficient 2330 rows for period {start_date} to {end_date}")
    first, last = int(period_idx[0]), int(period_idx[-1])
    period_bars = bars.iloc[first:last + 1].reset_index(drop=True)
    period_low = low_signal[first:last + 1]
    period_high = high_signal[first:last + 1]
    period_atr_stop_mask = (
        np.ones(len(period_bars), dtype=bool) if atr_stop_mask is None
        else np.asarray(atr_stop_mask, dtype=bool)[first:last + 1]
    )
    if len(period_atr_stop_mask) != len(period_bars):
        raise ValueError("ATR stop mask must align with the full bars frame")
    if trailing_drawdown_pct is not None and not 0 < trailing_drawdown_pct < 1:
        raise ValueError("trailing_drawdown_pct must be between 0 and 1")
    slip = slippage_bps / 10_000.0
    cash = float(initial_capital)
    shares = 0
    peak_close = None
    pending_buy = not strategy
    pending_sell = False
    entry_date = None
    entry_cash_cost = 0.0
    trade_dividends = 0.0
    trades = []
    equity_rows = []
    total_dividends = 0.0
    total_commission = 0
    total_tax = 0
    total_slippage = 0.0
    missing_limit_orders = 0
    blocked_limit_orders = 0
    exposure_days = 0

    def attempt_buy(i: int) -> bool:
        nonlocal cash, shares, entry_date, entry_cash_cost, trade_dividends, peak_close
        nonlocal total_commission, total_slippage, blocked_limit_orders, missing_limit_orders
        row = period_bars.iloc[i]
        day = row["date"].strftime("%Y-%m-%d")
        if day not in limit_lookup:
            missing_limit_orders += 1
            return False
        up, down = limit_lookup[day]
        if is_limit_locked("BUY", float(row.raw_open), float(row.raw_high),
                           float(row.raw_low), up, down):
            blocked_limit_orders += 1
            return False
        raw_open = float(row.raw_open)
        fill = raw_open * (1 + slip)
        qty = buy_quantity(cash, fill)
        if qty <= 0:
            return False
        notional = qty * fill
        fee = commission(notional)
        cash -= notional + fee
        shares = qty
        peak_close = raw_open
        entry_date = row["date"].date().isoformat()
        entry_cash_cost = notional + fee
        trade_dividends = 0.0
        total_commission += fee
        total_slippage += qty * (fill - raw_open)
        return True

    def attempt_sell(i: int, forced: bool = False) -> bool:
        nonlocal cash, shares, entry_date, entry_cash_cost, trade_dividends, peak_close
        nonlocal total_commission, total_tax, total_slippage, blocked_limit_orders, missing_limit_orders
        row = period_bars.iloc[i]
        day = row["date"].strftime("%Y-%m-%d")
        if not forced:
            if day not in limit_lookup:
                missing_limit_orders += 1
                return False
            up, down = limit_lookup[day]
            if is_limit_locked("SELL", float(row.raw_open), float(row.raw_high),
                               float(row.raw_low), up, down):
                blocked_limit_orders += 1
                return False
            raw_price = float(row.raw_open)
        else:
            raw_price = float(row.raw_close)
        fill = raw_price * (1 - slip)
        notional = shares * fill
        fee = commission(notional)
        tax = sell_tax(notional)
        proceeds = notional - fee - tax
        cash += proceeds
        pnl = proceeds + trade_dividends - entry_cash_cost
        trades.append({
            "entry_date": entry_date,
            "exit_date": row["date"].date().isoformat(),
            "shares": shares,
            "entry_cost": round(entry_cash_cost, 2),
            "exit_net_proceeds": round(proceeds, 2),
            "cash_dividends": round(trade_dividends, 2),
            "net_pnl": round(pnl, 2),
            "net_return": pnl / entry_cash_cost if entry_cash_cost else None,
            "forced_end_liquidation": bool(forced),
        })
        total_commission += fee
        total_tax += tax
        total_slippage += shares * (raw_price - fill)
        shares = 0
        peak_close = None
        entry_date = None
        entry_cash_cost = 0.0
        trade_dividends = 0.0
        return True

    for i, row in period_bars.iterrows():
        day = row["date"].strftime("%Y-%m-%d")
        # Dividend entitlement belongs to shares held at the prior session close.
        if shares > 0 and day in dividend_lookup:
            credit = shares * dividend_lookup[day]
            cash += credit
            trade_dividends += credit
            total_dividends += credit

        # Execute orders generated from the previous close.
        if pending_sell and shares > 0:
            if attempt_sell(i):
                pending_sell = False
        elif pending_buy and shares == 0:
            if attempt_buy(i):
                pending_buy = False
            elif strategy:
                # Strategy entry is cancelled if it cannot fill at the next open.
                pending_buy = False
            # Buy-and-hold baseline keeps its initial order pending until fill.

        close = float(row.raw_close)
        equity = cash + shares * close
        equity_rows.append({
            "date": day, "equity": equity, "cash": cash,
            "shares": shares, "close": close,
        })
        if shares > 0:
            exposure_days += 1
            peak_close = close if peak_close is None else max(peak_close, close)
            if trailing_drawdown_pct is not None and trailing_drawdown_stop_hit(
                close, peak_close, float(trailing_drawdown_pct)
            ):
                pending_sell = True
            if atr_stop_multiple is not None and period_atr_stop_mask[i]:
                atr_pct = pd.to_numeric(pd.Series([row.get("atr_pct14", np.nan)]), errors="coerce").iloc[0]
                if np.isfinite(atr_pct) and atr_pct > 0:
                    if trailing_atr_stop_hit(close, peak_close, float(atr_pct), float(atr_stop_multiple)):
                        pending_sell = True

        if strategy and i < len(period_bars) - 1:
            if shares == 0 and period_low[i]:
                pending_buy = True
            elif shares > 0 and not pending_sell and period_high[i]:
                pending_sell = True

    # Liquidate any remaining position at the final close for comparable
    # calendar-period equity; this is an explicit end-of-test assumption.
    if shares > 0:
        attempt_sell(len(period_bars) - 1, forced=True)
        equity_rows[-1]["equity"] = cash
        equity_rows[-1]["cash"] = cash
        equity_rows[-1]["shares"] = 0

    curve = pd.DataFrame(equity_rows)
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1
    final_equity = float(curve["equity"].iloc[-1])
    elapsed_days = (pd.Timestamp(curve["date"].iloc[-1]) - pd.Timestamp(curve["date"].iloc[0])).days
    years = elapsed_days / 365.25 if elapsed_days > 0 else 0
    cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 and final_equity > 0 else None
    trade_frame = pd.DataFrame(trades)
    return {
        "period_start": curve["date"].iloc[0],
        "period_end": curve["date"].iloc[-1],
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return": final_equity / initial_capital - 1,
        "cagr": cagr,
        "max_drawdown": float(curve["drawdown"].min()),
        "trade_count": int(len(trades)),
        "win_rate": float((trade_frame["net_pnl"] > 0).mean()) if len(trade_frame) else None,
        "average_trade_return": float(trade_frame["net_return"].mean()) if len(trade_frame) else None,
        "exposure_ratio": exposure_days / len(curve),
        "total_cash_dividends": total_dividends,
        "total_commission": total_commission,
        "total_sell_tax": total_tax,
        "estimated_slippage_cost": total_slippage,
        "blocked_limit_orders": blocked_limit_orders,
        "missing_price_limit_orders": missing_limit_orders,
        "trades": trades,
        "equity_curve": curve.to_dict(orient="records"),
    }


def run(features_path: str, rules_path: str, limits_path: str, dividends_path: str,
        output_dir: str, initial_capital: float = INITIAL_CAPITAL) -> dict:
    bars, rules, limits, dividend_lookup = load_inputs(
        features_path, rules_path, limits_path, dividends_path
    )
    low_signal, high_signal = build_signals(bars, rules)
    limit_lookup = {
        row.date.strftime("%Y-%m-%d"): (float(row.limit_up), float(row.limit_down))
        for row in limits.itertuples(index=False)
        if np.isfinite(row.limit_up) and np.isfinite(row.limit_down)
    }
    min_limit_coverage = float(bars["date"].dt.strftime("%Y-%m-%d").isin(limit_lookup).mean())
    if min_limit_coverage < 0.98:
        raise RuntimeError(f"Daily price-limit coverage too low for execution simulation: {min_limit_coverage:.3%}")

    periods = {
        "validation_selection_period": ("2017-01-01", "2021-12-31"),
        "historical_holdout": ("2022-01-01", bars["date"].max().date().isoformat()),
    }
    results = {}
    for period_name, (start, end) in periods.items():
        results[period_name] = {}
        for slip in SLIPPAGE_BPS:
            results[period_name][f"strategy_{slip}bps"] = _simulate_period(
                bars, low_signal, high_signal, limit_lookup, dividend_lookup,
                start, end, slip, initial_capital, True,
            )
            results[period_name][f"buy_hold_{slip}bps"] = _simulate_period(
                bars, low_signal, high_signal, limit_lookup, dividend_lookup,
                start, end, slip, initial_capital, False,
            )

    summary = {
        "symbol": "2330",
        "rule_source": "frozen rules selected using 2010-2016 training and 2017-2021 validation",
        "signal_timing": "signal at close t; execute at next available 2330 session open",
        "execution_price": "raw OHLC; configured adverse slippage applied per side",
        "price_limit_source": "TWSE official Rule 63 and tick-size schedule; ex-dividend reference prices from FinMind; latest TWSE TWT84U snapshot cross-check when available",
        "price_limit_coverage": min_limit_coverage,
        "dividend_source": "FinMind TaiwanStockDividendResult; cash-only (息) events credited to shares held at prior session close",
        "commission_rate_per_side": COMMISSION_RATE,
        "minimum_commission_ntd": MIN_COMMISSION_NTD,
        "sell_transaction_tax_rate": SELL_TAX_RATE,
        "slippage_scenarios_bps_per_side": list(SLIPPAGE_BPS),
        "initial_capital_per_period": initial_capital,
        "position_policy": "long-only, fully invested, no leverage, no shorting, one position at a time; 1-share granularity",
        "exit_policy": "frozen high-point rule only; no stop loss or discretionary exit",
        "end_of_period": "remaining shares liquidated at final close with adverse slippage and costs",
        "periods": periods,
        "limit_lock_policy": "blocked buy at locked limit-up is cancelled; sell at locked limit-down remains pending",
        "historical_limit_regime": "7% through 2015-05-29; 10% from 2015-06-01; tick-size rounding follows TWSE schedule",
        "cash_interest": 0,
        "rule_parameters": rules.get("rules", {}),
        "results": {
            period: {
                key: {k: v for k, v in result.items() if k not in {"trades", "equity_curve"}}
                for key, result in period_results.items()
            }
            for period, period_results in results.items()
        },
        "validation_warning": "2017-2021 was used to select the rules; its strategy results are selection-period diagnostics, not unbiased validation.",
        "holdout_warning": "2022+ was inspected in earlier pivot/classifier diagnostics and is not a pristine blind holdout. Historical results are exploratory; forward paper validation remains required.",
        "limitations": [
            "FinMind data has not yet been independently reconciled row-by-row against TWSE official files.",
            "Daily OHLC cannot model queue priority, intraday order-book liquidity, or exact fills.",
            "Slippage is a scenario assumption, not measured historical bid-ask spread.",
            "Broker-specific commission discounts are not applied.",
            "No cash interest or financing is modeled.",
            "One-share granularity is a capital-sizing proxy, not proof of actual odd-lot fills; the latest TWT84U snapshot AllowOddLotTrade field reported 不可 and requires reconciliation before live use.",
        ],
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tsmc_2330_execution_backtest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for period, period_results in results.items():
        for key, result in period_results.items():
            pd.DataFrame(result["trades"]).to_csv(out / f"{period}_{key}_trades.csv", index=False)
            pd.DataFrame(result["equity_curve"]).to_csv(out / f"{period}_{key}_equity.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--limits", required=True)
    parser.add_argument("--dividends", required=True)
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.rules, args.limits, args.dividends,
                         args.output, args.initial_capital), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
