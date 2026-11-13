"""Continuous-account annual expanding OOF for causal trend re-entry.

This is an isolated experiment. It reuses the frozen baseline low/high event
rules (not a repeated grid search) and selects only among a small predeclared
re-entry family on each expanding historical prefix. Every yearly rule is
frozen before its validation year. All validation years are stitched into a
single account. No annual liquidation or account reset occurs.

Execution is long-only, all-in/all-out, integer shares. Slippage-adjusted
prices are rounded adversely to TWSE legal tick sizes. Missing official daily
price-limit data blocks an order. The final open position is marked to market
at the last close rather than forcibly sold.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tsmc_2330_40pct_rule_search import inputs, WIN_FLOOR

INITIAL_CAPITAL = 100_000.0
SLIPPAGE_BPS = (5, 10, 20, 30)
MIN_TRAIN_TRADES = 5
FIRST_VALIDATION_YEAR = 2015
LAST_VALIDATION_YEAR = 2026

# The fixed original rule is deliberately reused; previously completed
# low/high parameter searches are not repeated here.
FROZEN_BASE_PARAMS = {
    "low_return": -0.006,
    "low_relative_volume": 1.2,
    "high_close_location": 0.8,
    "high_distance_sma200": 0.12,
}

# Small predeclared family, chosen to test trend re-entry, not to expand
# the earlier ATR/trailing-stop or 54-combination entry/exit grids.
REENTRY_RULES = (
    "none",
    "break_prev_high",
    "rsi_recovery",
    "macd_momentum",
    "ema20_rsi_macd",
    "ema20_rsi_macd_cd10",
    "ema20_rsi_macd_cd20",
)


def commission(notional: float) -> int:
    return max(20, int(math.ceil(notional * 0.001425))) if notional > 0 else 0


def sell_tax(notional: float) -> int:
    return int(math.floor(notional * 0.003)) if notional > 0 else 0


def tick_size(price: float) -> float:
    """TWSE common-stock tick schedule; price band is checked per legal quote."""
    if price < 10:
        return 0.01
    if price < 50:
        return 0.05
    if price < 100:
        return 0.10
    if price < 500:
        return 0.50
    if price < 1000:
        return 1.00
    return 5.00


def _is_legal_tick(price: float) -> bool:
    if not np.isfinite(price) or price <= 0:
        return False
    # A quote exactly on a boundary belongs to the higher price band, but the
    # legal grid must also satisfy the price band's increment.
    return abs(price / tick_size(price) - round(price / tick_size(price))) < 1e-8


def adverse_fill(raw_open: float, side: str, slippage_bps: int) -> float:
    """Find the nearest legal quote no better than adverse percentage slippage."""
    target = raw_open * (1 + slippage_bps / 10000) if side == "BUY" else raw_open * (1 - slippage_bps / 10000)
    if side not in {"BUY", "SELL"}:
        raise ValueError(f"Unknown side: {side}")
    # Search legal quotes around target; the tick can change at a band boundary.
    # Iterate adjacent price grids until a quote satisfies both the band and tick.
    if side == "BUY":
        candidate = target
        for _ in range(100):
            tick = tick_size(candidate)
            candidate = math.ceil((candidate - 1e-10) / tick) * tick
            candidate = round(candidate, 2)
            if _is_legal_tick(candidate) and candidate + 1e-8 >= target:
                return candidate
            candidate = round(candidate + 0.01, 2)
        raise RuntimeError(f"Could not find legal BUY fill for {raw_open}")
    candidate = target
    for _ in range(100):
        tick = tick_size(candidate)
        candidate = math.floor((candidate + 1e-10) / tick) * tick
        candidate = round(candidate, 2)
        if _is_legal_tick(candidate) and candidate <= target + 1e-8:
            return candidate
        candidate = round(candidate - 0.01, 2)
    raise RuntimeError(f"Could not find legal SELL fill for {raw_open}")

def _signals(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    action = d["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    low = (
        (d["relative_volume20"] >= FROZEN_BASE_PARAMS["low_relative_volume"])
        & (d["return_1"] <= FROZEN_BASE_PARAMS["low_return"])
        & (d["stoch_k14"] <= 30)
        & (d["close_location"] <= 0.25)
        & (d["obv_flow_ratio_5"] <= 0)
    ).fillna(False).to_numpy(bool) & ~action
    high = (
        (d["return_1"] >= 0.01)
        & (d["close_location"] >= FROZEN_BASE_PARAMS["high_close_location"])
        & (d["intraday_range_pct"] >= 0.016)
        & (d["distance_sma_200"] >= FROZEN_BASE_PARAMS["high_distance_sma200"])
        & (d["macd_hist_change_3"] <= 0)
        & (d["sma50_slope_10"] > 0)
    ).fillna(False).to_numpy(bool) & ~action
    macd_prev = d["macd_hist_pct"].shift(1)
    rsi_prev = d["rsi14"].shift(1)
    prev_high = d["raw_high"].shift(1)
    common = d["raw_close"].notna() & d["raw_open"].notna() & ~d["corporate_action_window"].fillna(True).astype(bool)
    re = {
        "break_prev_high": (common & (d["raw_close"] > prev_high)).fillna(False).to_numpy(bool),
        "rsi_recovery": (common & (d["rsi14"] >= 50) & (d["rsi14"] > rsi_prev)).fillna(False).to_numpy(bool),
        "macd_momentum": (common & (d["macd_hist_pct"] > macd_prev)).fillna(False).to_numpy(bool),
        "ema20_rsi_macd": (
            common & (d["distance_ema_20"] > 0)
            & (d["rsi14"] >= 50)
            & (d["macd_hist_pct"] > macd_prev)
        ).fillna(False).to_numpy(bool),
        "ema20_rsi_macd_cd10": (
            common & (d["distance_ema_20"] > 0)
            & (d["rsi14"] >= 50)
            & (d["macd_hist_pct"] > macd_prev)
        ).fillna(False).to_numpy(bool),
        "ema20_rsi_macd_cd20": (
            common & (d["distance_ema_20"] > 0)
            & (d["rsi14"] >= 50)
            & (d["macd_hist_pct"] > macd_prev)
        ).fillna(False).to_numpy(bool),
    }
    return low, high, re


def _simulate(
    d: pd.DataFrame,
    low: np.ndarray,
    high: np.ndarray,
    reentry: dict[str, np.ndarray],
    rule_by_day: list[str],
    limits: dict[str, tuple[float, float]],
    dividends: dict[str, float],
    slippage_bps: int,
    start_index: int,
    end_index: int,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """One continuous event-driven portfolio account over [start_index,end_index]."""
    cash = float(initial_capital)
    shares = 0
    entry_date = None
    entry_cost = 0.0
    trade_dividend = 0.0
    pending_order = None
    last_exit_index = None
    trades = []
    curve = []
    blocked_missing = 0
    blocked_locked = 0
    blocked_tick_limit = 0
    total_fee = 0
    total_tax = 0
    total_slippage = 0.0
    total_dividend = 0.0
    exposure_days = 0

    for i in range(start_index, end_index + 1):
        row = d.iloc[i]
        day = row["date"].strftime("%Y-%m-%d")

        # Cash dividends are credited to shares held at previous close.
        if shares > 0 and day in dividends:
            amount = shares * dividends[day]
            cash += amount
            trade_dividend += amount
            total_dividend += amount

        # Execute only orders signalled on the previous close.
        if pending_order is not None:
            side = pending_order["side"]
            if day not in limits:
                blocked_missing += 1
                # For an exit, keep the pending sell alive; for an entry,
                # cancel to avoid inventing a future fill.
                if side == "BUY":
                    pending_order = None
            else:
                up, down = limits[day]
                op = float(row["raw_open"])
                hi = float(row["raw_high"])
                lo = float(row["raw_low"])
                if not all(np.isfinite(v) for v in (op, hi, lo, up, down)):
                    blocked_missing += 1
                    if side == "BUY":
                        pending_order = None
                elif (side == "BUY" and op >= up - 1e-8) or (side == "SELL" and op <= down + 1e-8):
                    blocked_locked += 1
                    # Keep locked SELL orders pending; cancel locked BUY orders.
                    if side == "BUY":
                        pending_order = None
                else:
                    fill = adverse_fill(op, side, slippage_bps)
                    # A modeled adverse fill beyond the official daily range is
                    # not a legal fill; fail closed rather than price-cap.
                    if fill > up + 1e-8 or fill < down - 1e-8:
                        blocked_tick_limit += 1
                        if side == "BUY":
                            pending_order = None
                    elif side == "BUY" and shares == 0:
                        qty = int(cash // fill)
                        while qty > 0 and qty * fill + commission(qty * fill) > cash + 1e-8:
                            qty -= 1
                        if qty > 0:
                            notional = qty * fill
                            fee = commission(notional)
                            cash -= notional + fee
                            shares = qty
                            entry_date = day
                            entry_cost = notional + fee
                            trade_dividend = 0.0
                            total_fee += fee
                            total_slippage += qty * (fill - op)
                        pending_order = None
                    elif side == "SELL" and shares > 0:
                        notional = shares * fill
                        fee = commission(notional)
                        tax = sell_tax(notional)
                        proceeds = notional - fee - tax
                        cash += proceeds
                        pnl = proceeds + trade_dividend - entry_cost
                        trades.append({
                            "entry_date": entry_date,
                            "exit_date": day,
                            "shares": shares,
                            "entry_cost": round(entry_cost, 2),
                            "exit_net_proceeds": round(proceeds, 2),
                            "cash_dividends": round(trade_dividend, 2),
                            "net_pnl": round(pnl, 2),
                            "net_return": pnl / entry_cost if entry_cost else None,
                            "forced_end_liquidation": False,
                        })
                        total_fee += fee
                        total_tax += tax
                        total_slippage += shares * (op - fill)
                        shares = 0
                        entry_date = None
                        entry_cost = 0.0
                        trade_dividend = 0.0
                        last_exit_index = i
                        pending_order = None

        close = float(row["raw_close"])
        equity = cash + shares * close
        curve.append({
            "date": day, "equity": equity, "cash": cash,
            "shares": shares, "close": close,
            "pending_order": pending_order["side"] if pending_order else None,
        })
        if shares > 0:
            exposure_days += 1

        # Signal at today's close for next available open. Avoid overwriting an
        # unfilled sell; when blocked at limit-down it remains pending.
        if pending_order is None and i < end_index:
            active_rule = rule_by_day[i]
            if shares == 0:
                can_reenter = active_rule != "none" and last_exit_index is not None
                if can_reenter:
                    cooldown = 10 if active_rule.endswith("cd10") else 20 if active_rule.endswith("cd20") else 0
                    if reentry[active_rule][i] and (i - last_exit_index) >= cooldown:
                        pending_order = {"side": "BUY", "reason": active_rule}
                elif low[i]:
                    pending_order = {"side": "BUY", "reason": "base_low"}
            elif high[i]:
                pending_order = {"side": "SELL", "reason": "base_high"}

    # Mark-to-market, do not force-sell at year ends or final observation.
    curve_frame = pd.DataFrame(curve)
    eq = curve_frame["equity"].astype(float)
    peak = eq.cummax()
    drawdown = eq / peak - 1
    years = (pd.Timestamp(curve_frame["date"].iloc[-1]) - pd.Timestamp(curve_frame["date"].iloc[0])).days / 365.25
    final_equity = float(eq.iloc[-1])
    cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 and final_equity > 0 else None
    trade_frame = pd.DataFrame(trades)
    return {
        "period_start": curve_frame["date"].iloc[0],
        "period_end": curve_frame["date"].iloc[-1],
        "initial_capital": initial_capital,
        "final_equity_mark_to_market": final_equity,
        "cagr_mark_to_market": cagr,
        "max_drawdown": float(drawdown.min()),
        "trade_count": len(trades),
        "win_rate": float((trade_frame.net_pnl > 0).mean()) if len(trade_frame) else None,
        "exposure_ratio": exposure_days / len(curve_frame),
        "total_commission": total_fee,
        "total_sell_tax": total_tax,
        "estimated_slippage_cost": total_slippage,
        "total_cash_dividends": total_dividend,
        "blocked_missing_price_limit": blocked_missing,
        "blocked_locked_limit": blocked_locked,
        "blocked_slippage_outside_limit": blocked_tick_limit,
        "trades": trades,
        "equity_curve": curve,
    }


def _score(metrics: dict) -> tuple:
    cagr = metrics.get("cagr_mark_to_market")
    win = metrics.get("win_rate")
    return (
        cagr if cagr is not None else -1,
        win if win is not None else -1,
        -abs(metrics.get("max_drawdown") or -1),
    )


def _select_rule(d: pd.DataFrame, low, high, reentry, limits, dividends, end_i: int) -> dict:
    candidates = []
    rule_by_day = ["none"] * len(d)
    for rule in REENTRY_RULES:
        config = ["none"] * len(d)
        # For training prefix, apply the same single candidate rule on every day.
        config[:end_i + 1] = [rule] * (end_i + 1)
        metrics = _simulate(
            d, low, high, reentry, config, limits, dividends, 10,
            0, end_i, INITIAL_CAPITAL,
        )
        eligible = (
            metrics["trade_count"] >= MIN_TRAIN_TRADES
            and metrics["win_rate"] is not None
            and metrics["win_rate"] >= WIN_FLOOR
            and metrics["cagr_mark_to_market"] is not None
        )
        candidates.append({
            "rule": rule,
            "eligible": bool(eligible),
            "metrics": {k: v for k, v in metrics.items() if k not in {"trades", "equity_curve"}},
        })
    eligible = [x for x in candidates if x["eligible"]]
    chosen = max(eligible, key=lambda x: _score(x["metrics"])) if eligible else None
    return {
        "selected_rule": chosen["rule"] if chosen else None,
        "selected_metrics": chosen["metrics"] if chosen else None,
        "eligible_count": len(eligible),
        "candidate_count": len(candidates),
        "candidate_diagnostics": candidates,
    }


def run(features_path: str, limits_path: str, dividends_path: str, output_dir: str) -> dict:
    d, limits, dividends = inputs(features_path, limits_path, dividends_path)
    required = {
        "raw_open", "raw_high", "raw_low", "raw_close", "corporate_action_window",
        "return_1", "relative_volume20", "stoch_k14", "close_location",
        "obv_flow_ratio_5", "intraday_range_pct", "distance_sma_200",
        "macd_hist_change_3", "macd_hist_pct", "rsi14", "distance_ema_20",
        "sma50_slope_10",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Missing required causal columns: {missing}")
    d = d.sort_values("date").reset_index(drop=True)
    years = sorted(d.date.dt.year.unique().astype(int))
    if not years or years[0] > 2010 or years[-1] < 2015:
        raise RuntimeError("Frozen source must cover 2010-2015+")
    low, high, reentry = _signals(d)
    annual_selections = []
    selection_by_year = {}
    for year in range(FIRST_VALIDATION_YEAR, min(LAST_VALIDATION_YEAR, years[-1]) + 1):
        mask = np.flatnonzero((d.date.dt.year < year).to_numpy())
        if len(mask) == 0:
            continue
        end_i = int(mask[-1])
        selection = _select_rule(d, low, high, reentry, limits, dividends, end_i)
        selection_by_year[year] = selection["selected_rule"] or "none"
        annual_selections.append({
            "validation_year": year,
            "training_start": d.date.iloc[0].date().isoformat(),
            "training_end": d.date.iloc[end_i].date().isoformat(),
            "status": "selected" if selection["selected_rule"] else "no_eligible_rule_use_base_entry_only",
            **selection,
            "win_rate_floor": WIN_FLOOR,
            "minimum_training_trades": MIN_TRAIN_TRADES,
            "frozen_before_validation_year": True,
        })

    # Stitch yearly frozen choices into one continuous OOF path.
    rule_by_day = ["none"] * len(d)
    for i, row in d.iterrows():
        y = int(row.date.year)
        if FIRST_VALIDATION_YEAR <= y <= LAST_VALIDATION_YEAR:
            rule_by_day[i] = selection_by_year.get(y, "none")

    # Validation starts at 2015 and uses the same one account through the last
    # available session. No annual reset, no year-end liquidation.
    validation_indices = np.flatnonzero((d.date.dt.year >= FIRST_VALIDATION_YEAR).to_numpy())
    start_i, end_i = int(validation_indices[0]), int(validation_indices[-1])
    result_by_slippage = {}
    trades_by_slippage = {}
    equity_by_slippage = {}
    annual_by_slippage = {}
    for slip in SLIPPAGE_BPS:
        sim = _simulate(d, low, high, reentry, rule_by_day, limits, dividends, slip, start_i, end_i)
        result_by_slippage[f"{slip}bps"] = {k: v for k, v in sim.items() if k not in {"trades", "equity_curve"}}
        trades_by_slippage[f"{slip}bps"] = [{"trade_number":n+1,**x} for n,x in enumerate(sim["trades"])]
        equity_by_slippage[f"{slip}bps"] = sim["equity_curve"]
        c = pd.DataFrame(sim["equity_curve"])
        c["date"] = pd.to_datetime(c["date"])
        c["year"] = c.date.dt.year
        annual = []
        prior = INITIAL_CAPITAL
        for y, g in c.groupby("year", sort=True):
            end_eq=float(g.equity.iloc[-1])
            ytrades=pd.DataFrame(sim["trades"])
            if not ytrades.empty:
                ytrades=ytrades[pd.to_datetime(ytrades.exit_date).dt.year==y]
            annual.append({
                "year":int(y),"start_equity":prior,"end_equity":end_eq,
                "annual_return":end_eq/prior-1 if prior else None,
                "trade_count":len(ytrades),
                "win_rate":float((ytrades.net_pnl>0).mean()) if len(ytrades) else None,
                "partial_year":int(y)==int(c.year.max()) and g.date.max().month<12,
            })
            prior=end_eq
        annual_by_slippage[f"{slip}bps"]=annual

    summary = {
        "symbol":"2330",
        "data_period":[d.date.min().date().isoformat(),d.date.max().date().isoformat()],
        "seed_period":"2010-2014",
        "validation_period":"2015-2026 (latest partial)",
        "protocol":"Expanding annual walk-forward; every year uses only 2010..Y-1 for rule selection; single continuous account; mark-to-market final position; no year-end liquidation.",
        "selection_objective":"Maximize 10bps training-prefix CAGR subject to >=5 closed trades and win rate >=86.9565%; tie-break by higher win rate then lower MDD.",
        "win_rate_floor":WIN_FLOOR,
        "frozen_base_entry_exit_params":FROZEN_BASE_PARAMS,
        "predeclared_reentry_rules":list(REENTRY_RULES),
        "annual_frozen_rule_selections":annual_selections,
        "continuous_oof_metrics_by_cost":result_by_slippage,
        "annual_oof_metrics_by_cost":annual_by_slippage,
        "target_status":"UNMET unless at least 50% CAGR and >=86.9565% win rate at every cost scenario; MDD goal <25% is separately reported.",
        "data_integrity":"Frozen source Artifact 13914111364 and original OOF Artifact 13923037627; no new market data fetched.",
        "limitations":[
            "The re-entry family was chosen after historical event diagnostics on inspected history; resulting full-period statistics are exploratory and not pristine blind OOS.",
            "Selection on historical prefixes reduces direct future leakage but research-family design itself remains exposed to historical inspection.",
            "Official price-limit history in the frozen source artifact is not yet independently row-by-row reconciled to TWSE primary files.",
            "Daily OHLC cannot model order-book queue priority or true historical spread.",
            "Fractional odd-lot fill assumptions are represented with integer share quantities but need broker-market feasibility reconciliation."
        ],
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/"tsmc_2330_trend_reentry_annual_oof.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    pd.DataFrame(annual_selections).to_csv(out/"tsmc_2330_trend_reentry_fold_selections.csv",index=False)
    for slip in SLIPPAGE_BPS:
        pd.DataFrame(trades_by_slippage[f"{slip}bps"]).to_csv(out/f"tsmc_2330_trend_reentry_trades_{slip}bps.csv",index=False)
        pd.DataFrame(equity_by_slippage[f"{slip}bps"]).to_csv(out/f"tsmc_2330_trend_reentry_equity_{slip}bps.csv",index=False)
        pd.DataFrame(annual_by_slippage[f"{slip}bps"]).to_csv(out/f"tsmc_2330_trend_reentry_annual_{slip}bps.csv",index=False)
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",required=True)
    ap.add_argument("--limits",required=True)
    ap.add_argument("--dividends",required=True)
    ap.add_argument("--output",default="artifacts/tsmc_2330_50cagr_trend_reentry")
    a=ap.parse_args()
    result=run(a.features,a.limits,a.dividends,a.output)
    print(json.dumps({k:v for k,v in result.items() if k!="annual_frozen_rule_selections"},ensure_ascii=False,indent=2,default=float))


if __name__=="__main__":
    main()
