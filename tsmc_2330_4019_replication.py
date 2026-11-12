"""Replication audit for the handoff's exploratory ~40.19% candidate.

The rule set is frozen from the user-provided handoff; no parameter grid is run.
This is a historical replication on already-inspected 2010-2026 data, not a
pristine blind OOF claim. Uses the isolated branch's TWSE tick-rounded adverse
fills, official daily price limits, cash dividends, Taiwan fees and tax.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from tsmc_2330_40pct_rule_search import inputs, make_signals
from tsmc_2330_trend_reentry_annual_oof import (
    INITIAL_CAPITAL, SLIPPAGE_BPS, adverse_fill, commission, sell_tax,
)

RULES = {
    "low_return": -0.006,
    "low_relative_volume": 1.2,
    "high_close_location": 0.8,
    "high_distance_sma200": 0.12,
    "low_exclude_distance_sma50_gt": 0.04,
    "risk_lookback_sessions": 80,
    "risk_below_sma200_min_sessions": 60,
    "risk_rsi_below": 40,
    "risk_exit_cooldown_sessions": 70,
    "high_immediate_exit_if_distance_sma20_below": 0.0,
    "high_immediate_exit_if_rsi_below": 58,
    "high_wait_exit_if_rsi_below": 48,
    "high_wait_max_sessions": 14,
}


def build_candidate_signals(d: pd.DataFrame):
    required = {
        "date", "raw_open", "raw_high", "raw_low", "raw_close",
        "corporate_action_window", "distance_sma_20", "distance_sma_50",
        "distance_sma_200", "rsi14", "relative_volume20", "return_1",
        "stoch_k14", "close_location", "obv_flow_ratio_5",
        "intraday_range_pct", "macd_hist_change_3", "sma50_slope_10",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Missing candidate features: {missing}")
    low, high = make_signals(
        d,
        RULES["low_return"],
        RULES["low_relative_volume"],
        RULES["high_close_location"],
        RULES["high_distance_sma200"],
    )
    action = d["corporate_action_window"].fillna(True).astype(bool).to_numpy()
    low &= ~(d["distance_sma_50"] > RULES["low_exclude_distance_sma50_gt"]).fillna(False).to_numpy(bool)
    low &= ~action
    high &= ~action

    below = (d["distance_sma_200"] < 0).fillna(False).astype(int)
    below_count = below.rolling(
        RULES["risk_lookback_sessions"],
        min_periods=RULES["risk_lookback_sessions"],
    ).sum()
    risk = (
        (below_count >= RULES["risk_below_sma200_min_sessions"])
        & (d["rsi14"] < RULES["risk_rsi_below"])
        & ~d["corporate_action_window"].fillna(True).astype(bool)
    ).fillna(False).to_numpy(bool)
    return low, high, risk


def simulate(
    d: pd.DataFrame,
    low: np.ndarray,
    high: np.ndarray,
    risk: np.ndarray,
    limits: dict[str, tuple[float, float]],
    dividends: dict[str, float],
    slippage_bps: int,
    start_i: int,
    end_i: int,
    initial_capital: float = INITIAL_CAPITAL,
    risk_exit_policy: str = "per_position",
) -> dict:
    if risk_exit_policy not in {"per_position", "per_regime_latched", "recovery_required"}:
        raise ValueError(f"Unknown risk_exit_policy: {risk_exit_policy}")
    cash = float(initial_capital)
    shares = 0
    entry_date = None
    entry_cost = 0.0
    trade_dividend = 0.0
    pending = None
    risk_exit_fired = False
    cooldown_until = -1
    high_wait_start = None
    trades = []
    curve = []
    total_commission = total_tax = total_dividend = 0
    total_slippage = 0.0
    missing_limit = locked_limit = outside_limit = 0
    exposure_days = 0
    risk_exit_count = high_exit_count = 0

    for i in range(start_i, end_i + 1):
        # Alternative interpretation: one risk exit per continuous weak regime;
        # re-arm only when the actual risk condition has recovered.
        if risk_exit_policy in {"per_regime_latched", "recovery_required"} and not risk[i]:
            risk_exit_fired = False
        row = d.iloc[i]
        day = row.date.strftime("%Y-%m-%d")

        if shares > 0 and day in dividends:
            credit = shares * dividends[day]
            cash += credit
            trade_dividend += credit
            total_dividend += credit

        if pending is not None:
            side = pending["side"]
            reason = pending["reason"]
            if day not in limits:
                missing_limit += 1
                if side == "BUY":
                    pending = None
            else:
                up, down = limits[day]
                op, hi, lo = float(row.raw_open), float(row.raw_high), float(row.raw_low)
                if not all(np.isfinite(x) for x in (op, hi, lo, up, down)):
                    missing_limit += 1
                    if side == "BUY":
                        pending = None
                elif (side == "BUY" and op >= up - 1e-8) or (side == "SELL" and op <= down + 1e-8):
                    locked_limit += 1
                    if side == "BUY":
                        pending = None
                else:
                    fill = adverse_fill(op, side, slippage_bps)
                    if fill > up + 1e-8 or fill < down - 1e-8:
                        outside_limit += 1
                        if side == "BUY":
                            pending = None
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
                            if risk_exit_policy == "per_position":
                                risk_exit_fired = False
                            high_wait_start = None
                            total_commission += fee
                            total_slippage += qty * (fill - op)
                        pending = None
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
                            "exit_reason": reason,
                            "shares": shares,
                            "entry_cost": round(entry_cost, 2),
                            "exit_net_proceeds": round(proceeds, 2),
                            "cash_dividends": round(trade_dividend, 2),
                            "net_pnl": round(pnl, 2),
                            "net_return": pnl / entry_cost if entry_cost else None,
                            "forced_end_liquidation": False,
                        })
                        total_commission += fee
                        total_tax += tax
                        total_slippage += shares * (op - fill)
                        if reason == "sma200_risk_exit":
                            risk_exit_count += 1
                            cooldown_until = i + RULES["risk_exit_cooldown_sessions"]
                        else:
                            high_exit_count += 1
                        shares = 0
                        entry_date = None
                        entry_cost = 0.0
                        trade_dividend = 0.0
                        high_wait_start = None
                        pending = None

        close = float(row.raw_close)
        equity = cash + shares * close
        curve.append({
            "date": day, "equity": equity, "cash": cash,
            "shares": shares, "close": close,
            "pending_order": pending["side"] if pending else None,
            "cooldown_until_index": cooldown_until,
        })
        if shares > 0:
            exposure_days += 1

        if pending is None and i < end_i:
            if shares == 0:
                risk_recovered = not risk[i]
                entry_allowed = not (risk_exit_policy == "recovery_required" and not risk_recovered)
                if low[i] and i >= cooldown_until and entry_allowed:
                    pending = {"side": "BUY", "reason": "base_low_after_filters"}
            else:
                # The SMA200 risk exit has priority over high-point processing.
                if risk[i] and not risk_exit_fired:
                    risk_exit_fired = True
                    high_wait_start = None
                    pending = {"side": "SELL", "reason": "sma200_risk_exit"}
                elif high_wait_start is None and high[i]:
                    immediate = (
                        float(row.distance_sma_20) < RULES["high_immediate_exit_if_distance_sma20_below"]
                        or float(row.rsi14) < RULES["high_immediate_exit_if_rsi_below"]
                    )
                    if immediate:
                        pending = {"side": "SELL", "reason": "high_confirm_immediate"}
                    else:
                        high_wait_start = i
                elif high_wait_start is not None:
                    waited = i - high_wait_start
                    if (
                        float(row.rsi14) < RULES["high_wait_exit_if_rsi_below"]
                        or waited >= RULES["high_wait_max_sessions"]
                    ):
                        pending = {"side": "SELL", "reason": "high_confirm_rsi_or_timeout"}

    c = pd.DataFrame(curve)
    equity = c.equity.astype(float)
    dd = equity / equity.cummax() - 1
    years = (pd.Timestamp(c.date.iloc[-1]) - pd.Timestamp(c.date.iloc[0])).days / 365.25
    final = float(equity.iloc[-1])
    trade_df = pd.DataFrame(trades)
    return {
        "period_start": c.date.iloc[0],
        "period_end": c.date.iloc[-1],
        "initial_capital": initial_capital,
        "final_equity_mark_to_market": final,
        "cagr_mark_to_market": (final / initial_capital) ** (1 / years) - 1 if years > 0 and final > 0 else None,
        "max_drawdown": float(dd.min()),
        "completed_trades": len(trades),
        "win_rate_completed_only": float((trade_df.net_pnl > 0).mean()) if len(trade_df) else None,
        "exposure_ratio": exposure_days / len(c),
        "risk_exit_count": risk_exit_count,
        "high_confirm_exit_count": high_exit_count,
        "total_commission": total_commission,
        "total_sell_tax": total_tax,
        "estimated_slippage_cost": total_slippage,
        "total_cash_dividends": total_dividend,
        "blocked_missing_price_limit": missing_limit,
        "blocked_locked_limit": locked_limit,
        "blocked_slippage_outside_limit": outside_limit,
        "trades": trades,
        "equity_curve": curve,
    }


def annual_metrics(equity_curve: list[dict], trades: list[dict], initial_capital: float) -> list[dict]:
    c = pd.DataFrame(equity_curve)
    c["date"] = pd.to_datetime(c.date)
    c["year"] = c.date.dt.year
    t = pd.DataFrame(trades)
    if not t.empty:
        t["exit_year"] = pd.to_datetime(t.exit_date).dt.year
    prior = initial_capital
    out = []
    last_year = int(c.year.max())
    for year, g in c.groupby("year", sort=True):
        end = float(g.equity.iloc[-1])
        ty = t[t.exit_year == year] if not t.empty else pd.DataFrame()
        out.append({
            "year": int(year),
            "start_equity": prior,
            "end_equity": end,
            "annual_return": end / prior - 1 if prior else None,
            "completed_trades": len(ty),
            "win_rate": float((ty.net_pnl > 0).mean()) if len(ty) else None,
            "partial_year": int(year) == last_year and g.date.max().month < 12,
        })
        prior = end
    return out


def run(features_path: str, limits_path: str, dividends_path: str, output_dir: str) -> dict:
    d, limits, dividends = inputs(features_path, limits_path, dividends_path)
    d = d.sort_values("date").reset_index(drop=True)
    low, high, risk = build_candidate_signals(d)
    years = sorted(d.date.dt.year.unique())
    start_i = int(np.flatnonzero((d.date.dt.year >= 2015).to_numpy())[0])
    end_i = len(d) - 1
    limit_lookup = limits
    summaries, annuals = {}, {}
    sensitivity = {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Keep the literal per-position interpretation as the primary reported
    # reproduction. Add two predeclared semantic interpretations without
    # selecting one based on return.
    for slip in SLIPPAGE_BPS:
        sim = simulate(d, low, high, risk, limit_lookup, dividends, slip, start_i, end_i,
                       risk_exit_policy="per_position")
        summaries[f"{slip}bps"] = {k:v for k,v in sim.items() if k not in {"trades","equity_curve"}}
        annuals[f"{slip}bps"] = annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL)
        pd.DataFrame(sim["trades"]).to_csv(out/f"tsmc_2330_4019_replication_trades_{slip}bps.csv",index=False)
        pd.DataFrame(sim["equity_curve"]).to_csv(out/f"tsmc_2330_4019_replication_equity_{slip}bps.csv",index=False)
        pd.DataFrame(annuals[f"{slip}bps"]).to_csv(out/f"tsmc_2330_4019_replication_annual_{slip}bps.csv",index=False)

    for policy in ("per_regime_latched", "recovery_required"):
        sensitivity[policy] = {}
        for slip in SLIPPAGE_BPS:
            sim = simulate(d, low, high, risk, limit_lookup, dividends, slip, start_i, end_i,
                           risk_exit_policy=policy)
            sensitivity[policy][f"{slip}bps"] = {
                **{k:v for k,v in sim.items() if k not in {"trades","equity_curve"}},
                "annual_metrics": annual_metrics(sim["equity_curve"],sim["trades"],INITIAL_CAPITAL),
            }
            pd.DataFrame(sim["trades"]).to_csv(
                out/f"tsmc_2330_4019_{policy}_trades_{slip}bps.csv",index=False
            )
    result = {
        "symbol":"2330",
        "classification":"EXPLORATORY_REPLICATION_OF_PREVIOUSLY_INSPECTED_CANDIDATE_NOT_PRISTINE_OOF",
        "data_period":[d.date.min().date().isoformat(),d.date.max().date().isoformat()],
        "validation_account_period":[d.date.iloc[start_i].date().isoformat(),d.date.iloc[end_i].date().isoformat()],
        "initial_capital":INITIAL_CAPITAL,
        "rule_constants":RULES,
        "signal_timing":"all conditions known at close t; fills next trading session open",
        "account_policy":"one continuous account; no annual resets; no annual forced liquidation; final position marked to market",
        "costs":{"commission_per_side":0.001425,"minimum_commission_ntd":20,"sell_tax":0.003,"tested_one_way_slippage_bps":list(SLIPPAGE_BPS)},
        "fill_policy":"adverse slippage rounded to TWSE common-stock legal tick; official daily limit data missing or locked orders fail closed",
        "risk_exit_policy_primary": "per_position",
        "metrics_by_cost":summaries,
        "annual_metrics_by_cost":annuals,
        "risk_exit_semantic_sensitivity": sensitivity,
        "integrity_notes":[
            "The rule constants were supplied in the handoff from a historically inspected candidate; the entire 2010-2026 period has been inspected in prior research.",
            "This is a replication audit, not an independent blind OOF result.",
            "The stated one-time SMA200 risk exit is implemented once per position and cooldown begins from its actual fill session.",
            "The high-signal wait is triggered by the baseline high condition; exit follows immediately if distance_sma_20 < 0 or RSI14 < 58, otherwise RSI14 < 48 or 14-session timeout.",
            "Only fully closed trades count toward completed-trade win rate; the final open position remains marked to market.",
        ],
    }
    (out/"tsmc_2330_4019_replication.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",required=True)
    p.add_argument("--limits",required=True)
    p.add_argument("--dividends",required=True)
    p.add_argument("--output",default="artifacts/tsmc_2330_4019_replication")
    a=p.parse_args()
    result=run(a.features,a.limits,a.dividends,a.output)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=float))


if __name__=="__main__":
    main()
