"""Research utilities for the Taiwan stock paper-trading app.

All functions use historical OHLCV only. Fundamental data is deliberately kept
as a latest-available snapshot and must not be mixed into historical signals.
"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple
from corporate_actions import validate_action_frame
import numpy as np
import pandas as pd
import yfinance as yf

STRATEGIES = [
    "趨勢結構延續",
    "量價突破",
    "動能趨勢突破",
    "均值回歸",
    "波動收縮突破",
]


def _signals(frame: pd.DataFrame) -> Dict[str, Tuple[pd.Series, pd.Series]]:
    z = frame.copy()
    c = z["close"].astype(float)
    s50 = c.rolling(50, min_periods=50).mean()
    s150 = c.rolling(150, min_periods=150).mean()
    s200 = c.rolling(200, min_periods=200).mean()
    h52 = z["high"].rolling(252, min_periods=120).max()
    l52 = z["low"].rolling(252, min_periods=120).min()
    r20, r60 = c.pct_change(20), c.pct_change(60)
    h20 = z["high"].shift(1).rolling(20, min_periods=20).max()
    v20 = z["volume"].shift(1).rolling(20, min_periods=20).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema21_up = ema21 > ema21.shift(5)
    vol_ok = z["volume"] > 1.4 * v20
    sma20 = c.rolling(20, min_periods=20).mean()
    std20 = c.rolling(20, min_periods=20).std(ddof=0)
    lower_band = sma20 - 2.0 * std20
    delta = c.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
    bandwidth = (4.0 * std20 / sma20.replace(0, np.nan))
    squeeze_threshold = bandwidth.shift(2).rolling(120, min_periods=60).quantile(0.20)
    recent_squeeze = (bandwidth.shift(1) <= squeeze_threshold).rolling(10, min_periods=1).max().fillna(0).astype(bool)
    return {
        STRATEGIES[0]: (
            (c > s50) & (s50 > s150) & (s150 > s200)
            & (s200 > s200.shift(20)) & (c >= l52 * 1.25)
            & (c >= h52 * .75) & (r20 > 0) & (r60 > 0),
            (c < s50) | (c < s150),
        ),
        STRATEGIES[1]: (
            (c > h20) & vol_ok & (c > s50) & (r20 > 0),
            (c < s50) | (c < ema21),
        ),
        STRATEGIES[2]: (
            (c > h20) & (c > ema50) & ema21_up & (r20 > 0),
            (c < ema21) | (c < ema50),
        ),
        # Mean reversion is deliberately restricted to the long-term uptrend;
        # the signal is observed at close and executed no earlier than next open.
        STRATEGIES[3]: (
            (c > s200) & (c < lower_band) & (rsi < 35),
            (c > sma20) | (rsi > 55),
        ),
        # A squeeze must have occurred within the previous ten completed bars.
        # Today's breakout is confirmed with prior-volume average and trend.
        STRATEGIES[4]: (
            (c > h20) & vol_ok & (c > s50) & recent_squeeze & (r20 > 0),
            (c < ema21) | (c < s50),
        ),
    }


def strategy_signals(frame: pd.DataFrame):
    """Public strategy signal helper; returns entry/exit Series per strategy."""
    return _signals(frame)


def daily_strategy_returns(
    frame: pd.DataFrame,
    strategy: str,
    fee: float = 0.001425,
    sell_tax: float = 0.003,
    slippage: float = 0.001,
) -> pd.Series:
    """Single-symbol daily net returns; signals execute on the next session.

    This is a single-position return series, not a portfolio capital simulator.
    For multiple symbols use shared_capital_backtest().
    """
    if frame.empty:
        return pd.Series(dtype=float)
    z = frame.reset_index(drop=True)
    entries, exits = _signals(z)[strategy]
    held = False
    position, turnover, sell = [], [], []
    for i in range(len(z)):
        was_held = held
        if held and bool(exits.iloc[i]):
            held = False
        elif not held and bool(entries.iloc[i]):
            held = True
        position.append(1.0 if held else 0.0)
        turnover.append(1.0 if was_held != held else 0.0)
        sell.append(1.0 if was_held and not held else 0.0)

    # Positions are shifted: signal at t is filled at t+1 open.
    pos = pd.Series(position).shift(1).fillna(0.0)
    prev_pos = pos.shift(1).fillna(0.0)
    open_to_open = z["open"].shift(-1) / z["open"] - 1.0
    open_to_open = open_to_open.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Costs charged on changes in exposure; costs are conservative estimates.
    turn = pos.sub(prev_pos).abs()
    exit_turn = ((prev_pos > 0) & (pos == 0)).astype(float)
    net = pos * open_to_open - turn * (fee + slippage) - exit_turn * sell_tax
    idx = pd.to_datetime(z["date"])
    return pd.Series(net.to_numpy(), index=idx, name=strategy)


def performance(returns: pd.Series, initial_capital: float) -> dict:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {
            "期末資產": float(initial_capital), "總報酬": 0.0,
            "年化報酬": 0.0, "最大回撤": 0.0, "Sharpe": 0.0, "交易日數": 0,
        }
    equity = float(initial_capital) * (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    years = max(len(r) / 252.0, 1.0 / 252.0)
    ending = float(equity.iloc[-1])
    cagr = (ending / float(initial_capital)) ** (1.0 / years) - 1.0 if ending > 0 else -1.0
    vol = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / vol * np.sqrt(252)) if vol > 0 else 0.0
    return {
        "期末資產": ending,
        "總報酬": ending / float(initial_capital) - 1.0,
        "年化報酬": float(cagr),
        "最大回撤": float(dd.min()),
        "Sharpe": sharpe,
        "交易日數": int(len(r)),
    }



def _update_exit_management(position: dict, row: pd.Series) -> None:
    """Apply completed-session high/close/ATR data to next-session protection."""
    entry_price = float(position["entry_price"])
    position["high_water"] = max(
        float(position.get("high_water", entry_price)),
        float(row["high"]),
        float(row["close"]),
    )
    position["atr"] = max(
        float(row.get("_atr", position["atr"])),
        entry_price * 0.001,
    )
    initial_risk = max(float(position["initial_risk"]), 0.01)
    r_multiple = (float(row["close"]) - entry_price) / initial_risk
    if position["management"] == "oneil":
        if float(row["close"]) >= entry_price * 1.20:
            position["profit_review_triggered"] = True
            # 20% is a review zone, not a hard take-profit.
            position["stop"] = max(
                position["stop"],
                position["high_water"] - 3.0 * position["atr"],
            )
    elif position["management"] == "minervini":
        if r_multiple >= 2.0:
            position["stop"] = max(position["stop"], entry_price)
        if r_multiple >= 3.0:
            position["stop"] = max(
                position["stop"],
                position["high_water"] - 2.5 * position["atr"],
            )
    else:
        position["stop"] = max(
            position["stop"],
            position["high_water"] - position["trail_multiple"] * position["atr"],
        )


def shared_capital_backtest(
    frames: Mapping[str, pd.DataFrame],
    strategy: str,
    initial_capital: float = 1_000_000,
    max_positions: int = 5,
    risk_fraction: float = 0.005,
    fee: float = 0.001425,
    sell_tax: float = 0.003,
    slippage: float = 0.001,
    warmup: int = 260,
    report_start_date=None,
    trading_start_date=None,
    exit_profile: str = "baseline_atr2",
    signal_overrides: Mapping[str, Tuple[pd.Series, pd.Series]] | None = None,
    max_participation_rate: float = 0.01,
    corporate_actions: Mapping[str, pd.DataFrame] | None = None,
) -> dict:
    """Multi-symbol portfolio simulation with one shared cash balance.

    At each close, eligible entries are ranked by 20-session momentum. Orders
    execute at the next session open. Available cash is split equally among
    available slots; existing positions are marked to market daily. No
    leverage, shorting, or fractional shares. Risk sizing is capped by an
    ATR-based stop distance, while total exposure is limited by available cash.
    """
    exit_profiles = {"baseline_atr2", "oneil_8pct", "minervini_structure", "atr_1_5", "atr_2", "atr_2_5", "atr_3"}
    if exit_profile not in exit_profiles:
        raise ValueError(f"未知出場設定：{exit_profile}；可用值：{sorted(exit_profiles)}")

    initial_capital = float(initial_capital)
    risk_fraction = float(risk_fraction)
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be a finite positive number")
    if int(max_positions) < 1:
        raise ValueError("max_positions must be at least 1")
    if not np.isfinite(risk_fraction) or not 0 <= risk_fraction <= 0.10:
        raise ValueError("risk_fraction must be between 0 and 0.10")
    if any(not np.isfinite(float(v)) or float(v) < 0 for v in (fee, sell_tax, slippage)):
        raise ValueError("fee, sell_tax and slippage must be finite non-negative rates")
    if not np.isfinite(max_participation_rate) or not 0 < max_participation_rate <= 0.10:
        raise ValueError("max_participation_rate 必須介於 0 與 10%")

    if not frames:
        return {"metrics": performance(pd.Series(dtype=float), initial_capital),
                "equity": pd.DataFrame(), "trades": pd.DataFrame(),
                "coverage": {"symbols_requested": 0, "symbols_used": 0}}

    prepared = {}
    for code, raw in frames.items():
        if raw is None or raw.empty or len(raw) <= warmup + 2:
            continue
        z = raw.copy().sort_values("date").reset_index(drop=True)
        if signal_overrides is not None and code in signal_overrides:
            entry, exit_ = signal_overrides[code]
            entry = pd.Series(entry).reset_index(drop=True)
            exit_ = pd.Series(exit_).reset_index(drop=True)
            if len(entry) != len(z) or len(exit_) != len(z):
                raise ValueError(f"{code} 的自訂訊號長度必須與歷史資料相同")
        else:
            sigs = _signals(z)
            if strategy not in sigs:
                continue
            entry, exit_ = sigs[strategy]
        z["_entry"] = entry.fillna(False).to_numpy()
        z["_exit"] = exit_.fillna(False).to_numpy()
        z["_momentum"] = z["close"].pct_change(20).fillna(-999).to_numpy()
        # ADV is known at the signal close (t-1) before the next-open fill (t).
        # Never size an opening fill with the current session's eventual volume.
        if "volume" in z:
            z["_adv20"] = (pd.to_numeric(z["close"], errors="coerce") *
                           pd.to_numeric(z["volume"], errors="coerce")).rolling(20, min_periods=20).mean().to_numpy()
        else:
            z["_adv20"] = np.nan
        # Lowest low of the prior 10 completed bars: transparent structural-stop proxy.
        # Structural invalidation uses only the 10 completed bars BEFORE the signal bar.
        z["_structure_low"] = z["low"].shift(1).rolling(10, min_periods=10).min().to_numpy()
        if "atr" in z:
            z["_atr"] = pd.to_numeric(z["atr"], errors="coerce").fillna(z["close"] * .04).to_numpy()
        else:
            tr = pd.concat([
                z["high"] - z["low"],
                (z["high"] - z["close"].shift()).abs(),
                (z["low"] - z["close"].shift()).abs(),
            ], axis=1).max(axis=1)
            z["_atr"] = tr.ewm(alpha=1/14, adjust=False).mean().fillna(z["close"] * .04).to_numpy()
        prepared[code] = z

    if not prepared:
        return {"metrics": performance(pd.Series(dtype=float), initial_capital),
                "equity": pd.DataFrame(), "trades": pd.DataFrame(),
                "coverage": {"symbols_requested": len(frames), "symbols_used": 0}}

    # Build date lookups once. Repeated Series-to-datetime conversions and
    # boolean scans inside every portfolio session were a major runtime hotspot.
    row_lookup = {}
    index_lookup = {}
    for code, z in prepared.items():
        date_values = pd.to_datetime(z["date"])
        row_lookup[code] = {
            pd.Timestamp(date_values.iloc[i]): z.iloc[i]
            for i in range(len(z))
        }
        index_lookup[code] = {
            pd.Timestamp(date_values.iloc[i]): i
            for i in range(len(z))
        }
    # Calendar dates are unioned. A symbol without a bar on a date is not traded.
    dates = sorted(set().union(*[set(x) for x in row_lookup.values()]))
    cash = float(initial_capital)
    positions = {}
    last_closes = {}
    trades = []
    equity_rows = []
    fee_rate = float(fee)
    tax_rate = float(sell_tax)
    slip_rate = float(slippage)
    action_lookup = {
        str(code): validate_action_frame(events, str(code))
        for code, events in (corporate_actions or {}).items()
    }
    processed_actions = set()
    dividend_receivables = []

    for d in dates:
        todays = {
            code: lookup[d]
            for code, lookup in row_lookup.items()
            if d in lookup
        }

        # Corporate actions apply before the ex-date open to positions held
        # through the prior close. Cash dividends become receivables on ex-date;
        # share-changing actions adjust quantity and per-share risk levels.
        for code, events in action_lookup.items():
            if code not in positions or code not in todays or events.empty:
                continue
            due = events.loc[pd.to_datetime(events["effective_date"]).dt.normalize() == pd.Timestamp(d).normalize()]
            for event in due.itertuples(index=False):
                event_id = str(event.event_id)
                if event_id in processed_actions:
                    continue
                p = positions[code]
                if event.action_type == "cash_dividend":
                    amount = float(p["shares"]) * float(event.cash_per_share)
                    if amount > 0:
                        p["dividends"] = float(p.get("dividends", 0.0)) + amount
                        dividend_receivables.append({
                            "symbol": code, "event_id": event_id,
                            "payment_date": pd.Timestamp(event.payment_date).normalize(),
                            "amount": amount,
                        })
                elif event.action_type in {
                    "stock_dividend", "stock_split", "capital_reduction", "par_value_change"
                }:
                    multiplier = float(event.share_multiplier)
                    exact_shares = float(p["shares"]) * multiplier
                    whole_shares = float(np.floor(exact_shares + 1e-9))
                    fractional_shares = max(0.0, exact_shares - whole_shares)
                    if fractional_shares > 1e-8:
                        reference = event.reference_price
                        try:
                            reference = float(reference)
                        except (TypeError, ValueError):
                            reference = float(todays[code]["open"])
                        if not np.isfinite(reference) or reference <= 0:
                            reference = float(todays[code]["open"])
                        cash_in_lieu = fractional_shares * reference
                        cash += cash_in_lieu
                        p["cash_in_lieu"] = float(p.get("cash_in_lieu", 0.0)) + cash_in_lieu
                    p["shares"] = whole_shares
                    for key in ("entry_price", "stop", "high_water", "atr", "initial_risk"):
                        p[key] = float(p[key]) / multiplier
                processed_actions.add(event_id)

        # Settle previously accrued cash dividends on their published payment date.
        still_receivable = []
        for item in dividend_receivables:
            if item["payment_date"] <= pd.Timestamp(d).normalize():
                cash += float(item["amount"])
            else:
                still_receivable.append(item)
        dividend_receivables = still_receivable

        # Mark current portfolio using today's close; execute exits at today's
        # open based on signals formed at the prior close.
        for code in list(positions):
            row = todays.get(code)
            if row is None:
                continue
            p = positions[code]
            open_px = float(row["open"])
            low_px = float(row["low"])
            prior_exit = p["exit_next"]
            # A signal formed at the prior close is executed at today's open.
            # Do not let today's later low retroactively override that fill.
            stop_hit = (not prior_exit) and low_px <= p["stop"]
            if stop_hit or prior_exit:
                raw_fill = min(open_px, p["stop"]) if stop_hit else open_px
                fill = raw_fill * (1.0 - slip_rate)
                proceeds = p["shares"] * fill * (1.0 - fee_rate - tax_rate)
                dividend_income = float(p.get("dividends", 0.0))
                cash_in_lieu_income = float(p.get("cash_in_lieu", 0.0))
                pnl = proceeds + dividend_income + cash_in_lieu_income - p["entry_cost"]
                trades.append({
                    "代號": code, "訊號日": p["signal_date"], "進場日": p["entry_date"],
                    "出場日": str(pd.Timestamp(d).date()), "股數": p["shares"],
                    "進場成本": p["entry_cost"], "出場收入": proceeds,
                    "股利收入": dividend_income,
                    "零股權利現金結算": cash_in_lieu_income,
                    "損益": pnl, "報酬率": pnl / p["entry_cost"] if p["entry_cost"] else 0.0,
                    "出場原因": "策略出場" if prior_exit else "停損／移動停損",
                    "曾觸發獲利檢視": bool(p.get("profit_review_triggered", False)),
                    "最高風險倍數": round((p.get("high_water", p["entry_price"]) - p["entry_price"]) / max(p.get("initial_risk", 0.01), 0.01), 2),
                    "估計ADV20參與率": (p["shares"] * p["entry_price"] / p["adv20"]) if p.get("adv20", 0) else np.nan,
                })
                cash += proceeds
                del positions[code]
            else:
                # Completed today's bar updates protection for the next session.
                _update_exit_management(p, row)

        # Entry decisions from previous close; the first warmup period is not traded.
        candidates = []
        for code, row in todays.items():
            z = prepared[code]
            i = index_lookup[code][d]
            if i < warmup or code in positions:
                continue
            if trading_start_date is not None and pd.Timestamp(d) < pd.Timestamp(trading_start_date):
                continue
            # Today's order may only use information available at the prior close.
            # Fill at today's open; never use today's close/high/low to create the entry.
            if i < 1 or i >= len(z) - 1:
                # Do not open a new position on a symbol's final bar: there is
                # no subsequent session in which the position can be managed.
                continue
            signal_row = z.iloc[i - 1]
            if not bool(signal_row["_entry"]):
                continue
            candidates.append((float(signal_row["_momentum"]), code, row, signal_row))

        free_slots = max(0, int(max_positions) - len(positions))
        if free_slots and candidates:
            candidates.sort(reverse=True)
            chosen = candidates[:free_slots]
            for _, code, row, signal_row in chosen:
                slots_left = max(1, int(max_positions) - len(positions))
                allocation = cash / slots_left
                open_px = float(row["open"]) * (1.0 + slip_rate)
                atr = max(float(signal_row["_atr"]), open_px * .005)
                if exit_profile in ("baseline_atr2", "atr_2"):
                    stop = open_px - 2.0 * atr
                elif exit_profile == "oneil_8pct":
                    # Commonly cited 7–8% discipline represented at 8% for comparison.
                    stop = open_px * 0.92
                elif exit_profile == "minervini_structure":
                    structure_low = float(signal_row["_structure_low"])
                    if not np.isfinite(structure_low) or structure_low <= 0 or structure_low >= open_px:
                        continue  # no valid structural stop => no trade
                    stop = structure_low
                else:
                    multiple = {"atr_1_5": 1.5, "atr_2_5": 2.5, "atr_3": 3.0}[exit_profile]
                    stop = open_px - multiple * atr
                if not np.isfinite(stop) or stop <= 0 or stop >= open_px:
                    continue
                prior_equity = cash + sum(
                    p["shares"] * float(last_closes.get(k, p["entry_price"]))
                    for k, p in positions.items()
                )
                risk_budget = max(float(prior_equity), 0.0) * risk_fraction
                # Apply the same account-level risk budget to each profile, but
                # size shares from that profile's ACTUAL initial stop distance.
                # This prevents narrow structural stops or wide 8% stops from
                # silently changing the intended cash-at-risk.
                shares_by_risk = int(risk_budget / max(open_px - stop, 0.01))
                shares_by_cash = int(cash / (open_px * (1.0 + fee_rate)))
                shares_by_alloc = int(allocation / (open_px * (1.0 + fee_rate)))
                adv20 = float(signal_row["_adv20"])
                shares_by_liquidity = (int(max(0.0, adv20) * max_participation_rate / open_px)
                                       if np.isfinite(adv20) and adv20 > 0 else 0)
                shares = max(0, min(shares_by_risk, shares_by_cash, shares_by_alloc, shares_by_liquidity))
                if shares <= 0:
                    continue
                entry_cost = shares * open_px * (1.0 + fee_rate)
                cash -= entry_cost
                initial_risk = open_px - stop
                management = ("oneil" if exit_profile == "oneil_8pct"
                              else "minervini" if exit_profile == "minervini_structure"
                              else "atr")
                trail_multiple = {"atr_1_5": 1.5, "atr_2": 2.0,
                                  "baseline_atr2": 2.0, "atr_2_5": 2.5,
                                  "atr_3": 3.0}.get(exit_profile, 2.0)
                positions[code] = {
                    "shares": shares, "entry_cost": entry_cost,
                    "signal_date": str(pd.Timestamp(signal_row["date"]).date()),
                    "entry_date": str(pd.Timestamp(d).date()), "stop": stop,
                    "entry_price": open_px, "initial_risk": initial_risk,
                    "high_water": open_px, "atr": atr,
                    "management": management, "trail_multiple": trail_multiple,
                    "profit_review_triggered": False, "exit_next": False,
                    "adv20": adv20, "max_participation_rate": max_participation_rate,
                    "dividends": 0.0, "cash_in_lieu": 0.0,
                }
                # If the stop is touched later on the entry day, simulate the
                # stop fill rather than waiting until the following session.
                if float(row["low"]) <= stop:
                    fill = min(float(row["open"]), stop) * (1.0 - slip_rate)
                    proceeds = shares * fill * (1.0 - fee_rate - tax_rate)
                    dividend_income = float(positions[code].get("dividends", 0.0))
                    cash_in_lieu_income = float(positions[code].get("cash_in_lieu", 0.0))
                    pnl = proceeds + dividend_income + cash_in_lieu_income - entry_cost
                    trades.append({
                        "代號": code, "訊號日": positions[code]["signal_date"],
                        "進場日": positions[code]["entry_date"],
                        "出場日": str(pd.Timestamp(d).date()), "股數": shares,
                        "進場成本": entry_cost, "出場收入": proceeds,
                        "股利收入": dividend_income,
                        "零股權利現金結算": cash_in_lieu_income,
                        "損益": pnl, "報酬率": pnl / entry_cost if entry_cost else 0.0,
                        "出場原因": "進場日觸及初始停損",
                        "曾觸發獲利檢視": False, "最高風險倍數": 0.0,
                        "估計ADV20參與率": (shares * open_px / adv20) if adv20 > 0 else np.nan,
                    })
                    cash += proceeds
                    del positions[code]
                else:
                    # Include the entry session's completed high/close/ATR.
                    # The resulting stop is only checked from the next session.
                    _update_exit_management(positions[code], row)

        # The current close creates an exit instruction for the next session.
        for code, p in positions.items():
            row = todays.get(code)
            if row is not None:
                p["exit_next"] = bool(todays[code]["_exit"])

        # A symbol's history can end before the portfolio calendar. Liquidate
        # on that symbol's own final available close instead of postponing the
        # cash credit until the final date of another symbol.
        for code in list(positions):
            z = prepared[code]
            last_date = pd.Timestamp(z["date"].iloc[-1])
            if last_date != pd.Timestamp(d) or code not in todays:
                continue
            p = positions[code]
            fill = float(todays[code]["close"]) * (1.0 - slip_rate)
            proceeds = p["shares"] * fill * (1.0 - fee_rate - tax_rate)
            dividend_income = float(p.get("dividends", 0.0))
            cash_in_lieu_income = float(p.get("cash_in_lieu", 0.0))
            pnl = proceeds + dividend_income + cash_in_lieu_income - p["entry_cost"]
            trades.append({
                "代號": code, "訊號日": p["signal_date"], "進場日": p["entry_date"],
                "出場日": str(pd.Timestamp(d).date()), "股數": p["shares"],
                "進場成本": p["entry_cost"], "出場收入": proceeds,
                "股利收入": dividend_income,
                "零股權利現金結算": cash_in_lieu_income,
                "損益": pnl, "報酬率": pnl / p["entry_cost"] if p["entry_cost"] else 0.0,
                "出場原因": "該標的資料期末平倉",
                "曾觸發獲利檢視": bool(p.get("profit_review_triggered", False)),
                "最高風險倍數": round((p.get("high_water", p["entry_price"]) - p["entry_price"]) / max(p.get("initial_risk", 0.01), 0.01), 2),
                "估計ADV20參與率": (p["shares"] * p["entry_price"] / p["adv20"]) if p.get("adv20", 0) else np.nan,
            })
            cash += proceeds
            del positions[code]

        market_value = 0.0
        for code, p in positions.items():
            row = todays.get(code)
            if row is None:
                z = prepared[code]
                if code not in last_closes:
                    continue
                mark = float(last_closes[code])
            else:
                mark = float(row["close"])
            market_value += p["shares"] * mark
        for code, row in todays.items():
            last_closes[code] = float(row["close"])

        receivable_value = sum(float(item["amount"]) for item in dividend_receivables)
        if report_start_date is None or pd.Timestamp(d) >= pd.Timestamp(report_start_date):
            equity_rows.append({"日期": pd.Timestamp(d), "現金": cash,
                                "股利應收": receivable_value,
                                "持倉市值": market_value, "持倉數": len(positions),
                                "資產": cash + receivable_value + market_value})

    eq = pd.DataFrame(equity_rows)
    if not eq.empty:
        eq.loc[eq.index[-1], "現金"] = cash
        eq.loc[eq.index[-1], "持倉市值"] = 0.0
        eq.loc[eq.index[-1], "持倉數"] = 0
        eq.loc[eq.index[-1], "股利應收"] = sum(float(item["amount"]) for item in dividend_receivables)
        eq.loc[eq.index[-1], "資產"] = cash + sum(float(item["amount"]) for item in dividend_receivables)
    if not eq.empty:
        daily_returns = eq.set_index("日期")["資產"].pct_change()
        # Include first-day execution costs and mark-to-market P&L relative to
        # the initial cash balance instead of forcing the first return to zero.
        daily_returns.iloc[0] = float(eq["資產"].iloc[0]) / float(initial_capital) - 1.0
        daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        metrics = performance(daily_returns, initial_capital)
        ending_equity = float(cash) + sum(float(item["amount"]) for item in dividend_receivables)
        metrics["期末資產"] = ending_equity
        metrics["總報酬"] = float(ending_equity / initial_capital - 1.0)
        years = max((pd.Timestamp(eq["日期"].iloc[-1]) - pd.Timestamp(eq["日期"].iloc[0])).days / 365.25, 1/252)
        metrics["年化報酬"] = float((ending_equity / initial_capital) ** (1 / years) - 1) if ending_equity > 0 else -1.0
        peak = pd.Series(
            np.maximum.accumulate(np.r_[float(initial_capital), eq["資產"].to_numpy()])[1:],
            index=eq.index,
        )
        metrics["最大回撤"] = float((eq["資產"] / peak - 1).min())
    else:
        metrics = performance(pd.Series(dtype=float), initial_capital)

    return {
        "metrics": metrics,
        "equity": eq,
        "trades": pd.DataFrame(trades),
        "coverage": {"symbols_requested": len(frames), "symbols_used": len(prepared)},
        "corporate_actions_applied": len(processed_actions),
        "unpaid_dividend_receivables": sum(float(item["amount"]) for item in dividend_receivables),
        "exit_profile": exit_profile,
        "exit_profile_note": ("Minervini 使用前 10 根已完成日 K 最低價作為結構失效代理；不是完整 SEPA 型態辨識。" if exit_profile == "minervini_structure" else "出場設定僅作候選比較，尚未證明任何一組參數具有樣本外優勢。"),
    }


def fundamental_snapshot(codes):
    """Latest snapshot only; never use this function for historical backtests."""
    rows = []
    for code in codes:
        info = {}
        for suffix in [".TW", ".TWO"]:
            try:
                info = yf.Ticker(str(code) + suffix).get_info()
                if info:
                    break
            except Exception:
                continue
        rows.append({
            "代號": code,
            "ROE": info.get("returnOnEquity"),
            "ROA": info.get("returnOnAssets"),
            "營收成長率": info.get("revenueGrowth"),
            "獲利成長率": info.get("earningsGrowth"),
            "毛利率": info.get("grossMargins"),
            "營業利益率": info.get("operatingMargins"),
            "自由現金流": info.get("freeCashflow"),
            "負債／權益": info.get("debtToEquity"),
            "流動比率": info.get("currentRatio"),
            "本益比": info.get("trailingPE"),
            "股價淨值比": info.get("priceToBook"),
            "資料狀態": "有回傳（最新快照，非歷史資料）" if info else "來源未提供",
        })
    return pd.DataFrame(rows)
