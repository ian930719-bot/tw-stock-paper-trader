"""Annual prior-year top-20 turnover universe and shared-capital strategy research.

Signals use completed adjusted bars; fills use raw bars no earlier than next open.
This is exploratory until historical membership, delisted coverage, corporate
actions, and exchange execution rules are independently reconciled.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from corporate_action_adjustments import apply_back_adjustments

BUY_FEE = 0.001425
SELL_FEE_TAX = 0.001425 + 0.003
SLIPPAGE = 0.001
INITIAL_CAPITAL = 100_000.0
RISK_FRACTION = 0.005
MAX_POSITIONS = 5
MAX_WEIGHT = 0.20
MAX_ADV_PARTICIPATION = 0.01
MAX_HOLD_BARS = 60
STRATEGIES = ("trend_breakout", "momentum_rotation", "trend_pullback",
              "mean_reversion", "squeeze_breakout")
START_YEAR = 2016
END_YEAR = 2026


def _valid_bars(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume", "turnover"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{symbol}: missing columns {sorted(missing)}")
    z = frame.copy()
    z["date"] = pd.to_datetime(z["date"], errors="coerce").dt.normalize()
    for col in ("open", "high", "low", "close", "volume", "turnover"):
        z[col] = pd.to_numeric(z[col], errors="coerce")
    z = z.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    valid = (
        (z[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (z["volume"] >= 0)
        & (z["high"] >= z[["open", "close", "low"]].max(axis=1))
        & (z["low"] <= z[["open", "close", "high"]].min(axis=1))
    )
    return z.loc[valid].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def load_history(root: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    manifest_path = root / "download_manifest.json"
    candidates_path = root / "candidate_symbols.csv"
    bars_dir = root / "bars"
    if not manifest_path.exists() or not candidates_path.exists() or not bars_dir.is_dir():
        raise FileNotFoundError("Need download_manifest.json, candidate_symbols.csv, and bars/")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = pd.read_csv(candidates_path, dtype={"symbol": str, "stock_id": str})
    symbol_col = "symbol" if "symbol" in candidates else "stock_id"
    if symbol_col not in candidates:
        raise ValueError("candidate_symbols.csv missing symbol column")
    symbols = candidates[symbol_col].dropna().astype(str).str.strip().unique()
    events_path = root / "corporate_actions" / "events.csv"
    events = pd.read_csv(events_path, dtype={"symbol": str}) if events_path.exists() else pd.DataFrame()
    if not events.empty:
        if "conflict" in events:
            events = events.loc[~events["conflict"].fillna(False).astype(bool)].copy()
    frames = {}
    for symbol in symbols:
        path = bars_dir / f"{symbol}.csv"
        if not path.exists() or path.stat().st_size < 20:
            continue
        raw = _valid_bars(pd.read_csv(path), str(symbol))
        if raw.empty:
            continue
        action = events.loc[events["symbol"].astype(str) == str(symbol)] if not events.empty else pd.DataFrame()
        adjusted = apply_back_adjustments(raw, action)
        frames[str(symbol)] = adjusted
    if not frames:
        raise ValueError("No valid OHLCV symbols loaded")
    return frames, manifest


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    z = frame.copy().sort_values("date").reset_index(drop=True)
    c, h, l, v = (z["adj_close"].astype(float), z["adj_high"].astype(float),
                  z["adj_low"].astype(float), z["volume"].astype(float))
    z["sma20"] = c.rolling(20, min_periods=20).mean()
    z["sma50"] = c.rolling(50, min_periods=50).mean()
    z["sma100"] = c.rolling(100, min_periods=100).mean()
    z["sma200"] = c.rolling(200, min_periods=200).mean()
    z["ema21"] = c.ewm(span=21, adjust=False, min_periods=21).mean()
    z["ema50"] = c.ewm(span=50, adjust=False, min_periods=50).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    z["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    z["atr_pct"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / c
    z["mom63"] = c.pct_change(63)
    z["mom126"] = c.pct_change(126)
    z["mom252"] = c.pct_change(252)
    z["prior_high55"] = h.shift(1).rolling(55, min_periods=55).max()
    z["prior_high20"] = h.shift(1).rolling(20, min_periods=20).max()
    z["prior_low20"] = l.shift(1).rolling(20, min_periods=20).min()
    z["volume_avg20"] = v.shift(1).rolling(20, min_periods=20).mean()
    # Turnover is the raw market value traded; lagged average prevents look-ahead.
    z["adv20"] = z["turnover"].shift(1).rolling(20, min_periods=20).mean()
    mid = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    z["bb_low"] = mid - 2 * sd
    z["bb_width"] = (4 * sd / mid.replace(0, np.nan))
    width_cut = z["bb_width"].shift(1).rolling(120, min_periods=60).quantile(0.20)
    z["recent_squeeze"] = (z["bb_width"].shift(1) <= width_cut).rolling(10, min_periods=1).max().fillna(0).astype(bool)
    z["rsi_reclaim"] = (z["rsi"] > 40) & (z["rsi"].shift(1) <= 40)
    z["pullback_recent"] = (z["rsi"].shift(1).rolling(5, min_periods=1).min() < 40)
    z["signal_atr_raw"] = z["atr_pct"] * z["close"]
    action_mask = z.get("corporate_action_window", pd.Series(False, index=z.index)).fillna(False).astype(bool)
    ready = z[["sma20", "sma50", "sma100", "sma200", "atr_pct", "adv20"]].notna().all(axis=1) & ~action_mask
    trend = (c > z["sma200"]) & (z["sma50"] > z["sma200"]) & (z["mom126"] > 0)
    z["entry_trend_breakout"] = (ready & trend & (c > z["prior_high55"]) &
                                  (v >= 1.5 * z["volume_avg20"])).fillna(False)
    z["exit_trend_breakout"] = ((c < z["ema21"]) | (c < z["sma50"])).fillna(False)
    z["entry_trend_pullback"] = (ready & trend & z["pullback_recent"] &
                                  z["rsi_reclaim"] & (c > z["ema21"])).fillna(False)
    z["exit_trend_pullback"] = ((c < z["ema50"]) | (z["mom63"] < 0)).fillna(False)
    z["entry_mean_reversion"] = (ready & (c > z["sma200"]) & (c < z["bb_low"]) &
                                  (z["rsi"] < 30)).fillna(False)
    z["exit_mean_reversion"] = ((c > z["sma20"]) | (c < z["sma200"])).fillna(False)
    z["entry_squeeze_breakout"] = (ready & trend & z["recent_squeeze"] &
                                    (c > z["prior_high20"]) &
                                    (v >= 1.5 * z["volume_avg20"])).fillna(False)
    z["exit_squeeze_breakout"] = ((c < z["ema21"]) | (c < z["sma50"])).fillna(False)
    return z


def build_annual_universe(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        prior = year - 1
        for symbol, frame in frames.items():
            sample = frame.loc[pd.to_datetime(frame["date"]).dt.year == prior]
            turnover = pd.to_numeric(sample["turnover"], errors="coerce").dropna()
            if len(turnover) < 100:
                continue
            rows.append({"year": year, "ranking_year": prior, "symbol": symbol,
                         "prior_year_turnover": float(turnover.sum()),
                         "turnover_days": int(len(turnover))})
    if not rows:
        raise ValueError("No annual turnover observations; cannot form a point-in-time top-20 list")
    raw = pd.DataFrame(rows)
    raw["rank"] = raw.groupby("year")["prior_year_turnover"].rank(method="first", ascending=False).astype(int)
    selected = raw.loc[raw["rank"] <= 20].sort_values(["year", "rank"]).copy()
    selected["universe_status"] = np.where(selected.groupby("year")["symbol"].transform("count") == 20,
                                           "20_symbols", "incomplete_under_20")
    return selected


def add_momentum_ranks(frames: dict[str, pd.DataFrame], universe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    memberships = {int(y): set(g["symbol"].astype(str)) for y, g in universe.groupby("year")}
    rows = []
    for symbol, frame in frames.items():
        x = frame[["date", "mom63", "mom126", "mom252"]].copy()
        x["symbol"] = symbol
        rows.append(x)
    panel = pd.concat(rows, ignore_index=True)
    panel["year"] = pd.to_datetime(panel["date"]).dt.year
    panel["in_universe"] = [str(s) in memberships.get(int(y), set()) for s, y in zip(panel.symbol, panel.year)]
    panel = panel.loc[panel["in_universe"]].copy()
    for col in ("mom63", "mom126", "mom252"):
        panel[f"rank_{col}"] = panel.groupby("date")[col].rank(pct=True)
    panel["momentum_score"] = (0.2 * panel["rank_mom63"] +
                               0.35 * panel["rank_mom126"] +
                               0.45 * panel["rank_mom252"])
    panel["momentum_rank"] = panel.groupby("date")["momentum_score"].rank(pct=True)
    ranks = panel.set_index(["symbol", "date"])["momentum_rank"]
    out = {}
    for symbol, frame in frames.items():
        z = frame.copy()
        keys = pd.MultiIndex.from_arrays([[symbol] * len(z), pd.to_datetime(z["date"])])
        z["momentum_rank"] = ranks.reindex(keys).to_numpy()
        trend = (z["adj_close"] > z["sma200"]) & (z["sma50"] > z["sma200"]) & (z["mom126"] > 0)
        ready = z[["sma100", "mom63", "mom126", "mom252"]].notna().all(axis=1)
        z["entry_momentum_rotation"] = (ready & trend & (z["momentum_rank"] >= 0.80)).fillna(False)
        z["exit_momentum_rotation"] = ((z["momentum_rank"] < 0.40) |
                                        (z["adj_close"] < z["sma100"]) |
                                        (z["mom63"] < 0)).fillna(False)
        out[symbol] = z
    return out


def _close_trade(position: dict, day: pd.Timestamp, fill: float, reason: str,
                 cost_multiplier: float = 1.0) -> dict:
    sell_cost = SELL_FEE_TAX * cost_multiplier
    buy_cost = BUY_FEE * cost_multiplier
    proceeds = position["shares"] * fill * (1 - sell_cost)
    return {**position, "exit_date": day, "exit_price": fill,
            "exit_reason": reason, "pnl": proceeds - position["entry_cost"],
            "net_return": (fill / position["entry_price"]) * (1 - buy_cost) * (1 - sell_cost) - 1}


def simulate(frames: dict[str, pd.DataFrame], universe: pd.DataFrame, strategy: str,
             initial_capital: float = INITIAL_CAPITAL,
             cost_multiplier: float = 1.0) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    entry_col, exit_col = f"entry_{strategy}", f"exit_{strategy}"
    if strategy == "momentum_rotation":
        frames = add_momentum_ranks(frames, universe)
    memberships = {int(y): set(g["symbol"].astype(str)) for y, g in universe.groupby("year")}
    row_lookup = {s: {pd.Timestamp(r.date): r for r in f.itertuples(index=False)}
                  for s, f in frames.items()}
    calendar = sorted(set().union(*(set(x) for x in row_lookup.values())))
    calendar = [d for d in calendar if pd.Timestamp(f"{START_YEAR}-01-01") <= d <= pd.Timestamp(f"{END_YEAR}-12-31")]
    schedules: dict[pd.Timestamp, list] = {}
    exit_flags = {}
    last_dates = {s: max(rows) for s, rows in row_lookup.items()}
    for symbol, frame in frames.items():
        for i, row in enumerate(frame.itertuples(index=False)):
            day = pd.Timestamp(row.date)
            year = day.year
            if symbol not in memberships.get(year, set()):
                continue
            exit_flags[(symbol, day)] = bool(getattr(row, exit_col))
            if bool(getattr(row, entry_col)) and i + 1 < len(frame):
                next_day = pd.Timestamp(frame.iloc[i + 1]["date"])
                if next_day.year in memberships and symbol in memberships[next_day.year]:
                    schedules.setdefault(next_day, []).append({
                        "symbol": symbol, "signal_date": day,
                        "atr": float(row.signal_atr_raw) if pd.notna(row.signal_atr_raw) else 0.0,
                        "adv20": float(row.adv20) if pd.notna(row.adv20) else 0.0,
                        "score": float(row.momentum_rank) if strategy == "momentum_rotation" and pd.notna(row.momentum_rank) else float(row.mom126) if pd.notna(row.mom126) else 0.0,
                    })
    cash, positions, trades, curve = float(initial_capital), {}, [], []
    last_close = {}
    for day in calendar:
        today = {s: rows[day] for s, rows in row_lookup.items() if day in rows}
        closed_today = set()
        # Pending signal/holding exits and overnight stop gaps are filled at today's open.
        for symbol in list(positions):
            if symbol not in today:
                continue
            p, row = positions[symbol], today[symbol]
            op = float(row.open)
            if p["pending_exit"]:
                fill, reason = op * (1 - SLIPPAGE * cost_multiplier), p["pending_reason"]
            elif op <= p["stop"]:
                fill, reason = op * (1 - SLIPPAGE * cost_multiplier), "gap_stop"
            else:
                continue
            trades.append(_close_trade(p, day, fill, reason, cost_multiplier))
            cash += p["shares"] * fill * (1 - SELL_FEE_TAX * cost_multiplier)
            del positions[symbol]
            closed_today.add(symbol)
        # Candidate priority is cross-sectional strength, then momentum.
        for candidate in sorted(schedules.get(day, []), key=lambda x: x["score"], reverse=True):
            symbol = candidate["symbol"]
            if symbol in positions or symbol in closed_today or symbol not in today:
                continue
            if symbol not in memberships.get(day.year, set()) or len(positions) >= MAX_POSITIONS:
                continue
            row = today[symbol]
            entry = float(row.open) * (1 + SLIPPAGE * cost_multiplier)
            atr = max(candidate["atr"], entry * 0.005)
            stop = entry - 3.0 * atr
            if stop <= 0 or candidate["adv20"] <= 0:
                continue
            equity_now = cash + sum(p["shares"] * last_close.get(s, p["entry_price"]) for s, p in positions.items())
            by_risk = math.floor(equity_now * RISK_FRACTION / max(entry - stop, 0.01))
            by_weight = math.floor(equity_now * MAX_WEIGHT / (entry * (1 + BUY_FEE * cost_multiplier)))
            by_cash = math.floor(cash / (entry * (1 + BUY_FEE * cost_multiplier)))
            by_adv = math.floor(candidate["adv20"] * MAX_ADV_PARTICIPATION / entry)
            shares = max(0, min(by_risk, by_weight, by_cash, by_adv))
            if shares <= 0:
                continue
            cost = shares * entry * (1 + BUY_FEE * cost_multiplier)
            cash -= cost
            positions[symbol] = {
                **candidate, "entry_date": day, "entry_price": entry,
                "shares": shares, "entry_cost": cost, "stop": stop,
                "high_water": entry, "bars_held": 0,
                "pending_exit": False, "pending_reason": "",
            }
        # Intraday stop handling; conservative fill at the worse of open or stop.
        for symbol in list(positions):
            if symbol not in today:
                continue
            p, row = positions[symbol], today[symbol]
            op, low = float(row.open), float(row.low)
            if low <= p["stop"]:
                fill = min(op, p["stop"]) * (1 - SLIPPAGE * cost_multiplier)
                trades.append(_close_trade(p, day, fill, "atr_stop", cost_multiplier))
                cash += p["shares"] * fill * (1 - SELL_FEE_TAX * cost_multiplier)
                del positions[symbol]
                closed_today.add(symbol)
        for symbol in list(positions):
            if symbol not in today:
                continue
            p, row = positions[symbol], today[symbol]
            p["bars_held"] += 1
            p["high_water"] = max(p["high_water"], float(row.high), float(row.close))
            atr = max(float(row.signal_atr_raw) if pd.notna(row.signal_atr_raw) else p["atr"], float(row.close) * 0.005)
            p["stop"] = max(p["stop"], p["high_water"] - 3.5 * atr)
            if exit_flags.get((symbol, day), False):
                p["pending_exit"], p["pending_reason"] = True, "strategy_exit_next_open"
            elif p["bars_held"] >= MAX_HOLD_BARS:
                p["pending_exit"], p["pending_reason"] = True, "max_hold_next_open"
            last_close[symbol] = float(row.close)
            # If a security stops publishing bars (e.g. delisting/data end), do not
            # carry a stale mark to the end of the portfolio sample.
            if last_dates.get(symbol) == day:
                fill = float(row.close) * (1 - SLIPPAGE * cost_multiplier)
                trades.append(_close_trade(p, day, fill, "symbol_data_end_close", cost_multiplier))
                cash += p["shares"] * fill * (1 - SELL_FEE_TAX * cost_multiplier)
                del positions[symbol]
        for symbol, row in today.items():
            last_close[symbol] = float(row.close)
        market_value = sum(p["shares"] * last_close.get(s, p["entry_price"]) for s, p in positions.items())
        equity = cash + market_value
        curve.append({"date": day, "equity": equity, "cash": cash,
                      "market_value": market_value, "exposure": market_value / equity if equity > 0 else 0,
                      "positions": len(positions)})
    # Liquidate all remaining positions at the last available close in the sample.
    if calendar:
        final_day = calendar[-1]
        for symbol, p in list(positions.items()):
            row = row_lookup[symbol].get(final_day)
            if row is None:
                available = [d for d in row_lookup[symbol] if d <= final_day]
                if not available:
                    continue
                row = row_lookup[symbol][available[-1]]
                exit_day = available[-1]
            else:
                exit_day = final_day
            fill = float(row.close) * (1 - SLIPPAGE * cost_multiplier)
            trades.append(_close_trade(p, exit_day, fill, "period_end_close", cost_multiplier))
            cash += p["shares"] * fill * (1 - SELL_FEE_TAX * cost_multiplier)
        positions.clear()
        if curve:
            curve[-1].update({"equity": cash, "cash": cash, "market_value": 0.0, "exposure": 0.0, "positions": 0})
    eq = pd.DataFrame(curve)
    tr = pd.DataFrame(trades)
    metrics = summarize(eq, tr, initial_capital)
    metrics["strategy"] = strategy
    metrics["cost_multiplier"] = cost_multiplier
    return metrics, tr, eq


def summarize(equity: pd.DataFrame, trades: pd.DataFrame, initial: float) -> dict:
    if equity.empty:
        return {"trades": 0, "ending_equity": initial, "CAGR": 0.0, "max_drawdown": 0.0}
    values = equity["equity"].astype(float)
    days = max((pd.Timestamp(equity.date.iloc[-1]) - pd.Timestamp(equity.date.iloc[0])).days, 1)
    years = days / 365.25
    ending = float(values.iloc[-1])
    returns = values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    dd = values / np.maximum.accumulate(np.r_[initial, values.to_numpy()])[1:] - 1
    pnl = pd.to_numeric(trades.get("pnl", pd.Series(dtype=float)), errors="coerce")
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    return {
        "start_date": str(pd.Timestamp(equity.date.iloc[0]).date()),
        "end_date": str(pd.Timestamp(equity.date.iloc[-1]).date()),
        "ending_equity": ending, "total_return": ending / initial - 1,
        "CAGR": (ending / initial) ** (1 / years) - 1 if ending > 0 else -1.0,
        "max_drawdown": float(dd.min()), "Sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0,
        "trades": int(len(trades)), "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else float("inf") if len(wins) else 0.0,
        "avg_exposure": float(equity.exposure.mean()),
        "avg_positions": float(equity.positions.mean()),
    }


def run(input_dir: str, output_dir: str) -> dict:
    root, out = Path(input_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_frames, source_manifest = load_history(root)
    universe = build_annual_universe(raw_frames)
    selected_symbols = set(universe["symbol"].astype(str))
    # Compute indicators only for symbols that enter at least one annual top-20
    # universe; all pre-year bars are retained for indicator warm-up.
    frames = {s: build_features(raw_frames[s]) for s in selected_symbols if s in raw_frames}
    metrics, trades, curves = [], [], []
    for cost_multiplier in (1.0, 1.5, 2.0):
        for strategy in STRATEGIES:
            m, t, e = simulate(frames, universe, strategy, cost_multiplier=cost_multiplier)
            label = f"{strategy}_cost_{cost_multiplier:g}x"
            m["strategy_label"] = label
            metrics.append(m)
            if not t.empty:
                trades.append(t.assign(strategy=label, cost_multiplier=cost_multiplier))
            if not e.empty:
                curves.append(e.assign(strategy=label, cost_multiplier=cost_multiplier))
    metrics_frame = pd.DataFrame(metrics)
    universe.to_csv(out / "annual_top20_turnover.csv", index=False)
    metrics_frame.to_csv(out / "strategy_metrics.csv", index=False)
    pd.concat(trades, ignore_index=True).to_csv(out / "trade_ledger.csv", index=False) if trades else pd.DataFrame().to_csv(out / "trade_ledger.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(out / "equity_curves.csv", index=False) if curves else pd.DataFrame().to_csv(out / "equity_curves.csv", index=False)
    year_rows = []
    for curve in curves:
        if curve.empty:
            continue
        strategy = str(curve.strategy.iloc[0])
        x = curve.copy()
        x["year"] = pd.to_datetime(x["date"]).dt.year
        for year, group in x.groupby("year"):
            prior = x.loc[pd.to_datetime(x.date).dt.year < year, "equity"]
            start_equity = float(prior.iloc[-1]) if not prior.empty else INITIAL_CAPITAL
            year_rows.append({"strategy": strategy, "year": int(year),
                              "is_partial_year": int(year) == END_YEAR,
                              "year_end_date": str(pd.Timestamp(group.date.iloc[-1]).date()),
                              "year_end_equity": float(group.equity.iloc[-1]),
                              "annual_return": float(group.equity.iloc[-1] / start_equity - 1)})
    annual = pd.DataFrame(year_rows)
    annual.to_csv(out / "annual_returns.csv", index=False)
    report = {
        "research_status": "RESEARCH_ONLY",
        "target_net_cagr": 1.0,
        "data_source": source_manifest.get("source", "unknown"),
        "data_start": min(str(f.date.min().date()) for f in frames.values()),
        "data_end": max(str(f.date.max().date()) for f in frames.values()),
        "symbols_with_valid_bars": len(frames),
        "source_collection_complete": bool(source_manifest.get("collection_complete", False)),
        "point_in_time_universe_complete": bool(source_manifest.get("point_in_time_universe_complete", False)),
        "delisted_price_coverage_verified": bool(source_manifest.get("delisted_price_coverage_verified", False)),
        "corporate_actions_complete": bool(source_manifest.get("corporate_actions_complete", False)),
        "annual_top20_years": sorted(universe.year.unique().astype(int).tolist()),
        "annual_top20_coverage": universe.groupby("year").symbol.nunique().to_dict(),
        "execution": "signal at adjusted close; next available raw open; integer shares; shared cash; max 5 positions; 0.5% risk budget; 20% max position; 1% ADV participation; ATR stop; 60-bar max hold",
        "costs": {"base_buy_fee": BUY_FEE, "base_sell_fee_plus_tax": SELL_FEE_TAX,
                   "base_slippage_each_side": SLIPPAGE,
                   "stress_multipliers": [1.0, 1.5, 2.0],
                   "stress_applies_to": "buy fee, sell fee plus tax, and slippage"},
        "limitations": [
            "Annual top-20 membership is ranked from prior calendar-year turnover, but candidate coverage is only as complete as the collected symbol list.",
            "Historical point-in-time exchange membership and delisted price coverage are not verified.",
            "Corporate action data may be incomplete; conflicting event rows are excluded.",
            "Price-limit, suspension, board-lot and partial-fill constraints are not fully simulated.",
            "Results are exploratory and must not be used as a live-trading performance claim.",
        ],
        "strategy_results": metrics,
    }
    (out / "research_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/taiwan_history")
    parser.add_argument("--output", default="artifacts/annual_top20_2016_2026")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
