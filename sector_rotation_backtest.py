#!/usr/bin/env python3
"""Monthly Taiwan sector rotation and cross-sectional mean-reversion backtest.

Signals are formed after a month-end close and filled no earlier than the next
available session open. Outputs are explicitly preliminary until point-in-time
industry/universe and corporate-action coverage are independently verified.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

def load_panel(root: Path, min_rows: int = 260):
    meta = pd.read_csv(root / "candidate_symbols.csv", dtype={"symbol": str})
    meta["symbol"] = meta["symbol"].astype(str).str.strip()
    meta = meta.drop_duplicates("symbol").set_index("symbol")
    bars_dir = root / "bars"
    symbols, calendar_parts = [], []
    for path in sorted(bars_dir.glob("*.csv")):
        try:
            d = pd.read_csv(path, usecols=["date"])
            dates = pd.to_datetime(d["date"], errors="coerce").dropna().drop_duplicates()
            if len(dates) >= min_rows:
                symbols.append(path.stem)
                calendar_parts.append(dates)
        except Exception:
            continue
    if not calendar_parts:
        raise RuntimeError("No usable price files found")
    calendar = pd.DatetimeIndex(sorted(pd.concat(calendar_parts).drop_duplicates()))
    month_ends = pd.DatetimeIndex(
        pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")).max().values
    )
    data, snapshots = {}, []
    for code in symbols:
        try:
            d = pd.read_csv(bars_dir / f"{code}.csv")
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                d[col] = pd.to_numeric(d[col], errors="coerce")
            d = d.dropna(subset=["date", "open", "high", "low", "close"])
            d = d.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
            if len(d) < min_rows:
                continue
            c = d["close"]
            sma50, sma200 = c.rolling(50, min_periods=50).mean(), c.rolling(200, min_periods=200).mean()
            ret5, ret20, ret60, ret120 = c.pct_change(5), c.pct_change(20), c.pct_change(60), c.pct_change(120)
            high20 = c.rolling(20, min_periods=20).max()
            high55 = d["high"].rolling(55, min_periods=55).max()
            distance_high55 = c / high55.replace(0, np.nan) - 1.0
            prev_close = c.shift(1)
            tr = pd.concat([d["high"] - d["low"], (d["high"] - prev_close).abs(),
                            (d["low"] - prev_close).abs()], axis=1).max(axis=1)
            atr_pct14 = tr.rolling(14, min_periods=14).mean() / c.replace(0, np.nan)
            avg_turnover20 = d["turnover"].rolling(20, min_periods=20).mean()
            avg_volume20 = d["volume"].rolling(20, min_periods=20).mean()
            industry = str(meta.loc[code, "industry_category"]) if code in meta.index else "未知"
            if not industry or industry.lower() == "nan":
                industry = "未知"
            frame = pd.DataFrame({
                "date": d["date"], "open": d["open"], "close": c, "sma50": sma50,
                "sma200": sma200, "ret5": ret5, "ret20": ret20, "ret60": ret60,
                "ret120": ret120, "high20": high20, "distance_high55": distance_high55,
                "atr_pct14": atr_pct14, "avg_turnover20": avg_turnover20,
                "avg_volume20": avg_volume20,
            })
            data[code] = {
                "industry": industry,
                "dates": frame["date"].to_numpy(dtype="datetime64[ns]"),
                **{key: frame[key].to_numpy(dtype=np.float32) for key in
                   ["open", "close", "sma50", "ret5", "ret20", "ret60", "ret120",
                    "high20", "avg_volume20"]},
            }
            monthly = frame.loc[frame["date"].isin(month_ends),
                ["date", "close", "sma50", "sma200", "ret5", "ret20", "ret60",
                 "ret120", "distance_high55", "atr_pct14", "avg_turnover20"]]
            monthly = monthly.replace([np.inf, -np.inf], np.nan).dropna()
            if not monthly.empty:
                eligible = (
                    (monthly["close"] >= 10) & (monthly["close"] > monthly["sma200"])
                    & (monthly["sma50"] > monthly["sma200"]) & (monthly["ret60"] > 0)
                    & (monthly["avg_turnover20"] >= 20_000_000)
                )
                monthly = monthly.assign(eligible=eligible, symbol=code, industry=industry)
                snapshots.extend(monthly[
                    ["date", "symbol", "industry", "close", "ret5", "ret20", "ret60",
                     "ret120", "distance_high55", "atr_pct14", "avg_turnover20", "eligible"]
                ].to_dict("records"))
        except Exception:
            continue
    snap = pd.DataFrame(snapshots)
    if snap.empty:
        raise RuntimeError("No monthly feature snapshots")
    return data, calendar, snap

def build_monthly_targets(snap, top_sectors=3, stocks_per_sector=3, min_sector_names=5):
    targets, sector_rows, stock_rows = {}, [], []
    for day, group in snap.groupby("date", sort=True):
        eligible = group[group["eligible"]].copy()
        if eligible.empty:
            targets[day] = set()
            continue
        sector = eligible.groupby("industry").agg(
            names=("symbol", "count"), median_ret60=("ret60", "median"),
            median_ret120=("ret120", "median"),
        ).reset_index()
        totals = group.groupby("industry")["symbol"].nunique().rename("universe_names")
        sector = sector.merge(totals, on="industry", how="left")
        sector["breadth"] = sector["names"] / sector["universe_names"].clip(lower=1)
        sector = sector[sector["names"] >= min_sector_names].copy()
        if sector.empty:
            targets[day] = set()
            continue
        sector["score"] = (
            0.60 * sector["median_ret60"].rank(pct=True)
            + 0.30 * sector["median_ret120"].rank(pct=True)
            + 0.10 * sector["breadth"].rank(pct=True)
        )
        sector = sector.sort_values(["score", "names"], ascending=False)
        chosen_names = set(sector.head(top_sectors)["industry"])
        sector["selected"] = sector["industry"].isin(chosen_names)
        for r in sector.itertuples(index=False):
            sector_rows.append({"date": day, "industry": r.industry, "names": int(r.names),
                "median_ret60": r.median_ret60, "median_ret120": r.median_ret120,
                "breadth": r.breadth, "score": r.score, "selected": bool(r.selected)})
        pool = eligible[eligible["industry"].isin(chosen_names)].copy()
        pool["score"] = (
            0.45 * pool.groupby("industry")["atr_pct14"].rank(pct=True, ascending=False)
            + 0.30 * pool.groupby("industry")["ret5"].rank(pct=True, ascending=False)
            + 0.15 * pool.groupby("industry")["distance_high55"].rank(pct=True, ascending=True)
            + 0.10 * pool.groupby("industry")["ret120"].rank(pct=True, ascending=True)
        )
        chosen_rows = []
        for _, sub in pool.groupby("industry"):
            chosen_rows.extend(sub.sort_values(["score", "avg_turnover20"], ascending=False)
                               .head(stocks_per_sector).to_dict("records"))
        names = set()
        for r in chosen_rows:
            names.add(r["symbol"])
            stock_rows.append({"date": day, "symbol": r["symbol"], "industry": r["industry"],
                "score": r["score"], "ret5": r["ret5"], "ret20": r["ret20"],
                "ret60": r["ret60"], "ret120": r["ret120"],
                "distance_high55": r["distance_high55"], "atr_pct14": r["atr_pct14"],
                "avg_turnover20": r["avg_turnover20"]})
        targets[day] = names
    return targets, pd.DataFrame(sector_rows), pd.DataFrame(stock_rows)

def at(data, code, day):
    z = data[code]
    i = int(np.searchsorted(z["dates"], np.datetime64(day)))
    return i if i < len(z["dates"]) and z["dates"][i] == np.datetime64(day) else None

def run_portfolio(data, calendar, targets, fee=0.001425, tax=0.003, slip=0.001,
                  stop_pct=0.12, max_positions=9, max_participation=0.01,
                  initial_capital=1_000_000):
    cash, positions, trades = float(initial_capital), {}, []
    equity_rows, last_close = [], {}
    target_by_next = {}
    for day, names in targets.items():
        k = int(np.searchsorted(calendar.values, np.datetime64(day)))
        if k + 1 < len(calendar):
            target_by_next[calendar[k + 1]] = set(names)
    pending_target, pending_stop, pending_exit = None, set(), set()
    total_fees = total_tax = total_slippage = 0.0
    for day in calendar:
        if day in target_by_next:
            pending_target = target_by_next[day]
            pending_exit.update(code for code in positions if code not in pending_target)
        sold_today = set()
        # Sells execute before buys at the next available open.
        for code in list(positions):
            idx = at(data, code, day)
            if idx is None or (code not in pending_stop and code not in pending_exit):
                continue
            p = positions.pop(code)
            op = float(data[code]["open"][idx])
            if not np.isfinite(op) or op <= 0:
                positions[code] = p
                continue
            gross = p["shares"] * op
            slip_cost = gross * slip
            proceeds = gross - slip_cost - gross * fee - gross * tax
            cash += proceeds
            total_fees += gross * fee
            total_tax += gross * tax
            total_slippage += slip_cost
            pnl = proceeds - p["entry_cost"]
            trades.append({
                "symbol": code, "industry": p["industry"],
                "entry_date": str(p["entry_date"].date()), "exit_date": str(pd.Timestamp(day).date()),
                "entry_price": p["entry_price"], "exit_price": op * (1 - slip),
                "shares": p["shares"], "entry_cost": p["entry_cost"],
                "exit_proceeds": proceeds, "pnl": pnl,
                "return": pnl / p["entry_cost"] if p["entry_cost"] else 0,
                "exit_reason": "stop" if code in pending_stop else "rebalance",
            })
            last_close[code] = op
            pending_stop.discard(code)
            pending_exit.discard(code)
            sold_today.add(code)
        if pending_target is not None:
            desired = list(pending_target)
            held = set(positions)
            candidates = [c for c in desired if c not in held and c not in sold_today
                          and c not in pending_stop and c in data]
            marked = sum(p["shares"] * last_close.get(c, p["entry_price"])
                         for c, p in positions.items())
            equity_open = max(0.0, cash + marked)
            slots = max(1, min(max_positions, len(desired)))
            per_name = min(equity_open * 0.10, equity_open / slots)
            for code in candidates:
                if len(positions) >= max_positions:
                    break
                idx = at(data, code, day)
                if idx is None:
                    continue
                op, adv = float(data[code]["open"][idx]), float(data[code]["avg_volume20"][idx])
                if not np.isfinite(op) or op <= 0 or not np.isfinite(adv) or adv <= 0:
                    continue
                max_shares = max(0, int(math.floor(adv * max_participation)))
                budget = min(per_name, cash / (1 + fee + slip))
                shares = min(max_shares, int(math.floor(budget / (op * (1 + fee + slip)))))
                if shares <= 0:
                    continue
                gross = shares * op
                slip_cost, commission = gross * slip, gross * fee
                cost = gross + slip_cost + commission
                if cost > cash:
                    continue
                cash -= cost
                total_fees += commission
                total_slippage += slip_cost
                positions[code] = {"shares": shares, "entry_price": op * (1 + slip),
                    "entry_cost": cost, "entry_date": pd.Timestamp(day),
                    "industry": data[code]["industry"]}
                last_close[code] = op
            pending_target = None
        market_value = 0.0
        for code, p in list(positions.items()):
            idx = at(data, code, day)
            if idx is not None:
                cl = float(data[code]["close"][idx])
                if np.isfinite(cl) and cl > 0:
                    last_close[code] = cl
                    sma, high20 = float(data[code]["sma50"][idx]), float(data[code]["high20"][idx])
                    if (np.isfinite(sma) and cl < sma) or (
                        np.isfinite(high20) and high20 > 0 and cl < high20 * (1 - stop_pct)
                    ):
                        pending_stop.add(code)
            market_value += p["shares"] * last_close.get(code, p["entry_price"])
        nav = cash + market_value
        equity_rows.append({"date": day, "cash": cash, "market_value": market_value,
            "equity": nav, "positions": len(positions), "exposure": market_value / nav if nav > 0 else 0})
    eq, tr = pd.DataFrame(equity_rows), pd.DataFrame(trades)
    return eq, tr, {"fees": total_fees, "sell_tax": total_tax, "slippage": total_slippage,
        "ending_cash": cash, "open_positions": len(positions)}

def metrics(eq, tr, initial_capital, period_start=None):
    if eq.empty:
        return {}
    full = eq.copy()
    full["date"] = pd.to_datetime(full["date"])
    q, trades_period = full.copy(), tr.copy()
    if period_start is None:
        start_nav, start_date = float(initial_capital), pd.Timestamp(full.date.iloc[0])
    else:
        cut = pd.Timestamp(period_start)
        q = q[q.date >= cut]
        prior = full.loc[full.date < cut, "equity"]
        start_nav, start_date = (float(prior.iloc[-1]) if not prior.empty else float(initial_capital)), cut
        if not trades_period.empty:
            trades_period = trades_period[pd.to_datetime(trades_period["exit_date"]) >= cut]
    if q.empty:
        return {}
    v = q["equity"].astype(float).to_numpy()
    rets = pd.Series(np.r_[v[0] / start_nav - 1.0, pd.Series(v).pct_change().dropna().to_numpy()])
    years = max((pd.Timestamp(q.date.iloc[-1]) - start_date).days / 365.25, 1 / 252)
    cagr = (v[-1] / start_nav) ** (1 / years) - 1 if v[-1] > 0 and start_nav > 0 else -1
    path = np.r_[start_nav, v]
    dd = path / np.maximum.accumulate(path) - 1
    vol = float(rets.std(ddof=1) * np.sqrt(252)) if len(rets) > 1 else 0
    sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(252)) if len(rets) > 1 and rets.std(ddof=1) > 0 else 0
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std(ddof=1) * np.sqrt(252)) if len(downside) > 1 and downside.std(ddof=1) > 0 else 0
    win = float((trades_period["pnl"] > 0).mean()) if not trades_period.empty else 0
    gp = float(trades_period.loc[trades_period.pnl > 0, "pnl"].sum()) if not trades_period.empty else 0
    gl = float(-trades_period.loc[trades_period.pnl < 0, "pnl"].sum()) if not trades_period.empty else 0
    return {"start": str(start_date.date()), "end": str(pd.Timestamp(q.date.iloc[-1]).date()),
        "sessions": int(len(q)), "starting_equity": start_nav, "ending_equity": float(v[-1]),
        "total_return": float(v[-1] / start_nav - 1), "cagr": float(cagr),
        "max_drawdown": float(dd.min()), "annualized_volatility": vol,
        "sharpe": sharpe, "sortino": sortino,
        "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else None,
        "closed_trades": int(len(trades_period)), "win_rate": win,
        "profit_factor": gp / gl if gl > 0 else None,
        "average_trade_return": float(trades_period["return"].mean()) if not trades_period.empty else 0,
        "average_exposure": float(q["exposure"].mean())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/taiwan_history")
    ap.add_argument("--output", default="artifacts/sector_rotation")
    ap.add_argument("--initial-capital", type=float, default=1_000_000)
    ap.add_argument("--top-sectors", type=int, default=3)
    ap.add_argument("--stocks-per-sector", type=int, default=3)
    ap.add_argument("--stop-pct", type=float, default=0.12)
    args = ap.parse_args()
    root, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data, calendar, snap = load_panel(root)
    targets, sector_scores, picks = build_monthly_targets(snap, args.top_sectors, args.stocks_per_sector)
    scenarios = [("base", .001425, .003, .001), ("2x_cost", .00285, .006, .002),
                 ("3x_cost", .004275, .009, .003)]
    summaries, base_eq, base_tr = {}, None, None
    for name, fee, tax, slip in scenarios:
        eq, tr, costs = run_portfolio(data, calendar, targets, fee=fee, tax=tax, slip=slip,
            stop_pct=args.stop_pct, max_positions=args.top_sectors * args.stocks_per_sector,
            initial_capital=args.initial_capital)
        summaries[name] = {"full_history": metrics(eq, tr, args.initial_capital),
            "post_2022": metrics(eq, tr, args.initial_capital, "2022-01-01"),
            "costs": costs, "assumptions": {"fee": fee, "sell_tax": tax, "slippage_each_side": slip}}
        if name == "base":
            base_eq, base_tr = eq, tr
    base_eq.to_csv(out / "equity_curve.csv", index=False)
    base_tr.to_csv(out / "trades.csv", index=False)
    sector_scores.to_csv(out / "sector_monthly_scores.csv", index=False)
    picks.to_csv(out / "monthly_stock_selections.csv", index=False)
    annual = base_eq.copy()
    annual["date"] = pd.to_datetime(annual["date"])
    annual["daily_return"] = annual["equity"].pct_change()
    if not annual.empty:
        annual.loc[annual.index[0], "daily_return"] = annual.loc[annual.index[0], "equity"] / args.initial_capital - 1
    annual["year"] = annual["date"].dt.year
    annual.groupby("year")["daily_return"].apply(lambda x: (1 + x).prod() - 1).rename("return").reset_index().to_csv(
        out / "annual_returns.csv", index=False)
    meta = pd.read_csv(root / "candidate_symbols.csv", dtype={"symbol": str})
    manifest = json.loads((root / "download_manifest.json").read_text()) if (root / "download_manifest.json").exists() else {}
    action_path = root / "corporate_actions" / "manifest.json"
    action_manifest = json.loads(action_path.read_text()) if action_path.exists() else {}
    report = {
        "strategy": "Monthly sector rotation + low-volatility pullback + long-term trend filter",
        "status": "PRELIMINARY_NOT_RESEARCH_GRADE",
        "data": {"candidate_symbols": int(len(meta)), "symbols_with_usable_history": int(len(data)),
            "daily_rows": int(sum(len(x["dates"]) for x in data.values())),
            "first_date": str(calendar[0].date()), "last_date": str(calendar[-1].date()),
            "point_in_time_universe_complete": bool(manifest.get("point_in_time_universe_complete", False)),
            "delisted_price_coverage_verified": bool(manifest.get("delisted_price_coverage_verified", False)),
            "corporate_actions_complete": bool(action_manifest.get("corporate_actions_complete", False)),
            "corporate_actions_applied_to_portfolio": False,
            "industry_classification": "current candidate_symbols.csv snapshot; historical industry membership not verified"},
        "rules": {"eligible": "close>10; close>SMA200; SMA50>SMA200; 60d momentum positive; 20d average turnover >= NTD 20m",
            "sector_score": "60% median 60d momentum rank + 30% median 120d momentum rank + 10% eligible breadth rank",
            "stock_score": "45% low ATR% rank + 30% 5d reversal rank + 15% 55d-high proximity + 10% 120d momentum rank",
            "rebalance": "monthly after final common trading-day close; fill next available session open",
            "positions": int(args.top_sectors * args.stocks_per_sector), "max_position_weight": .10,
            "stop": "exit next available open after close below SMA50 or trailing 20-session closing-high drawdown exceeds configured stop",
            "max_participation_rate": .01, "fractional_shares": False},
        "scenarios": summaries,
        "limitations": [
            "Current industry labels are applied historically; point-in-time industry membership is unavailable.",
            "Candidate list is not a verified point-in-time historical universe; survivorship and delisted coverage remain unresolved.",
            "This runner does not yet apply corporate-action events to portfolio NAV or signal prices; dividends, stock distributions, splits and capital reductions may distort results.",
            "No official-exchange raw-price reconciliation or full official calendar validation has been completed.",
            "This is a preliminary research result, not a 300% annual-return claim or a live-trading recommendation."
        ]}
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "symbols": len(data), "calendar_sessions": len(calendar),
        "scenarios": summaries, "output": str(out)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
