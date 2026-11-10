"""Collect TSMC 2330 institutional and broker-branch flow data.

Institutional data uses FinMind's free single-symbol wide dataset.
Broker-branch daily detail is a separate Sponsor-tier endpoint and is only
collected when FINMIND_TOKEN is configured and authorized. Missing broker
coverage is represented as unavailable, never as zero.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

DATA_URL = "https://api.finmindtrade.com/api/v4/data"
BROKER_URL = "https://api.finmindtrade.com/api/v4/taiwan_stock_trading_daily_report"


def _get_json(url: str, params: dict[str, Any], *, token: str = "", timeout: int = 30) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(url, params=params, headers=headers, timeout=(8, timeout))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("FinMind response is not a JSON object")
    if str(payload.get("status", "200")) not in {"200", "success", "Success"}:
        raise RuntimeError(str(payload.get("msg", payload.get("detail", "FinMind API error"))))
    return payload


def _to_number(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce").fillna(0.0)


def normalize_institutional(rows: list[dict]) -> pd.DataFrame:
    """Normalize FinMind wide data to shares; foreign dealers remain separate."""
    cols = [
        "date", "foreign_buy_shares", "foreign_sell_shares", "foreign_net_shares",
        "foreign_dealer_buy_shares", "foreign_dealer_sell_shares", "foreign_dealer_net_shares",
        "trust_buy_shares", "trust_sell_shares", "trust_net_shares",
        "dealer_buy_shares", "dealer_sell_shares", "dealer_net_shares",
        "institutional_buy_shares", "institutional_sell_shares", "institutional_net_shares",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    z = pd.DataFrame(rows).copy()
    if "date" not in z:
        raise ValueError("Institutional dataset missing date")
    z["date"] = pd.to_datetime(z["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    # Official three-institution total excludes foreign dealers from the foreign category.
    # Keep foreign-dealer proprietary trades as a separate series, not in the three-major total.
    foreign_buy = _to_number(z, "Foreign_Investor_buy")
    foreign_sell = _to_number(z, "Foreign_Investor_sell")
    foreign_dealer_buy = _to_number(z, "Foreign_Dealer_Self_buy")
    foreign_dealer_sell = _to_number(z, "Foreign_Dealer_Self_sell")
    trust_buy = _to_number(z, "Investment_Trust_buy")
    trust_sell = _to_number(z, "Investment_Trust_sell")
    # Legacy Dealer_buy/sell may be populated in older records. Prefer the
    # explicit proprietary + hedging columns when present to avoid double count.
    dealer_buy = (
        _to_number(z, "Dealer_self_buy") + _to_number(z, "Dealer_Hedging_buy")
        + _to_number(z, "Dealer_buy")
    )
    dealer_sell = (
        _to_number(z, "Dealer_self_sell") + _to_number(z, "Dealer_Hedging_sell")
        + _to_number(z, "Dealer_sell")
    )
    # The old aggregate field is only used where the detailed components are absent.
    detailed_buy = _to_number(z, "Dealer_self_buy") + _to_number(z, "Dealer_Hedging_buy")
    detailed_sell = _to_number(z, "Dealer_self_sell") + _to_number(z, "Dealer_Hedging_sell")
    legacy_buy = _to_number(z, "Dealer_buy")
    legacy_sell = _to_number(z, "Dealer_sell")
    dealer_buy = detailed_buy.where((detailed_buy != 0) | (detailed_sell != 0), legacy_buy)
    dealer_sell = detailed_sell.where((detailed_buy != 0) | (detailed_sell != 0), legacy_sell)
    out = pd.DataFrame({
        "date": z["date"],
        "foreign_buy_shares": foreign_buy,
        "foreign_sell_shares": foreign_sell,
        "foreign_net_shares": foreign_buy - foreign_sell,
        "foreign_dealer_buy_shares": foreign_dealer_buy,
        "foreign_dealer_sell_shares": foreign_dealer_sell,
        "foreign_dealer_net_shares": foreign_dealer_buy - foreign_dealer_sell,
        "trust_buy_shares": trust_buy,
        "trust_sell_shares": trust_sell,
        "trust_net_shares": trust_buy - trust_sell,
        "dealer_buy_shares": dealer_buy,
        "dealer_sell_shares": dealer_sell,
        "dealer_net_shares": dealer_buy - dealer_sell,
    })
    out["institutional_buy_shares"] = out[[
        "foreign_buy_shares", "trust_buy_shares", "dealer_buy_shares"
    ]].sum(axis=1)
    out["institutional_sell_shares"] = out[[
        "foreign_sell_shares", "trust_sell_shares", "dealer_sell_shares"
    ]].sum(axis=1)
    out["institutional_net_shares"] = out[[
        "foreign_net_shares", "trust_net_shares", "dealer_net_shares"
    ]].sum(axis=1)
    out = out.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
    return out.reset_index(drop=True)[cols]


def add_institutional_features(frame: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Create causal daily flow features; values are shares unless noted."""
    z = frame.copy().sort_values("date").reset_index(drop=True)
    volume = pd.to_numeric(bars.set_index(pd.to_datetime(bars["date"]))["volume"], errors="coerce")
    z_dates = pd.to_datetime(z["date"])
    z["volume_shares"] = z_dates.map(volume).astype(float)
    for group in ("foreign", "foreign_dealer", "trust", "dealer", "institutional"):
        net = f"{group}_net_shares"
        z[f"{group}_net_lots"] = z[net] / 1000.0
        z[f"{group}_net_to_volume"] = z[net] / z["volume_shares"].replace(0, np.nan)
        z[f"{group}_net_5d_lots"] = z[net].rolling(5, min_periods=1).sum() / 1000.0
        z[f"{group}_net_20d_lots"] = z[net].rolling(20, min_periods=1).sum() / 1000.0
        z[f"{group}_net_positive_streak"] = (
            (z[net].gt(0).groupby(z[net].le(0).cumsum()).cumcount() + 1)
            .where(z[net].gt(0), 0)
        )
        z[f"{group}_net_negative_streak"] = (
            (z[net].lt(0).groupby(z[net].ge(0).cumsum()).cumcount() + 1)
            .where(z[net].lt(0), 0)
        )
    z["institutional_data_available"] = 1
    return z


def normalize_broker_rows(rows: list[dict], trade_date: str) -> pd.DataFrame:
    """Normalize daily broker-branch data; source quantities are shares."""
    cols = [
        "date", "stock_id", "securities_trader_id", "securities_trader",
        "buy_shares", "sell_shares", "net_shares", "buy_lots", "sell_lots", "net_lots",
        "price",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    z = pd.DataFrame(rows).copy()
    required = {"securities_trader_id", "buy", "sell"}
    if not required.issubset(z.columns):
        raise ValueError(f"Broker endpoint schema missing {sorted(required - set(z.columns))}")
    z["date"] = trade_date
    z["stock_id"] = z.get("stock_id", "2330").astype(str) if isinstance(z.get("stock_id"), pd.Series) else "2330"
    z["securities_trader_id"] = z["securities_trader_id"].astype(str).str.strip()
    z["securities_trader"] = z.get("securities_trader", "").astype(str) if isinstance(z.get("securities_trader"), pd.Series) else ""
    z["buy_shares"] = pd.to_numeric(z["buy"], errors="coerce")
    z["sell_shares"] = pd.to_numeric(z["sell"], errors="coerce")
    z["net_shares"] = z["buy_shares"] - z["sell_shares"]
    z["buy_lots"] = z["buy_shares"] / 1000.0
    z["sell_lots"] = z["sell_shares"] / 1000.0
    z["net_lots"] = z["net_shares"] / 1000.0
    z["price"] = pd.to_numeric(z.get("price", np.nan), errors="coerce")
    return z[cols].dropna(subset=["buy_shares", "sell_shares"]).reset_index(drop=True)


def broker_daily_features(branches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate branch-level concentration features by date, preserving raw rows separately."""
    cols = [
        "date", "broker_data_available", "broker_branch_count", "broker_buy_lots",
        "broker_sell_lots", "broker_net_lots", "broker_abs_net_lots",
        "broker_top5_net_lots", "broker_top10_net_lots",
        "broker_top5_buy_net_lots", "broker_top10_buy_net_lots",
        "broker_top5_sell_net_lots", "broker_top10_sell_net_lots",
        "broker_top5_abs_net_share", "broker_buy_sell_balance_diff_lots",
    ]
    if branches.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for day, g in branches.groupby("date", sort=True):
        g = g.sort_values("net_lots", key=lambda s: s.abs(), ascending=False)
        abs_total = float(g["net_lots"].abs().sum())
        rows.append({
            "date": day, "broker_data_available": 1,
            "broker_branch_count": int(g["securities_trader_id"].nunique()),
            "broker_buy_lots": float(g["buy_lots"].sum()),
            "broker_sell_lots": float(g["sell_lots"].sum()),
            "broker_net_lots": float(g["net_lots"].sum()),
            "broker_abs_net_lots": abs_total,
            "broker_top5_net_lots": float(g.head(5)["net_lots"].sum()),
            "broker_top10_net_lots": float(g.head(10)["net_lots"].sum()),
            "broker_top5_buy_net_lots": float(g.loc[g["net_lots"] > 0].nlargest(5, "net_lots")["net_lots"].sum()),
            "broker_top10_buy_net_lots": float(g.loc[g["net_lots"] > 0].nlargest(10, "net_lots")["net_lots"].sum()),
            "broker_top5_sell_net_lots": float(g.loc[g["net_lots"] < 0].nsmallest(5, "net_lots")["net_lots"].sum()),
            "broker_top10_sell_net_lots": float(g.loc[g["net_lots"] < 0].nsmallest(10, "net_lots")["net_lots"].sum()),
            "broker_top5_abs_net_share": float(g.head(5)["net_lots"].abs().sum() / abs_total) if abs_total else 0.0,
            "broker_buy_sell_balance_diff_lots": float(g["buy_lots"].sum() - g["sell_lots"].sum()),
        })
    return pd.DataFrame(rows, columns=cols)


def collect(symbol: str, start_date: str, end_date: str, output_dir: str,
            token: str = "", broker_days: int = 20) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySellWide",
        "data_id": symbol, "start_date": start_date, "end_date": end_date,
    }
    inst_payload = _get_json(DATA_URL, params, token=token, timeout=60)
    inst = normalize_institutional(inst_payload.get("data", []))
    inst_path = out / f"{symbol}_institutional.csv"
    inst.to_csv(inst_path, index=False)

    # Branch-level detail is Sponsor-tier and queried once per trading day.
    # The coverage ledger supports a bounded historical backfill in CI cache.
    bars_path = out.parent / "bars" / f"{symbol}.csv"
    if not bars_path.exists():
        raise FileNotFoundError(f"Missing price history: {bars_path}")
    bars = pd.read_csv(bars_path, parse_dates=["date"])
    all_dates = (
        bars.loc[bars["date"].dt.strftime("%Y-%m-%d").between("2021-06-30", end_date), "date"]
        .drop_duplicates().sort_values().dt.strftime("%Y-%m-%d").tolist()
    )
    branches_path = out / f"{symbol}_broker_branches.csv"
    coverage_path = out / f"{symbol}_broker_coverage.csv"
    branch_columns = [
        "date", "stock_id", "securities_trader_id", "securities_trader",
        "buy_shares", "sell_shares", "net_shares", "buy_lots", "sell_lots", "net_lots", "price"
    ]
    existing_branches = (
        pd.read_csv(branches_path, dtype={"securities_trader_id": str})
        if branches_path.exists() else pd.DataFrame(columns=branch_columns)
    )
    coverage = (
        pd.read_csv(coverage_path, dtype={"date": str})
        if coverage_path.exists() else pd.DataFrame(columns=["date", "status", "row_count", "retrieved_at"])
    )
    covered_dates = set(coverage.get("date", pd.Series(dtype=str)).astype(str))
    missing_dates = [d for d in all_dates if d not in covered_dates]
    backfill_complete = not missing_dates
    selected_dates = missing_dates[:max(0, int(broker_days))]
    if backfill_complete and broker_days:
        selected_dates = all_dates[-int(broker_days):]
    branch_frames, new_coverage = [], []
    broker_status, broker_errors = "skipped_no_token", []
    if token and selected_dates:
        broker_status = "backfilling" if missing_dates else "refreshing_latest"
        for day in selected_dates:
            try:
                payload = _get_json(
                    BROKER_URL, {"data_id": symbol, "date": day}, token=token, timeout=40
                )
                rows = payload.get("data", [])
                if not isinstance(rows, list):
                    raise RuntimeError("Broker data field is not a list")
                day_frame = normalize_broker_rows(rows, day)
                branch_frames.append(day_frame)
                new_coverage.append({
                    "date": day, "status": "downloaded" if len(day_frame) else "empty",
                    "row_count": int(len(day_frame)),
                    "retrieved_at": datetime.now().astimezone().isoformat(),
                })
            except Exception as exc:
                broker_status = "partial" if branch_frames else "unavailable"
                broker_errors.append({"date": day, "error": str(exc)[:300]})
                break
    if new_coverage:
        refreshed = set(row["date"] for row in new_coverage)
        coverage = pd.concat([coverage, pd.DataFrame(new_coverage)], ignore_index=True)
        coverage = coverage.drop_duplicates("date", keep="last").sort_values("date")
        existing_branches = existing_branches.loc[
            ~existing_branches["date"].astype(str).isin(refreshed)
        ]
    branches = pd.concat([existing_branches, *branch_frames], ignore_index=True) if branch_frames else existing_branches
    if not branches.empty:
        branches["date"] = branches["date"].astype(str)
        branches["securities_trader_id"] = branches["securities_trader_id"].astype(str)
        branches = branches.drop_duplicates(
            ["date", "stock_id", "securities_trader_id"], keep="last"
        ).sort_values(["date", "securities_trader_id"])
    branches.to_csv(branches_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    broker_features = broker_daily_features(branches)
    broker_features.to_csv(out / f"{symbol}_broker_daily_features.csv", index=False)

    inst_features = add_institutional_features(inst, bars)
    # Broker columns are NaN when unavailable; do not encode lack of access as zero activity.
    combined = inst_features.merge(broker_features, on="date", how="left")
    if "broker_data_available" not in combined:
        combined["broker_data_available"] = np.nan
    combined_path = out / f"{symbol}_chip_features.csv"
    combined.to_csv(combined_path, index=False)
    manifest = {
        "symbol": symbol, "source": "FinMind API v4",
        "institutional_dataset": "TaiwanStockInstitutionalInvestorsBuySellWide",
        "institutional_status": "downloaded" if len(inst) else "empty",
        "institutional_rows": int(len(inst)),
        "institutional_first_date": str(inst["date"].min()) if len(inst) else None,
        "institutional_last_date": str(inst["date"].max()) if len(inst) else None,
        "institutional_units": "shares; *_lots columns are shares / 1000",
        "broker_dataset": "TaiwanStockTradingDailyReport",
        "broker_tier": "FinMind Sponsor",
        "broker_status": broker_status,
        "broker_days_requested": int(len(selected_dates)),
        "broker_days_covered_total": int(coverage["date"].nunique()) if not coverage.empty else 0,
        "broker_backfill_complete": bool(set(all_dates).issubset(set(coverage.get("date", pd.Series(dtype=str)).astype(str)))),
        "broker_first_covered_date": str(coverage["date"].min()) if not coverage.empty else None,
        "broker_last_covered_date": str(coverage["date"].max()) if not coverage.empty else None,
        "broker_days_returned": int(branches["date"].nunique()) if not branches.empty else 0,
        "broker_rows": int(len(branches)),
        "broker_units": "source buy/sell are shares; converted to lots by /1000",
        "broker_errors": broker_errors,
        "token_configured": bool(token),
        "retrieved_at_taipei": datetime.now().astimezone().isoformat(),
        "note": "Institutional and broker-branch data are separate datasets. Missing broker coverage is null, never zero. Broker branch history is backfilled in bounded batches using a coverage ledger; persist the chip directory between workflow runs using GitHub Actions cache."
    }
    (out / f"{symbol}_chip_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect TSMC 2330 institutional and broker-branch flows")
    parser.add_argument("--symbol", default="2330")
    parser.add_argument("--start-date", default="2010-01-04")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--output", default="data/tsmc_2330/chip")
    parser.add_argument("--broker-days", type=int, default=20)
    args = parser.parse_args()
    result = collect(
        args.symbol, args.start_date, args.end_date, args.output,
        token=os.getenv("FINMIND_TOKEN", "").strip(), broker_days=args.broker_days,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
