"""Collect official FinMind TaiwanStockPriceAdj daily OHLCV for the ten-stock study.

Raw TaiwanStockPrice remains the execution-price source. This dataset is the
research-price source for historical indicators and return calculations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

API_URL = "https://api.finmindtrade.com/api/v4/data"
STOCKS = {
    "2330": "台積電", "2454": "聯發科", "1303": "南亞", "2327": "國巨",
    "2408": "南亞科", "2308": "台達電", "3037": "欣興", "3008": "大立光",
    "2383": "台光電", "2303": "聯電",
}


def fetch_rows(symbol: str, start_date: str, end_date: str | None,
               token: str, retries: int = 4) -> list[dict]:
    params = {"dataset": "TaiwanStockPriceAdj", "data_id": symbol,
              "start_date": start_date}
    if end_date:
        params["end_date"] = end_date
    if token:
        params["token"] = token
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(API_URL, params=params, timeout=(10, 60))
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or str(payload.get("status", "200")) not in {"200", "success", "Success"}:
                raise RuntimeError(f"FinMind API rejected {symbol}: {payload}")
            rows = payload.get("data")
            if not isinstance(rows, list):
                raise RuntimeError(f"FinMind returned invalid data for {symbol}")
            return rows
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                delay = min(30, 2 ** (attempt + 1))
                print(f"[RETRY] {symbol} {attempt + 1}/{retries}: {exc}; sleep={delay}s", flush=True)
                time.sleep(delay)
    raise RuntimeError(f"Unable to collect adjusted bars for {symbol}: {last_error}")


def normalize(rows: list[dict], symbol: str) -> pd.DataFrame:
    if not rows:
        raise RuntimeError(f"No adjusted price rows returned for {symbol}")
    d = pd.DataFrame(rows).rename(columns={
        "stock_id": "symbol", "max": "high", "min": "low",
        "Trading_Volume": "volume", "Trading_money": "turnover",
    })
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(d.columns)
    if missing:
        raise RuntimeError(f"{symbol}: missing adjusted-price fields {sorted(missing)}")
    d["symbol"] = str(symbol)
    d["name"] = STOCKS[symbol]
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ("open", "high", "low", "close", "volume", "turnover"):
        if col not in d:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    valid = (
        (d[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (d["volume"] >= 0)
        & (d["high"] >= d[["open", "close", "low"]].max(axis=1))
        & (d["low"] <= d[["open", "close", "high"]].min(axis=1))
    )
    d = d.loc[valid].drop_duplicates("date", keep="last").sort_values("date")
    if d.empty:
        raise RuntimeError(f"No valid adjusted OHLCV rows for {symbol}")
    return d[["symbol", "name", "date", "open", "high", "low", "close", "volume", "turnover"]].reset_index(drop=True)


def collect(output: Path, start_date: str, end_date: str | None,
            pause_seconds: float = 2.0) -> dict:
    token = os.getenv("FINMIND_TOKEN", "").strip()
    if not token:
        raise RuntimeError("FINMIND_TOKEN is required; refusing unauthenticated adjusted-price collection")
    bars_dir = output / "bars"
    bars_dir.mkdir(parents=True, exist_ok=True)
    status = []
    for i, (symbol, name) in enumerate(STOCKS.items(), 1):
        rows = fetch_rows(symbol, start_date, end_date, token)
        frame = normalize(rows, symbol)
        path = bars_dir / f"{symbol}.csv"
        frame.to_csv(path, index=False, float_format="%.8f")
        dates = pd.to_datetime(frame["date"])
        gaps = int(max(0, len(pd.bdate_range(dates.min(), dates.max())) - len(frame)))
        status.append({
            "symbol": symbol, "name": name, "status": "ok", "rows": int(len(frame)),
            "first_date": str(frame.date.iloc[0]), "last_date": str(frame.date.iloc[-1]),
            "business_day_gap_count_diagnostic_only": gaps,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        print(f"[ADJUSTED] {i}/{len(STOCKS)} {symbol} {name}: {len(frame)} rows {frame.date.iloc[0]}..{frame.date.iloc[-1]}", flush=True)
        if i < len(STOCKS):
            time.sleep(max(0.0, pause_seconds))
    complete = len(status) == len(STOCKS) and all(x["status"] == "ok" for x in status)
    manifest = {
        "source": "FinMind API v4", "dataset": "TaiwanStockPriceAdj",
        "price_semantics": "official back-adjusted OHLC for historical indicators and return calculations",
        "raw_execution_prices": "TaiwanStockPrice; retain separately for live/order price display",
        "volume_semantics": "Trading_Volume returned by FinMind; not assumed split-adjusted",
        "start_date_requested": start_date, "end_date_requested": end_date,
        "collected_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
        "token_configured": bool(token), "symbols_expected": len(STOCKS),
        "symbols_collected": len(status), "collection_complete": complete,
        "status_rows": status,
        "note": "FinMind describes TaiwanStockPriceAdj as ex-right/ex-dividend adjusted OHLC. The latest trading date is the adjustment base. Volume is not assumed adjusted.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not complete:
        raise RuntimeError("Adjusted-price collection incomplete")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/top10_adjusted")
    parser.add_argument("--start-date", default="2010-01-04")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    args = parser.parse_args()
    manifest = collect(Path(args.output), args.start_date, args.end_date, args.pause_seconds)
    print(json.dumps({k: v for k, v in manifest.items() if k != "status_rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
