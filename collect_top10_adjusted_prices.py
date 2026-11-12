"""Collect adjusted OHLCV for the ten-stock Taiwan study.

Provider policy:
- auto: try FinMind TaiwanStockPriceAdj when a token exists; fall back per
  symbol to Yahoo Finance auto_adjust=True if FinMind fails.
- finmind: require FinMind; fail closed if unavailable.
- yahoo: force Yahoo Finance for every symbol.

Adjusted bars are for historical indicators/returns only. Raw exchange bars
remain the execution-price source and must be stored separately.
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
    params = {
        "dataset": "TaiwanStockPriceAdj", "data_id": symbol,
        "start_date": start_date, "token": token,
    }
    if end_date:
        params["end_date"] = end_date
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(API_URL, params=params, timeout=(10, 60))
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or str(payload.get("status", "200")) not in {"200", "success", "Success"}:
                raise RuntimeError(f"FinMind API rejected {symbol}: {payload}")
            rows = payload.get("data")
            if not isinstance(rows, list) or not rows:
                raise RuntimeError(f"FinMind returned no adjusted price rows for {symbol}")
            return rows
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                delay = min(30, 2 ** (attempt + 1))
                print(f"[FINMIND RETRY] {symbol} {attempt + 1}/{retries}: {exc}; sleep={delay}s", flush=True)
                time.sleep(delay)
    raise RuntimeError(f"Unable to collect adjusted bars for {symbol}: {last_error}")


def fetch_yahoo_history(symbol: str, start_date: str, end_date: str | None):
    import yfinance as yf

    return yf.Ticker(f"{symbol}.TW").history(
        start=start_date, end=end_date, auto_adjust=True, actions=True,
        repair=True, raise_errors=True, timeout=60,
    )


def _validate_ohlc(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    d = frame.copy()
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
    return d.reset_index(drop=True)


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
    d["symbol"], d["name"] = str(symbol), STOCKS[symbol]
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ("open", "high", "low", "close", "volume", "turnover"):
        if col not in d:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = _validate_ohlc(d, symbol)
    d["provider"] = "FinMind TaiwanStockPriceAdj"
    return d[["symbol", "name", "date", "open", "high", "low", "close", "volume", "turnover", "provider"]]


def normalize_yahoo(history: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if history is None or history.empty:
        raise RuntimeError(f"Yahoo Finance returned no history for {symbol}.TW")
    d = history.copy().reset_index()
    date_col = "Date" if "Date" in d.columns else "Datetime"
    if date_col not in d.columns:
        raise RuntimeError(f"Yahoo Finance date column missing for {symbol}")
    d = d.rename(columns={
        date_col: "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume", "Dividends": "dividends",
        "Stock Splits": "stock_splits",
    })
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ("open", "high", "low", "close", "volume"):
        if col not in d.columns:
            raise RuntimeError(f"Yahoo Finance field {col} missing for {symbol}")
        d[col] = pd.to_numeric(d[col], errors="coerce")
    for col in ("dividends", "stock_splits"):
        if col not in d.columns:
            d[col] = 0.0
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
    d["symbol"], d["name"] = str(symbol), STOCKS[symbol]
    # Adjusted close * volume is not raw traded value; never fabricate turnover.
    d["turnover"] = np.nan
    d = _validate_ohlc(d, symbol)
    d["provider"] = "Yahoo Finance auto_adjust=True"
    return d[[
        "symbol", "name", "date", "open", "high", "low", "close", "volume",
        "turnover", "dividends", "stock_splits", "provider",
    ]]


def fetch_symbol(symbol: str, start_date: str, end_date: str | None,
                 token: str, provider: str = "auto") -> tuple[pd.DataFrame, str]:
    """Fetch one symbol with explicit, auditable provider fallback."""
    if provider not in {"auto", "finmind", "yahoo"}:
        raise ValueError("provider must be auto, finmind, or yahoo")
    finmind_error = None
    if provider != "yahoo":
        if token:
            try:
                frame = normalize(fetch_rows(symbol, start_date, end_date, token), symbol)
                return frame, "FinMind TaiwanStockPriceAdj"
            except Exception as exc:
                finmind_error = exc
                if provider == "finmind":
                    raise
                print(f"[FALLBACK] {symbol}: FinMind failed ({type(exc).__name__}: {exc}); trying Yahoo", flush=True)
        elif provider == "finmind":
            raise RuntimeError("FINMIND_TOKEN is required when --provider finmind")
    try:
        frame = normalize_yahoo(fetch_yahoo_history(symbol, start_date, end_date), symbol)
        return frame, "Yahoo Finance auto_adjust=True"
    except Exception as yahoo_error:
        if finmind_error is not None:
            raise RuntimeError(
                f"{symbol}: FinMind failed ({finmind_error}); Yahoo fallback failed ({yahoo_error})"
            ) from yahoo_error
        raise


def collect(output: Path, start_date: str, end_date: str | None,
            pause_seconds: float = 2.0, provider: str = "auto") -> dict:
    token = os.getenv("FINMIND_TOKEN", "").strip()
    if provider == "finmind" and not token:
        raise RuntimeError("--provider finmind requires FINMIND_TOKEN")
    bars_dir = output / "bars"
    bars_dir.mkdir(parents=True, exist_ok=True)
    status = []
    for index, (symbol, name) in enumerate(STOCKS.items(), 1):
        frame, source = fetch_symbol(symbol, start_date, end_date, token, provider)
        path = bars_dir / f"{symbol}.csv"
        frame.to_csv(path, index=False, float_format="%.8f")
        status.append({
            "symbol": symbol, "name": name, "status": "ok",
            "provider": source, "rows": int(len(frame)),
            "first_date": str(frame.date.iloc[0]),
            "last_date": str(frame.date.iloc[-1]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        print(
            f"[ADJUSTED] {index}/{len(STOCKS)} {symbol} {name}: "
            f"{len(frame)} rows {frame.date.iloc[0]}..{frame.date.iloc[-1]} source={source}",
            flush=True,
        )
        if index < len(STOCKS):
            time.sleep(max(0.0, pause_seconds))

    complete = len(status) == len(STOCKS) and all(x["status"] == "ok" for x in status)
    providers = sorted({x["provider"] for x in status})
    manifest = {
        "source": providers,
        "dataset": "FinMind TaiwanStockPriceAdj and/or Yahoo Finance auto_adjust=True",
        "price_semantics": "back-adjusted OHLC for historical indicators and return calculations",
        "raw_execution_prices": "TaiwanStockPrice/raw exchange bars; retain separately for order execution and display",
        "volume_semantics": "provider-reported volume; Yahoo volume is not used to reconstruct raw turnover",
        "start_date_requested": start_date, "end_date_requested": end_date,
        "collected_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
        "token_configured": bool(token), "provider_mode": provider,
        "symbols_expected": len(STOCKS), "symbols_collected": len(status),
        "collection_complete": complete, "status_rows": status,
        "mixed_provider_run": len(providers) > 1,
        "note": (
            "Auto mode tries FinMind per symbol, then falls back to Yahoo on failure. "
            "Provider is recorded per symbol. Do not compare or concatenate runs without "
            "checking provider provenance. Yahoo auto_adjust=True adjusts OHLC for "
            "distributions/splits; turnover is intentionally blank."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not complete:
        raise RuntimeError("Adjusted-price collection incomplete")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/top10_adjusted")
    parser.add_argument("--start-date", default="2010-01-04")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument("--provider", choices=["auto", "finmind", "yahoo"], default="auto")
    args = parser.parse_args()
    manifest = collect(
        Path(args.output), args.start_date, args.end_date,
        args.pause_seconds, args.provider,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "status_rows"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# Research refresh trigger: 2026-11-12; data collection only, no strategy parameters changed.
