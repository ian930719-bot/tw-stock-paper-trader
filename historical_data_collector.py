"""Resumable FinMind daily-history collector for Taiwan equities.

The collector builds a candidate symbol list from the current security master
plus historical delisting announcements, then downloads raw daily bars per
symbol. This improves coverage but does NOT prove complete point-in-time
universe membership; that remains an explicit research limitation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import numpy as np
import requests

API_URL = "https://api.finmindtrade.com/api/v4/data"


def _api_get(params: dict, *, token: str = "", timeout: int = 30,
             retries: int = 4, pause_seconds: float = 6.1) -> list[dict]:
    query = dict(params)
    if token:
        query["token"] = token
    error = None
    for attempt in range(retries):
        try:
            response = requests.get(API_URL, params=query, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or str(payload.get("status", "200")) not in {"200", "success", "Success"}:
                raise RuntimeError(str(payload.get("msg", "FinMind API status error")) if isinstance(payload, dict) else "Invalid API payload")
            rows = payload.get("data")
            if not isinstance(rows, list):
                raise RuntimeError("FinMind data field is not a list")
            return rows
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(min(60, 2 ** attempt * 3))
    raise RuntimeError(f"FinMind request failed: {error}") from error


def discover_symbols(start_date: str, end_date: str | None = None, *,
                     token: str = "", timeout: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return current security master and historical delisting symbol candidates."""
    current = pd.DataFrame(_api_get({"dataset": "TaiwanStockInfo"}, token=token, timeout=timeout))
    if current.empty or not {"stock_id", "type"}.issubset(current.columns):
        raise RuntimeError("TaiwanStockInfo did not return a usable security master")
    current = current.copy()
    current["stock_id"] = current["stock_id"].astype(str).str.strip()
    current = current[current["type"].astype(str).str.lower().isin({"twse", "tpex"})]
    params = {"dataset": "TaiwanStockDelisting", "start_date": start_date}
    if end_date:
        params["end_date"] = end_date
    delisted = pd.DataFrame(_api_get(params, token=token, timeout=timeout))
    if not delisted.empty and "stock_id" in delisted:
        delisted["stock_id"] = delisted["stock_id"].astype(str).str.strip()
    return current, delisted


def _normalize_bars(rows: list[dict], symbol: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume", "turnover"])
    raw = pd.DataFrame(rows).rename(columns={
        "max": "high", "min": "low", "Trading_Volume": "volume",
        "Trading_money": "turnover",
    })
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(raw.columns):
        raise RuntimeError(f"{symbol}: missing columns {sorted(required - set(raw.columns))}")
    for col in ("open", "high", "low", "close", "volume"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    if "turnover" not in raw:
        raw["turnover"] = np.nan
    else:
        raw["turnover"] = pd.to_numeric(raw["turnover"], errors="coerce")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    raw = raw.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    valid = (
        (raw[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (raw["volume"] >= 0)
        & (raw["high"] >= raw[["open", "close", "low"]].max(axis=1))
        & (raw["low"] <= raw[["open", "close", "high"]].min(axis=1))
    )
    raw = raw.loc[valid].drop_duplicates("date", keep="last").sort_values("date")
    raw.insert(0, "symbol", str(symbol))
    return raw[["symbol", "date", "open", "high", "low", "close", "volume", "turnover"]].reset_index(drop=True)


def collect_history(output_dir: str, start_date: str, end_date: str | None = None, *,
                    token: str | None = None, pause_seconds: float = 6.1,
                    max_symbols: int | None = None) -> dict:
    """Resume per-symbol downloads and write raw CSV plus provenance manifest."""
    api_token = (token or os.getenv("FINMIND_TOKEN", "")).strip()
    if not api_token:
        # Free-tier limits are lower; use a conservative interval.
        pause_seconds = max(float(pause_seconds), 12.1)
    out = Path(output_dir)
    bars_dir = out / "bars"
    bars_dir.mkdir(parents=True, exist_ok=True)
    current, delisted = discover_symbols(start_date, end_date, token=api_token)
    symbols = sorted(set(current["stock_id"].astype(str)) |
                     (set(delisted["stock_id"].astype(str)) if not delisted.empty and "stock_id" in delisted else set()))
    if max_symbols is not None:
        symbols = symbols[:max(0, int(max_symbols))]
    pd.DataFrame({"symbol": symbols}).to_csv(out / "candidate_symbols.csv", index=False)
    current.to_csv(out / "current_security_master.csv", index=False)
    delisted.to_csv(out / "delisting_announcements.csv", index=False)
    statuses = []
    for i, symbol in enumerate(symbols):
        path = bars_dir / f"{symbol}.csv"
        if path.exists() and path.stat().st_size > 20:
            statuses.append({"symbol": symbol, "status": "cached", "rows": int(len(pd.read_csv(path)))})
            continue
        params = {"dataset": "TaiwanStockPrice", "data_id": symbol, "start_date": start_date}
        if end_date:
            params["end_date"] = end_date
        try:
            rows = _api_get(params, token=api_token)
            frame = _normalize_bars(rows, symbol)
            frame.to_csv(path, index=False)
            statuses.append({"symbol": symbol, "status": "downloaded" if not frame.empty else "empty",
                             "rows": int(len(frame))})
        except Exception as exc:
            statuses.append({"symbol": symbol, "status": "failed", "error": str(exc)})
        if i + 1 < len(symbols):
            time.sleep(max(0.0, float(pause_seconds)))
        manifest = {
            "source": "FinMind API v4",
            "price_dataset": "TaiwanStockPrice (raw, unadjusted execution bars)",
            "start_date": start_date, "end_date": end_date,
            "token_configured": bool(api_token),
            "requested_symbols": len(symbols),
            "completed_symbols": len(statuses),
            "successful_symbols": sum(x["status"] in {"downloaded", "cached"} for x in statuses),
            "failed_symbols": sum(x["status"] == "failed" for x in statuses),
            "empty_symbols": sum(x["status"] == "empty" for x in statuses),
            "status_rows": statuses,
            "point_in_time_universe_complete": False,
            "delisted_price_coverage_verified": False,
            "note": "Current master plus delisting announcements is a candidate list, not a verified historical point-in-time universe. Reconcile coverage before research-grade use.",
        }
        (out / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["data_files_sha256"] = {}
    for path in sorted(bars_dir.glob("*.csv")):
        manifest["data_files_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/taiwan_history")
    parser.add_argument("--start-date", default="2010-01-04")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--pause-seconds", type=float, default=6.1)
    parser.add_argument("--max-symbols", type=int, default=None)
    args = parser.parse_args()
    result = collect_history(args.output, args.start_date, args.end_date,
                             pause_seconds=args.pause_seconds, max_symbols=args.max_symbols)
    print(json.dumps({k: v for k, v in result.items() if k != "status_rows"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import numpy as np
    main()
