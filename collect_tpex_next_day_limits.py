"""Collect TPEx official next-session price limits for historical top-ten symbols.

The TPEx afterTrading/otc endpoint publishes next-day limit-up/down values.
Those official values are retained separately and later joined to raw bars.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import random
import re
import time
from pathlib import Path
import pandas as pd
import requests

URL = "https://www.tpex.org.tw/www/zh-tw/afterTrading/otc"

def clean(value):
    return re.sub(r"\\s+", "", html.unescape(re.sub(r"<[^>]*>", "", str(value or ""))))

def number(value):
    s = clean(value).replace(",", "").replace("−", "-").replace("－", "-")
    if s in {"", "--", "---", "----", "X", "nan", "None"}:
        return None
    try:
        x = float(s)
        return x if x > 0 else None
    except (TypeError, ValueError):
        return None

def fetch_day(session, day, targets, retries=8):
    params = {"date": pd.Timestamp(day).strftime("%Y/%m/%d"), "type": "EW", "response": "json"}
    last_error = None
    for attempt in range(retries):
        try:
            response = session.get(
                URL, params=params, timeout=(10, 45),
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
            )
            response.raise_for_status()
            payload = response.json()
            table = None
            for candidate in payload.get("tables") or []:
                fields = [clean(x) for x in candidate.get("fields", [])]
                if any(x in fields for x in ("代號", "股票代號", "證券代號")):
                    table = candidate
                    break
            if table is None and isinstance(payload.get("fields"), list):
                table = {"fields": payload["fields"], "data": payload.get("data", [])}
            if table is None:
                raise RuntimeError(f"TPEx quote table missing for {day}: {payload.get('stat')}")
            fields = [clean(x) for x in table.get("fields", [])]
            def index(*names):
                for name in names:
                    if clean(name) in fields:
                        return fields.index(clean(name))
                raise RuntimeError(f"Missing official TPEx field {names}; got {fields}")
            code_i = index("代號", "股票代號", "證券代號")
            up_i = index("次日漲停價")
            down_i = index("次日跌停價")
            out = {}
            for row in table.get("data", []):
                if len(row) <= max(code_i, up_i, down_i):
                    continue
                symbol = clean(row[code_i])
                if symbol not in targets:
                    continue
                key = symbol
                if key in out:
                    raise RuntimeError(f"Duplicate TPEx code {symbol} on {day}")
                out[key] = {
                    "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                    "stock_id": symbol,
                    "next_day_limit_up": number(row[up_i]),
                    "next_day_limit_down": number(row[down_i]),
                    "source": "TPEx official afterTrading/otc next-day limit fields",
                    "source_url": response.url,
                }
            return out
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(45, 2 ** attempt) + random.uniform(0.25, 1.25))
    raise RuntimeError(f"TPEx {day}: {last_error}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-dir", required=True)
    parser.add_argument("--universe-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pause", type=float, default=0.30)
    args = parser.parse_args()
    daily_dir = Path(args.daily_dir)
    universe = pd.read_csv(args.universe_file, dtype={"symbol": str, "stock_id": str})
    target_ids = set(universe.loc[universe.market.astype(str).str.upper() == "TPEX", "stock_id"].astype(str))
    if not target_ids:
        raise RuntimeError("No TPEx symbols found in point-in-time universe")
    daily_files = sorted(daily_dir.glob("*.csv"))
    if not daily_files:
        raise RuntimeError(f"No daily official market-wide files in {daily_dir}")
    session = requests.Session()
    records = []
    missing_symbols = []
    queried_dates = []
    for i, path in enumerate(daily_files, 1):
        day = path.stem
        market = pd.read_csv(path, dtype={"stock_id": str})
        if "market" not in market or "stock_id" not in market:
            raise RuntimeError(f"{path} missing market/stock_id columns")
        expected = set(market.loc[
            (market.market.astype(str).str.upper() == "TPEX")
            & market.stock_id.astype(str).isin(target_ids), "stock_id"
        ].astype(str))
        if not expected:
            continue
        rows = fetch_day(session, day, expected)
        queried_dates.append(day)
        for symbol in sorted(expected):
            record = rows.get(symbol)
            if record is None:
                missing_symbols.append({"date": day, "stock_id": symbol, "reason": "symbol missing from official TPEx response"})
                records.append({
                    "date": day, "stock_id": symbol, "next_day_limit_up": None,
                    "next_day_limit_down": None,
                    "source": "TPEx official afterTrading/otc; symbol row missing",
                    "source_url": None,
                })
            else:
                if record["next_day_limit_up"] is None or record["next_day_limit_down"] is None:
                    missing_symbols.append({"date": day, "stock_id": symbol, "reason": "official next-day limit field blank"})
                records.append(record)
        if i % 25 == 0 or i == len(daily_files):
            pd.DataFrame(records).to_csv(args.output, index=False)
            print(f"PROGRESS files={i}/{len(daily_files)} queried_dates={len(queried_dates)} rows={len(records)} missing={len(missing_symbols)}", flush=True)
        time.sleep(args.pause)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("No TPEx next-day limit rows collected")
    if frame.duplicated(["date", "stock_id"]).any():
        raise RuntimeError("Duplicate date/symbol rows in TPEx next-day limit output")
    frame.to_csv(out, index=False)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "source": "TPEx official afterTrading/otc",
        "fields": ["次日漲停價", "次日跌停價"],
        "first_source_date": str(frame.date.min()),
        "last_source_date": str(frame.date.max()),
        "queried_dates": len(queried_dates),
        "rows": len(frame),
        "symbols": int(frame.stock_id.nunique()),
        "missing_symbol_or_limit_rows": len(missing_symbols),
        "missing_examples": missing_symbols[:100],
        "sha256": digest,
        "complete": len(missing_symbols) == 0,
        "note": "Values are published on source_date for the next TPEx market session; no formula reconstruction is used."
    }
    out.with_name("manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
