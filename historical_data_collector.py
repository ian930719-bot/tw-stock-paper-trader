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


def _api_get(params: dict, *, token: str = "", timeout: int = 20,
             retries: int = 3, pause_seconds: float = 2.0) -> list[dict]:
    query = dict(params)
    if token:
        query["token"] = token
    error = None
    for attempt in range(retries):
        try:
            response = requests.get(API_URL, params=query, timeout=(8, timeout))
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
                delay = min(20, 2 ** (attempt + 1) * pause_seconds)
                print(f"[API RETRY] attempt={attempt + 2}/{retries} delay={delay:.1f}s error={type(exc).__name__}: {exc}", flush=True)
                time.sleep(delay)
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


def build_candidate_universe(current: pd.DataFrame, delisted: pd.DataFrame) -> pd.DataFrame:
    """Build a stock-only candidate list, excluding ETFs, bonds and warrants.

    Current common equities and historical delisting notices are candidates,
    not proof of complete point-in-time membership or delisted-price coverage.
    """
    id_pattern = r"^[1-9][0-9]{3}(?:-[A-Z]+)?$"
    current = current.copy()
    for col in ("stock_name", "industry_category", "type"):
        if col not in current:
            current[col] = ""
    industry = current["industry_category"].astype(str)
    current_equities = current[
        current["stock_id"].astype(str).str.fullmatch(id_pattern, na=False)
        & ~industry.str.contains(
            r"ETF|指數股票型基金|債|受益證券|權證|認購|認售|牛熊", case=False, regex=True
        )
    ][["stock_id", "stock_name", "industry_category", "type"]].copy()
    current_equities["candidate_source"] = "current_common_equity"
    current_equities["delisting_date"] = ""

    delisted_equities = pd.DataFrame(columns=[
        "stock_id", "stock_name", "industry_category", "type", "candidate_source", "delisting_date"
    ])
    if not delisted.empty and "stock_id" in delisted:
        old = delisted.copy()
        for col in ("stock_name", "industry_category", "type"):
            if col not in old:
                old[col] = ""
        old["stock_id"] = old["stock_id"].astype(str).str.strip()
        old = old[old["stock_id"].str.fullmatch(id_pattern, na=False)]
        delisted_equities = old[["stock_id", "stock_name", "industry_category", "type"]].copy()
        delisted_equities["candidate_source"] = "delisted_common_equity_candidate"
        delisted_equities["delisting_date"] = old["date"].astype(str) if "date" in old else ""

    candidates = pd.concat([current_equities, delisted_equities], ignore_index=True)
    candidates = candidates.drop_duplicates("stock_id", keep="first")
    return candidates.sort_values(["candidate_source", "stock_id"]).reset_index(drop=True)


def select_candidate_metadata(candidates: pd.DataFrame, max_symbols: int | None,
                              *, seed: int = 20261109) -> pd.DataFrame:
    """Select a deterministic, industry-stratified smoke-test sample.

    Reserve up to 20% for delisted candidates, then round-robin current names
    across industry groups. Full-universe collection passes max_symbols=None.
    """
    if max_symbols is None or len(candidates) <= max_symbols:
        return candidates.reset_index(drop=True)
    if max_symbols <= 0:
        return candidates.iloc[0:0].copy()
    z = candidates.copy()
    old = z[z["candidate_source"] == "delisted_common_equity_candidate"]
    current = z[z["candidate_source"] == "current_common_equity"]
    old_n = min(len(old), max(1, int(round(max_symbols * 0.20)))) if len(old) and max_symbols >= 5 else 0
    current_n = max_symbols - old_n
    selected_current = []
    groups = []
    for _, group in current.groupby("industry_category", dropna=False, sort=True):
        shuffled = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        groups.append(shuffled)
    while len(selected_current) < current_n and any(len(g) for g in groups):
        for group in groups:
            if len(selected_current) >= current_n:
                break
            if len(group):
                selected_current.append(group.iloc[0])
                group.drop(index=0, inplace=True)
                group.reset_index(drop=True, inplace=True)
    selected_old = old.sample(n=old_n, random_state=seed) if old_n else old.iloc[0:0]
    if selected_current:
        current_pick = pd.DataFrame(selected_current)
    else:
        current_pick = current.iloc[0:0]
    selected = pd.concat([current_pick, selected_old], ignore_index=True)
    return selected.sort_values(["candidate_source", "industry_category", "stock_id"]).reset_index(drop=True)


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
                    max_symbols: int | None = None,
                    max_runtime_minutes: float | None = None) -> dict:
    """Resume per-symbol downloads and write raw CSV plus provenance manifest."""
    api_token = (token or os.getenv("FINMIND_TOKEN", "")).strip()
    if not api_token:
        # FinMind free-tier requests are rate-limited; keep a conservative
        # inter-symbol delay while still bounding each failed request.
        pause_seconds = max(float(pause_seconds), 12.1)
    else:
        pause_seconds = max(float(pause_seconds), 2.1)
    out = Path(output_dir)
    bars_dir = out / "bars"
    bars_dir.mkdir(parents=True, exist_ok=True)
    frozen_universe_path = out / "candidate_universe_full.csv"
    if max_symbols is None and frozen_universe_path.exists():
        # Reuse the exact initial full-universe snapshot across chained batches.
        candidate_meta = pd.read_csv(frozen_universe_path, dtype={"stock_id": str})
    else:
        current, delisted = discover_symbols(start_date, end_date, token=api_token)
        candidate_meta = build_candidate_universe(current, delisted)
        current.to_csv(out / "current_security_master.csv", index=False)
        delisted.to_csv(out / "delisting_announcements.csv", index=False)
        if max_symbols is None:
            candidate_meta.to_csv(frozen_universe_path, index=False)
    candidate_meta = select_candidate_metadata(candidate_meta, max_symbols)
    symbols = candidate_meta["stock_id"].astype(str).tolist()
    candidate_meta.rename(columns={"stock_id": "symbol", "stock_name": "name"}).to_csv(
        out / "candidate_symbols.csv", index=False
    )
    statuses = []

    def write_checkpoint() -> None:
        checkpoint = {
            "source": "FinMind API v4",
            "price_dataset": "TaiwanStockPrice (raw, unadjusted execution bars)",
            "start_date": start_date, "end_date": end_date,
            "token_configured": bool(api_token),
            "requested_symbols": len(symbols),
            "candidate_universe_frozen": max_symbols is None and frozen_universe_path.exists(),
            "candidate_universe_sha256": hashlib.sha256(frozen_universe_path.read_bytes()).hexdigest() if max_symbols is None and frozen_universe_path.exists() else None,
            "completed_symbols": sum(x["status"] in {"downloaded", "cached", "empty"} for x in statuses),
            "attempted_symbols": len(statuses),
            "remaining_symbols": max(0, len(symbols) - sum(x["status"] in {"downloaded", "cached", "empty"} for x in statuses)),
            "collection_complete": (
                sum(x["status"] in {"downloaded", "cached", "empty"} for x in statuses) >= len(symbols)
                and not any(x["status"] == "failed" for x in statuses)
            ),
            "successful_symbols": sum(x["status"] in {"downloaded", "cached"} for x in statuses),
            "failed_symbols": sum(x["status"] == "failed" for x in statuses),
            "empty_symbols": sum(x["status"] == "empty" for x in statuses),
            "status_rows": statuses,
            "data_files_sha256": {},
            "checkpoint_only": True,
            "point_in_time_universe_complete": False,
            "delisted_price_coverage_verified": False,
            "note": "Checkpoint written during collection; hashes are finalized only after the run completes. Candidate symbols are not a verified historical point-in-time universe.",
        }
        (out / "download_manifest.json").write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    collection_started = time.monotonic()
    print(
        f"[COLLECT START] candidates={len(symbols)} cached_files={len(list(bars_dir.glob('*.csv')))} "
        f"token_configured={bool(api_token)} pause_seconds={pause_seconds:.1f}",
        flush=True,
    )
    for i, symbol in enumerate(symbols):
        if max_runtime_minutes is not None and (time.monotonic() - collection_started) >= max(0.0, float(max_runtime_minutes)) * 60:
            break
        path = bars_dir / f"{symbol}.csv"
        if path.exists() and path.stat().st_size > 20:
            cached_rows = int(len(pd.read_csv(path)))
            cached_status = "cached" if cached_rows > 0 else "empty"
            statuses.append({"symbol": symbol, "status": cached_status, "rows": cached_rows, "cache_hit": True})
            write_checkpoint()
            done = sum(x["status"] in {"downloaded", "cached", "empty"} for x in statuses)
            if done % 10 == 0 or done == len(symbols):
                print(f"[COLLECT PROGRESS] {done}/{len(symbols)} successful_or_empty; cached={sum(x['status']=='cached' for x in statuses)} empty={sum(x['status']=='empty' for x in statuses)} failed={sum(x['status']=='failed' for x in statuses)} current={symbol}", flush=True)
            continue
        params = {"dataset": "TaiwanStockPrice", "data_id": symbol, "start_date": start_date}
        if end_date:
            params["end_date"] = end_date
        try:
            rows = _api_get(params, token=api_token)
            frame = _normalize_bars(rows, symbol)
            frame.to_csv(path, index=False)
            status = "downloaded" if not frame.empty else "empty"
            statuses.append({"symbol": symbol, "status": status, "rows": int(len(frame))})
        except Exception as exc:
            statuses.append({"symbol": symbol, "status": "failed", "error": str(exc)[:500]})
        write_checkpoint()
        done = sum(x["status"] in {"downloaded", "cached", "empty"} for x in statuses)
        failed = sum(x["status"] == "failed" for x in statuses)
        print(
            f"[COLLECT PROGRESS] attempted={len(statuses)}/{len(symbols)} "
            f"successful_or_empty={done} failed={failed} current={symbol} "
            f"status={statuses[-1]['status']}",
            flush=True,
        )
        if i + 1 < len(symbols):
            time.sleep(max(0.0, float(pause_seconds)))
    processed_count = sum(x["status"] in {"downloaded", "cached", "empty"} for x in statuses)
    failed_count = sum(x["status"] == "failed" for x in statuses)
    manifest = {
        "source": "FinMind API v4",
        "price_dataset": "TaiwanStockPrice (raw, unadjusted execution bars)",
        "start_date": start_date, "end_date": end_date,
        "token_configured": bool(api_token),
        "requested_symbols": len(symbols),
        "candidate_universe_frozen": max_symbols is None and frozen_universe_path.exists(),
        "candidate_universe_sha256": hashlib.sha256(frozen_universe_path.read_bytes()).hexdigest() if max_symbols is None and frozen_universe_path.exists() else None,
        "completed_symbols": processed_count,
        "attempted_symbols": len(statuses),
        "remaining_symbols": max(0, len(symbols) - processed_count),
        "collection_complete": processed_count >= len(symbols) and failed_count == 0,
        "runtime_seconds": round(time.monotonic() - collection_started, 2),
        "successful_symbols": sum(x["status"] in {"downloaded", "cached"} for x in statuses),
        "failed_symbols": failed_count,
        "empty_symbols": sum(x["status"] == "empty" for x in statuses),
        "status_rows": statuses,
        "point_in_time_universe_complete": False,
        "delisted_price_coverage_verified": False,
        "checkpoint_only": False,
        "note": "Current master plus delisting announcements is a candidate list, not a verified historical point-in-time universe. Reconcile coverage before research-grade use.",
    }
    manifest["data_files_sha256"] = {}
    for path in sorted(bars_dir.glob("*.csv")):
        manifest["data_files_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[COLLECT END] complete={manifest['collection_complete']} "
        f"successful={manifest['successful_symbols']} empty={manifest['empty_symbols']} "
        f"failed={manifest['failed_symbols']} remaining={manifest['remaining_symbols']} "
        f"runtime_seconds={manifest['runtime_seconds']:.1f}",
        flush=True,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/taiwan_history")
    parser.add_argument("--start-date", default="2010-01-04")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--pause-seconds", type=float, default=6.1)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--max-runtime-minutes", type=float, default=None,
                        help="Stop cleanly after this many minutes so a later run can resume")
    args = parser.parse_args()
    result = collect_history(args.output, args.start_date, args.end_date,
                             pause_seconds=args.pause_seconds, max_symbols=args.max_symbols,
                             max_runtime_minutes=args.max_runtime_minutes)
    print(json.dumps({k: v for k, v in result.items() if k != "status_rows"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
