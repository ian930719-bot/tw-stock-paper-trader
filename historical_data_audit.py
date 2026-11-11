"""Offline audit of collected Taiwan OHLCV files and their provenance.

This utility checks file presence, hashes, row validity and candidate coverage.
It deliberately does not claim exchange reconciliation, delisted-price
completeness, or point-in-time-universe completeness without authoritative inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLUMNS = ("open", "high", "low", "close")
REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"symbol": str, "stock_id": str})


def audit_dataset(input_dir: str | Path) -> dict:
    root = Path(input_dir)
    manifest_path = root / "download_manifest.json"
    candidates_path = root / "candidate_symbols.csv"
    bars_dir = root / "bars"
    if not manifest_path.exists() or not candidates_path.exists() or not bars_dir.is_dir():
        raise FileNotFoundError("Audit requires download_manifest.json, candidate_symbols.csv and bars/")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = _read_csv(candidates_path)
    symbol_col = "symbol" if "symbol" in candidates else "stock_id"
    if symbol_col not in candidates:
        raise ValueError("candidate_symbols.csv must contain symbol or stock_id")
    candidates[symbol_col] = candidates[symbol_col].astype(str).str.strip()
    expected_hashes = manifest.get("data_files_sha256") or {}
    status_rows = {
        str(row.get("symbol", "")).strip(): row
        for row in manifest.get("status_rows", [])
        if row.get("symbol")
    }
    coverage_rows, quality_rows, integrity_rows = [], [], []
    valid_total = invalid_total = duplicate_total = 0
    successful = empty = failed = missing_files = 0
    date_min = date_max = None

    for candidate in candidates.to_dict("records"):
        symbol = str(candidate[symbol_col]).strip()
        filename = f"{symbol}.csv"
        path = bars_dir / filename
        source_row = status_rows.get(symbol, {})
        file_exists = path.is_file()
        raw = pd.DataFrame()
        parse_error = ""
        if file_exists:
            try:
                raw = _read_csv(path)
            except Exception as exc:
                parse_error = f"{type(exc).__name__}: {exc}"
        else:
            missing_files += 1

        row_count = len(raw)
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
        if row_count and not missing_columns:
            dates = pd.to_datetime(raw["date"], errors="coerce")
            numeric = raw[list(PRICE_COLUMNS) + ["volume"]].apply(pd.to_numeric, errors="coerce")
            finite = pd.Series(np.isfinite(numeric.to_numpy()).all(axis=1), index=raw.index)
            valid = dates.notna() & numeric.notna().all(axis=1) & finite
            valid &= (numeric[list(PRICE_COLUMNS)] > 0).all(axis=1)
            valid &= numeric["volume"] >= 0
            valid &= numeric["high"] >= numeric[["open", "close", "low"]].max(axis=1)
            valid &= numeric["low"] <= numeric[["open", "close", "high"]].min(axis=1)
            duplicate_mask = dates.notna() & dates.duplicated(keep="last")
            valid_unique = valid & ~duplicate_mask
            effective_rows = int(valid_unique.sum())
            invalid_rows = int((~valid).sum())
            duplicate_rows = int((valid & duplicate_mask).sum())
            first_date = dates.loc[valid_unique].min()
            last_date = dates.loc[valid_unique].max()
            first_date = first_date.date().isoformat() if pd.notna(first_date) else ""
            last_date = last_date.date().isoformat() if pd.notna(last_date) else ""
        else:
            effective_rows = invalid_rows = duplicate_rows = 0
            first_date = last_date = ""
            if row_count and missing_columns:
                invalid_rows = row_count
                parse_error = "missing required columns: " + ",".join(missing_columns)

        valid_total += effective_rows
        invalid_total += invalid_rows
        duplicate_total += duplicate_rows
        if first_date:
            date_min = min(date_min, first_date) if date_min else first_date
        if last_date:
            date_max = max(date_max, last_date) if date_max else last_date

        expected = expected_hashes.get(filename)
        actual = _sha256(path) if file_exists else ""
        hash_match = bool(expected and actual and expected == actual)
        if expected:
            integrity_rows.append({
                "file": f"bars/{filename}", "expected_sha256": expected,
                "actual_sha256": actual, "hash_match": hash_match,
            })
        source_status = str(source_row.get("status", "not_recorded"))
        if parse_error or not file_exists:
            candidate_status = "failed"
            failed += 1
        elif effective_rows:
            candidate_status = "success"
            successful += 1
        else:
            candidate_status = "empty"
            empty += 1
        coverage_rows.append({
            "symbol": symbol,
            "candidate_source": candidate.get("candidate_source", ""),
            "delisting_date": candidate.get("delisting_date", ""),
            "status": candidate_status,
            "manifest_status": source_status,
            "file_exists": file_exists,
            "raw_rows": row_count,
            "valid_ohlcv_rows": effective_rows,
            "first_date": first_date,
            "last_date": last_date,
            "expected_sha256": expected or "",
            "actual_sha256": actual,
            "sha256_match": hash_match if expected else False,
            "error": parse_error,
        })
        quality_rows.append({
            "symbol": symbol, "raw_rows": row_count,
            "valid_ohlcv_rows": effective_rows, "invalid_rows": invalid_rows,
            "duplicate_valid_dates": duplicate_rows,
            "missing_required_columns": ";".join(missing_columns),
            "parse_error": parse_error,
        })

    # Include frozen universe and immutable daily snapshots in integrity evidence.
    extra_hashes = []
    frozen = root / "candidate_universe_full.csv"
    if frozen.exists() and manifest.get("candidate_universe_sha256"):
        extra_hashes.append(("candidate_universe_full.csv", manifest["candidate_universe_sha256"]))
    snapshot = manifest.get("daily_universe_snapshot") or {}
    snapshot_path = snapshot.get("snapshot_path")
    if snapshot_path and snapshot.get("snapshot_sha256"):
        extra_hashes.append((snapshot_path, snapshot["snapshot_sha256"]))
    for relative, expected in extra_hashes:
        path = root / relative
        actual = _sha256(path) if path.is_file() else ""
        integrity_rows.append({
            "file": relative, "expected_sha256": expected,
            "actual_sha256": actual, "hash_match": bool(actual and actual == expected),
        })

    coverage = pd.DataFrame(coverage_rows)
    quality = pd.DataFrame(quality_rows)
    integrity = pd.DataFrame(integrity_rows)
    coverage.to_csv(root / "candidate_coverage.csv", index=False)
    quality.to_csv(root / "data_quality.csv", index=False)
    integrity.to_csv(root / "manifest_integrity.csv", index=False)

    delisted_mask = coverage["candidate_source"].astype(str).str.contains("delisted", case=False, na=False)
    delisted_total = int(delisted_mask.sum())
    delisted_with_bars = int((delisted_mask & coverage["valid_ohlcv_rows"].gt(0)).sum())
    action_manifest_path = root / "corporate_actions" / "manifest.json"
    action_manifest = (
        json.loads(action_manifest_path.read_text(encoding="utf-8"))
        if action_manifest_path.exists() else {}
    )
    hash_mismatches = int((~integrity["hash_match"].astype(bool)).sum()) if not integrity.empty else 0
    hash_missing = int((integrity["actual_sha256"].astype(str) == "").sum()) if not integrity.empty else 0
    summary = {
        "candidate_count": int(len(candidates)),
        "successful_symbols": successful, "empty_symbols": empty, "failed_symbols": failed,
        "missing_price_files": missing_files, "valid_ohlcv_rows": valid_total,
        "invalid_ohlcv_rows": invalid_total, "duplicate_valid_date_rows": duplicate_total,
        "data_date_start": date_min, "data_date_end": date_max,
        "hash_files_checked": int(len(integrity)),
        "hash_mismatches": hash_mismatches, "hash_missing_files": hash_missing,
        "delisted_candidate_count": delisted_total,
        "delisted_candidates_with_price_bars": delisted_with_bars,
        "delisted_price_coverage_verified": False,
        "point_in_time_universe_complete": False,
        "official_exchange_price_reconciled": False,
        "official_trading_calendar_reconciled": False,
        "corporate_actions_complete": bool(action_manifest.get("corporate_actions_complete", False)),
        "corporate_action_global_dataset_status": action_manifest.get("global_dataset_status", {}),
        "corporate_action_blocked_symbols": int(action_manifest.get("blocked_symbols", 0)),
        "research_grade_eligible": False,
        "research_grade_reason": "Official TWSE/TPEx reconciliation, complete point-in-time universe, and delisted-price verification are not established by this offline audit.",
    }
    (root / "data_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/taiwan_history")
    args = parser.parse_args()
    summary = audit_dataset(args.input)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
