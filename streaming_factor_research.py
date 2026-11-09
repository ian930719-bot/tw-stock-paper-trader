"""Memory-bounded historical factor screening for Taiwan daily equity data.

Processes one symbol at a time, writes compact year partitions, then computes
cross-sectional daily IC and quintile spreads. Outputs are exploratory gross
return diagnostics, not portfolio backtest results or proof of tradability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from corporate_action_adjustments import apply_back_adjustments

FACTOR_FAMILIES = {
    "momentum": ["ret_5", "ret_20", "ret_60", "ret_120"],
    "reversal": ["reversal_1", "reversal_5", "distance_sma_5", "distance_sma_20"],
    "trend": ["distance_sma_50", "distance_sma_100", "distance_sma_200", "sma50_slope_20", "trend_stack"],
    "breakout": ["breakout_20", "breakout_55", "distance_high_20", "distance_high_55"],
    "volume": ["relative_volume_20", "volume_trend_20", "turnover_shock_20"],
    "volatility": ["atr_pct_14", "realized_vol_20", "realized_vol_60", "volatility_compression"],
    "gap_range": ["overnight_gap", "intraday_range_pct", "close_location"],
}
FACTORS = [f for values in FACTOR_FAMILIES.values() for f in values]
HORIZONS = (5, 20, 60)
RAW_REQUIRED = {"date", "open", "high", "low", "close", "volume"}


def build_symbol_factors(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    missing = RAW_REQUIRED.difference(raw.columns)
    if missing:
        raise ValueError(f"{symbol}: missing columns {sorted(missing)}")
    z = raw.copy()
    z["date"] = pd.to_datetime(z["date"], errors="coerce").dt.normalize()
    for col in ("open", "high", "low", "close", "volume", "turnover"):
        if col in z:
            z[col] = pd.to_numeric(z[col], errors="coerce")
    z = z.dropna(subset=list(RAW_REQUIRED)).sort_values("date")
    valid = (
        (z[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (z["volume"] >= 0)
        & (z["high"] >= z[["open", "close", "low"]].max(axis=1))
        & (z["low"] <= z[["open", "close", "high"]].min(axis=1))
    )
    z = z.loc[valid].drop_duplicates("date", keep="last").copy()
    if z.empty:
        return pd.DataFrame(columns=["date", "symbol", *FACTORS, *[f"fwd_{h}d_next_open" for h in HORIZONS]])
    adjusted = all(f"adj_{col}" in z.columns for col in ("open", "high", "low", "close"))
    prefix = "adj_" if adjusted else ""
    close, open_, high, low, volume = (
        z[f"{prefix}close"], z[f"{prefix}open"], z[f"{prefix}high"],
        z[f"{prefix}low"], z["volume"]
    )
    prev_close = close.shift(1)
    z["symbol"] = str(symbol)
    for n in (5, 20, 60, 120):
        z[f"ret_{n}"] = close.pct_change(n)
    z["reversal_1"] = -close.pct_change(1)
    z["reversal_5"] = -close.pct_change(5)
    for n in (5, 20, 50, 100, 200):
        sma = close.rolling(n, min_periods=n).mean()
        z[f"distance_sma_{n}"] = close / sma - 1
    sma50 = close.rolling(50, min_periods=50).mean()
    sma100 = close.rolling(100, min_periods=100).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    z["sma50_slope_20"] = sma50.pct_change(20)
    z["trend_stack"] = ((close > sma50) & (sma50 > sma100) & (sma100 > sma200)).astype(float).where(sma200.notna())
    for n in (20, 55):
        prior_high = high.shift(1).rolling(n, min_periods=n).max()
        z[f"breakout_{n}"] = (close > prior_high).astype(float).where(prior_high.notna())
        z[f"distance_high_{n}"] = close / prior_high - 1
    avg_vol = volume.shift(1).rolling(20, min_periods=20).mean()
    z["relative_volume_20"] = volume / avg_vol.replace(0, np.nan)
    z["volume_trend_20"] = volume.rolling(5, min_periods=5).mean() / volume.rolling(20, min_periods=20).mean().replace(0, np.nan)
    if "corporate_action_window" in z:
        action_window = z["corporate_action_window"].fillna(False).astype(bool)
        z.loc[action_window, ["relative_volume_20", "volume_trend_20"]] = np.nan
    money = pd.to_numeric(z["turnover"], errors="coerce") if "turnover" in z else close * volume
    z["turnover_shock_20"] = money / money.shift(1).rolling(20, min_periods=20).mean().replace(0, np.nan)
    if "corporate_action_window" in z and "turnover" not in z:
        z.loc[z["corporate_action_window"].fillna(False).astype(bool), "turnover_shock_20"] = np.nan
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    z["atr_pct_14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / close
    daily_ret = close.pct_change()
    z["realized_vol_20"] = daily_ret.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252)
    z["realized_vol_60"] = daily_ret.rolling(60, min_periods=60).std(ddof=0) * np.sqrt(252)
    z["volatility_compression"] = z["realized_vol_20"] / z["realized_vol_60"].replace(0, np.nan)
    z["overnight_gap"] = open_ / prev_close - 1
    z["intraday_range_pct"] = (high-low) / prev_close
    z["close_location"] = (close-low) / (high-low).replace(0, np.nan)
    next_open = open_.shift(-1)
    for h in HORIZONS:
        z[f"fwd_{h}d_next_open"] = close.shift(-h) / next_open - 1
    cols = ["date", "symbol", *FACTORS, *[f"fwd_{h}d_next_open" for h in HORIZONS]]
    return z[cols].replace([np.inf, -np.inf], np.nan)


def hac_t_stat(values: list[float] | np.ndarray, max_lag: int) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    centered = x - x.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    long_run = gamma0
    for lag in range(1, min(max_lag, n - 1) + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run += 2.0 * weight * gamma
    variance_mean = long_run / n
    if variance_mean <= 0 or not np.isfinite(variance_mean):
        return float("nan")
    return float(x.mean() / math.sqrt(variance_mean))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out.tolist()
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    restored = np.empty_like(q)
    restored[order] = np.clip(q, 0, 1)
    out[np.where(valid)[0]] = restored
    return out.tolist()


def _normal_two_sided_p(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return float("nan")
    return math.erfc(abs(t_stat) / math.sqrt(2.0))


def run_streaming_research(input_dir: str, output_dir: str, *, split_date: str = "2024-01-01") -> dict:
    root, out = Path(input_dir), Path(output_dir)
    bars = root / "bars"
    files = sorted(bars.glob("*.csv"))
    if not files:
        raise ValueError(f"No per-symbol bar CSV files in {bars}")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    expected_hashes = manifest.get("data_files_sha256", {})
    action_events_path = root / "corporate_actions" / "events.csv"
    action_events = pd.read_csv(action_events_path, dtype={"symbol": str}) if action_events_path.exists() else pd.DataFrame()
    corporate_actions_complete = bool(manifest.get("corporate_actions_complete", False))
    corporate_action_conflicts = int(manifest.get("corporate_action_conflicts", 0))
    quality, integrity = [], []
    split = pd.Timestamp(split_date)
    with tempfile.TemporaryDirectory(prefix="tw_factor_") as tmp:
        tmpdir = Path(tmp)
        written_years: set[int] = set()
        total_rows = 0
        for path in files:
            raw = pd.read_csv(path)
            symbol = path.stem
            if not action_events.empty and {"symbol", "date", "adjustment_ratio"}.issubset(action_events.columns):
                symbol_events = action_events.loc[action_events["symbol"].astype(str) == str(symbol)]
                if not symbol_events.empty:
                    raw = apply_back_adjustments(raw, symbol_events)
            raw_dates = pd.to_datetime(raw.get("date"), errors="coerce")
            valid_date = raw_dates.notna()
            numeric = raw.copy()
            for col in ("open", "high", "low", "close", "volume"):
                numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
            required_missing = int(numeric[["open", "high", "low", "close", "volume"]].isna().any(axis=1).sum()) if len(numeric) else 0
            duplicate_count = int(raw_dates[valid_date].duplicated().sum())
            price_ok = (numeric[["open", "high", "low", "close"]] > 0).all(axis=1)
            vol_ok = numeric["volume"] >= 0
            ohlc_ok = (
                (numeric["high"] >= numeric[["open", "close", "low"]].max(axis=1))
                & (numeric["low"] <= numeric[["open", "close", "high"]].min(axis=1))
            )
            usable_ohlc = valid_date & price_ok & vol_ok & ohlc_ok
            usable_ohlc_count = int(usable_ohlc.sum()) if len(numeric) else 0
            invalid_count = int((~(price_ok & vol_ok & ohlc_ok)).sum()) if len(numeric) else 0
            clean_dates = raw_dates[valid_date].sort_values()
            clean = raw.loc[valid_date].copy()
            factors = build_symbol_factors(clean, symbol)
            total_rows += len(clean)
            quality.append({
                "symbol": symbol, "input_rows": int(len(raw)), "valid_date_rows": int(len(clean)),
                "factor_rows": int(len(factors)),
                "usable_ohlc_rows": usable_ohlc_count,
                "first_date": clean_dates.min().date().isoformat() if len(clean_dates) else "",
                "last_date": clean_dates.max().date().isoformat() if len(clean_dates) else "",
                "duplicate_dates": duplicate_count, "missing_numeric_rows": required_missing,
                "invalid_ohlc_rows": invalid_count,
                "zero_volume_rows": int((numeric["volume"] == 0).sum()) if len(numeric) else 0,
                "calendar_gaps_over_7d": int((clean_dates.diff().dt.days > 7).sum()) if len(clean_dates) else 0,
            })
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = expected_hashes.get(path.name)
            integrity.append({"file": path.name, "sha256": digest, "manifest_sha256": expected or "",
                              "hash_match": bool(expected and digest == expected)})
            if factors.empty:
                continue
            factors["year"] = pd.to_datetime(factors["date"]).dt.year
            for year, part in factors.groupby("year", sort=True):
                dest = tmpdir / f"factor_panel_{int(year)}.csv"
                part.drop(columns=["year"]).to_csv(dest, mode="a", header=not dest.exists(), index=False)
                written_years.add(int(year))

        daily_rows = []
        for year in sorted(written_years):
            panel = pd.read_csv(tmpdir / f"factor_panel_{year}.csv", parse_dates=["date"])
            for day, cross in panel.groupby("date", sort=True):
                n_cross = len(cross)
                if n_cross < 10:
                    continue
                for factor in FACTORS:
                    if factor not in cross:
                        continue
                    vals = cross[factor]
                    for horizon in HORIZONS:
                        target = f"fwd_{horizon}d_next_open"
                        pair = pd.DataFrame({"factor": vals, "target": cross[target]}).replace([np.inf, -np.inf], np.nan).dropna()
                        if len(pair) < 10 or pair["factor"].nunique() < 2:
                            continue
                        ic = pair["factor"].corr(pair["target"], method="spearman")
                        ranks = pair["factor"].rank(method="first")
                        buckets = pd.qcut(ranks, 5, labels=False, duplicates="drop")
                        bucket = pd.Series(buckets.to_numpy(), index=pair.index)
                        top = pair.loc[bucket == bucket.max(), "target"].mean()
                        bottom = pair.loc[bucket == bucket.min(), "target"].mean()
                        record = {
                            "date": day, "factor": factor, "horizon": horizon,
                            "rank_ic": float(ic), "top_bottom_spread": float(top-bottom),
                            "top_quintile_return": float(top), "bottom_quintile_return": float(bottom),
                            "n": int(len(pair)),
                        }
                        full_record = dict(record, segment="FULL_DESCRIPTIVE")
                        daily_rows.append(full_record)
                        if day >= split:
                            daily_rows.append(dict(record, segment="POST_SPLIT_DESCRIPTIVE"))

        daily = pd.DataFrame(daily_rows)
        if not daily.empty:
            # The full segment is intentionally one row per factor/date/horizon.
            daily = daily.drop_duplicates(["date", "factor", "horizon", "segment"], keep="last")
            results = []
            for (segment, factor, horizon), group in daily.groupby(["segment", "factor", "horizon"], sort=True):
                group = group.sort_values("date")
                ic_values = group["rank_ic"].astype(float).tolist()
                spread_values = group["top_bottom_spread"].astype(float).tolist()
                t_stat = hac_t_stat(ic_values, max_lag=int(horizon)-1)
                results.append({
                    "segment": segment, "family": next((k for k,v in FACTOR_FAMILIES.items() if factor in v), ""),
                    "factor": factor, "horizon": int(horizon), "dates": int(len(group)),
                    "mean_rank_ic": float(np.nanmean(ic_values)),
                    "rank_ic_positive_rate": float(np.nanmean(np.asarray(ic_values) > 0)),
                    "hac_t_stat_rank_ic": t_stat, "approx_p_value": _normal_two_sided_p(t_stat),
                    "mean_top_bottom_spread_gross": float(np.nanmean(spread_values)),
                    "mean_top_quintile_return_gross": float(group["top_quintile_return"].mean()),
                    "mean_bottom_quintile_return_gross": float(group["bottom_quintile_return"].mean()),
                    "median_cross_section": float(group["n"].median()),
                })
            results = pd.DataFrame(results)
            results["bh_q_value_within_segment"] = np.nan
            for segment, idx in results.groupby("segment").groups.items():
                results.loc[idx, "bh_q_value_within_segment"] = benjamini_hochberg(
                    results.loc[idx, "approx_p_value"].tolist()
                )
        else:
            results = pd.DataFrame()
        daily.to_csv(out / "daily_factor_diagnostics.csv", index=False)
        results.to_csv(out / "factor_research_results.csv", index=False)
        pd.DataFrame(quality).to_csv(out / "data_quality.csv", index=False)
        pd.DataFrame(integrity).to_csv(out / "manifest_integrity.csv", index=False)

    qdf = pd.DataFrame(quality)
    idf = pd.DataFrame(integrity)
    candidate_path = root / "candidate_symbols.csv"
    if candidate_path.exists():
        candidates = pd.read_csv(candidate_path, dtype={"symbol": str})
    else:
        candidates = pd.DataFrame({"symbol": [Path(f).stem for f in files]})
    if "symbol" not in candidates:
        candidates = candidates.rename(columns={candidates.columns[0]: "symbol"})
    candidates["symbol"] = candidates["symbol"].astype(str)
    coverage_cols = [c for c in ("symbol", "input_rows", "valid_date_rows", "factor_rows", "usable_ohlc_rows") if c in qdf.columns]
    candidate_coverage = candidates.merge(qdf[coverage_cols], on="symbol", how="left")
    candidate_coverage["has_bar_file"] = candidate_coverage["symbol"].isin({p.stem for p in files})
    candidate_coverage["has_valid_bars"] = candidate_coverage.get("usable_ohlc_rows", pd.Series(0, index=candidate_coverage.index)).fillna(0) > 0
    candidate_coverage["has_factor_rows"] = candidate_coverage.get("factor_rows", pd.Series(0, index=candidate_coverage.index)).fillna(0) > 0
    candidate_coverage.to_csv(out / "candidate_coverage.csv", index=False)
    missing_candidate_files = int((~candidate_coverage["has_bar_file"]).sum())
    candidates_without_bars = int((~candidate_coverage["has_valid_bars"]).sum())
    summary = {
        "source": manifest.get("source", "unknown"),
        "price_dataset": manifest.get("price_dataset", "unknown"),
        "start_date_requested": manifest.get("start_date"),
        "candidate_symbols": int(len(candidate_coverage)),
        "symbols_with_files": int(len(files)),
        "missing_candidate_files": missing_candidate_files,
        "candidate_symbols_without_valid_bars": candidates_without_bars,
        "failed_download_symbols": int(manifest.get("failed_symbols", 0)),
        "empty_download_symbols": int(manifest.get("empty_symbols", 0)),
        "symbols_with_valid_rows": int((qdf["valid_date_rows"] > 0).sum()) if not qdf.empty else 0,
        "input_rows": int(qdf["input_rows"].sum()) if not qdf.empty else 0,
        "valid_date_rows": int(qdf["valid_date_rows"].sum()) if not qdf.empty else 0,
        "usable_ohlc_rows": int(qdf["usable_ohlc_rows"].sum()) if not qdf.empty else 0,
        "first_date": qdf.loc[qdf["valid_date_rows"] > 0, "first_date"].min() if not qdf.empty and (qdf["valid_date_rows"] > 0).any() else None,
        "last_date": qdf["last_date"].max() if not qdf.empty else None,
        "duplicate_dates": int(qdf["duplicate_dates"].sum()) if not qdf.empty else 0,
        "missing_numeric_rows": int(qdf["missing_numeric_rows"].sum()) if not qdf.empty else 0,
        "invalid_ohlc_rows": int(qdf["invalid_ohlc_rows"].sum()) if not qdf.empty else 0,
        "manifest_hashes_verified": int(idf["hash_match"].sum()) if not idf.empty else 0,
        "manifest_hashes_missing_or_mismatched": int((~idf["hash_match"]).sum()) if not idf.empty else 0,
        "collection_complete": bool(manifest.get("collection_complete", False)),
        "point_in_time_universe_complete": bool(manifest.get("point_in_time_universe_complete", False)),
        "delisted_price_coverage_verified": bool(manifest.get("delisted_price_coverage_verified", False)),
        "corporate_actions_complete": corporate_actions_complete,
        "corporate_action_event_rows": int(len(action_events)),
        "corporate_action_conflicts": corporate_action_conflicts,
        "price_basis_for_signals": "event-adjusted historical OHLC; raw execution bars preserved" if corporate_actions_complete else "partial or unavailable event-adjusted OHLC; descriptive only",
        "research_grade_eligible": False,
        "research_grade_blockers": [
            "Historical point-in-time universe membership is not verified",
            f"{missing_candidate_files} candidate symbols have no downloaded bar file",
            f"{candidates_without_bars} candidate symbols have no valid price bars",
            "Delisted and suspended security price coverage is not verified",
            *([] if corporate_actions_complete else ["Corporate-action event collection is incomplete"]),
            *([] if corporate_action_conflicts == 0 else [f"{corporate_action_conflicts} same-day corporate-action ratios conflict and were excluded"]),
            "Corporate actions have not been independently reconciled against official exchange records",
            "Official trading-calendar completeness has not been verified",
            "Vendor raw bars have not been reconciled against official exchange samples",
            "Factor IC and quintile spreads are gross descriptive screens, not a costed portfolio backtest",
        ],
        "factor_screen_split_date": split_date,
        "split_label": "POST_SPLIT_DESCRIPTIVE is not a final blind test",
        "factor_results": "Exploratory cross-sectional rank IC and gross quintile spreads; HAC lag horizon-1 and BH q-values are reported",
    }
    (out / "data_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="artifacts/streaming_factor_research")
    parser.add_argument("--split-date", default="2024-01-01")
    args = parser.parse_args()
    result = run_streaming_research(args.input, args.output, split_date=args.split_date)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
