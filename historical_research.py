"""Historical Taiwan equity panel audit and broad factor discovery.

Signals are formed at the close. Forward returns enter at the next session open,
so same-bar close execution is never assumed. This module is research-only:
it does not place orders or claim a candidate is profitable.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

REQUIRED = {"symbol", "date", "open", "high", "low", "close", "volume"}
FACTOR_FAMILIES = {
    "momentum": ["ret_5", "ret_20", "ret_60", "ret_120", "relative_momentum_20", "relative_momentum_60"],
    "reversal": ["reversal_1", "reversal_5", "distance_sma_5", "distance_sma_20"],
    "trend": ["distance_sma_50", "distance_sma_100", "distance_sma_200", "sma50_slope_20", "trend_stack"],
    "breakout": ["breakout_20", "breakout_55", "distance_high_20", "distance_high_55"],
    "volume": ["relative_volume_20", "volume_trend_20", "turnover_shock_20"],
    "volatility": ["atr_pct_14", "realized_vol_20", "realized_vol_60", "volatility_compression"],
    "gap_range": ["overnight_gap", "intraday_range_pct", "close_location"],
}


def normalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError("歷史行情缺少欄位：" + ", ".join(sorted(missing)))
    z = frame.copy()
    z["symbol"] = z["symbol"].astype(str).str.strip()
    z["date"] = pd.to_datetime(z["date"], errors="coerce").dt.normalize()
    for col in ("open", "high", "low", "close", "volume"):
        z[col] = pd.to_numeric(z[col], errors="coerce")
    z = z.dropna(subset=["symbol", "date", "open", "high", "low", "close", "volume"])
    return z.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)


def audit_historical_panel(frame: pd.DataFrame, *, source: str = "unspecified",
                           price_basis: str = "unknown",
                           point_in_time_universe: bool = False,
                           delisted_coverage: bool = False,
                           trading_calendar: pd.DatetimeIndex | None = None) -> tuple[pd.DataFrame, dict]:
    """Audit raw panel quality and expose survivorship/source limitations."""
    z = normalize_panel(frame)
    duplicate_count = int(z.duplicated(["symbol", "date"], keep=False).sum())
    valid_price = (z[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid_volume = z["volume"] >= 0
    valid_ohlc = (
        (z["high"] >= z[["open", "close", "low"]].max(axis=1))
        & (z["low"] <= z[["open", "close", "high"]].min(axis=1))
    )
    invalid = ~(valid_price & valid_volume & valid_ohlc)
    z = z.loc[~invalid].drop_duplicates(["symbol", "date"], keep="last").copy()
    z["year"] = z["date"].dt.year
    rows = []
    for symbol, g in z.groupby("symbol", sort=True):
        dates = g["date"].sort_values()
        gaps = int((dates.diff().dt.days > 7).sum())
        rows.append({
            "symbol": symbol, "rows": int(len(g)),
            "first_date": dates.min().date().isoformat(),
            "last_date": dates.max().date().isoformat(),
            "years": int(g["year"].nunique()),
            "zero_volume_rows": int((g["volume"] == 0).sum()),
            "calendar_gaps_over_7d": gaps,
            "duplicate_rows_removed": int(frame.assign(date=pd.to_datetime(frame["date"], errors="coerce").dt.normalize())
                                          .duplicated(["symbol", "date"], keep=False).sum()) if len(frame) else 0,
        })
    by_symbol = pd.DataFrame(rows)
    if trading_calendar is not None:
        cal = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).normalize().unique().sort_values()
        coverage = []
        for symbol, g in z.groupby("symbol"):
            expected = cal[(cal >= g["date"].min()) & (cal <= g["date"].max())]
            actual = pd.DatetimeIndex(g["date"].unique())
            missing_sessions = len(expected.difference(actual))
            coverage.append({"symbol": symbol, "expected_sessions": len(expected),
                             "observed_sessions": len(actual), "missing_sessions": missing_sessions,
                             "session_coverage": len(actual) / len(expected) if len(expected) else np.nan})
        if not by_symbol.empty:
            by_symbol = by_symbol.merge(pd.DataFrame(coverage), on="symbol", how="left")
    limitations = []
    if price_basis != "raw":
        limitations.append("價格基礎未確認為 raw 原始成交價")
    if not point_in_time_universe:
        limitations.append("缺少完整 point-in-time 歷史股票池")
    if not delisted_coverage:
        limitations.append("下市／下櫃證券覆蓋未驗證")
    if duplicate_count:
        limitations.append(f"原始資料存在 {duplicate_count} 筆重複 symbol-date rows")
    if invalid.any():
        limitations.append(f"排除 {int(invalid.sum())} 筆無效 OHLCV rows")
    summary = {
        "source": source, "price_basis": price_basis,
        "input_rows": int(len(frame)), "valid_rows": int(len(z)),
        "symbols": int(z["symbol"].nunique()),
        "first_date": z["date"].min().date().isoformat() if not z.empty else None,
        "last_date": z["date"].max().date().isoformat() if not z.empty else None,
        "duplicate_rows": duplicate_count, "invalid_rows": int(invalid.sum()),
        "point_in_time_universe": bool(point_in_time_universe),
        "delisted_coverage": bool(delisted_coverage),
        "research_grade_eligible": not limitations,
        "limitations": limitations,
    }
    return by_symbol, summary


def build_factor_panel(frame: pd.DataFrame, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Build interpretable close-known factors and next-open forward returns."""
    z = normalize_panel(frame)
    outputs = []
    for symbol, g0 in z.groupby("symbol", sort=False):
        g = g0.sort_values("date").copy()
        close = g["close"]
        open_ = g["open"]
        high, low, volume = g["high"], g["low"], g["volume"]
        prev_close = close.shift(1)
        for n in (5, 20, 60, 120):
            g[f"ret_{n}"] = close.pct_change(n)
        g["reversal_1"] = -close.pct_change(1)
        g["reversal_5"] = -close.pct_change(5)
        for n in (5, 20, 50, 100, 200):
            sma = close.rolling(n, min_periods=n).mean()
            g[f"distance_sma_{n}"] = close / sma - 1
        g["sma50_slope_20"] = close.rolling(50, min_periods=50).mean().pct_change(20)
        g["trend_stack"] = (
            (close > close.rolling(50, min_periods=50).mean())
            & (close.rolling(50, min_periods=50).mean() > close.rolling(100, min_periods=100).mean())
            & (close.rolling(100, min_periods=100).mean() > close.rolling(200, min_periods=200).mean())
        ).astype(float)
        for n in (20, 55):
            prior_high = high.shift(1).rolling(n, min_periods=n).max()
            g[f"breakout_{n}"] = (close > prior_high).astype(float)
            g[f"distance_high_{n}"] = close / prior_high - 1
        avg_vol = volume.shift(1).rolling(20, min_periods=20).mean()
        g["relative_volume_20"] = volume / avg_vol.replace(0, np.nan)
        g["volume_trend_20"] = volume.rolling(5, min_periods=5).mean() / volume.rolling(20, min_periods=20).mean().replace(0, np.nan)
        money = g["turnover"].where(g["turnover"].notna()) if "turnover" in g else close * volume
        g["turnover_shock_20"] = money / money.shift(1).rolling(20, min_periods=20).mean().replace(0, np.nan)
        tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
        g["atr_pct_14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / close
        daily_ret = close.pct_change()
        g["realized_vol_20"] = daily_ret.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252)
        g["realized_vol_60"] = daily_ret.rolling(60, min_periods=60).std(ddof=0) * np.sqrt(252)
        g["volatility_compression"] = g["realized_vol_20"] / g["realized_vol_60"].replace(0, np.nan)
        g["overnight_gap"] = open_ / prev_close - 1
        g["intraday_range_pct"] = (high - low) / prev_close
        g["close_location"] = (close - low) / (high - low).replace(0, np.nan)
        if benchmark is not None:
            b = pd.Series(benchmark).copy()
            b.index = pd.to_datetime(b.index).normalize()
            for n in (20, 60):
                g[f"relative_momentum_{n}"] = g[f"ret_{n}"] - g["date"].map(b.pct_change(n))
        else:
            g["relative_momentum_20"] = np.nan
            g["relative_momentum_60"] = np.nan
        # Entry at next session open; exit at close after h sessions.
        next_open = open_.shift(-1)
        for h in (1, 5, 10, 20, 60):
            g[f"fwd_{h}d_next_open"] = close.shift(-h) / next_open - 1
        outputs.append(g)
    panel = pd.concat(outputs, ignore_index=True) if outputs else z
    for family, factors in FACTOR_FAMILIES.items():
        for factor in factors:
            if factor in panel:
                panel[f"rank_{factor}"] = panel.groupby("date")[factor].rank(pct=True)
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def evaluate_factor_panel(panel: pd.DataFrame, *, split_date: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate date-wise rank IC and top/bottom quintile spreads without fitting."""
    rows, registry = [], []
    for family, factors in FACTOR_FAMILIES.items():
        for factor in factors:
            if factor not in panel:
                continue
            for horizon in (5, 20, 60):
                target = f"fwd_{horizon}d_next_open"
                if target not in panel:
                    continue
                z = panel[["date", "symbol", factor, target]].replace([np.inf, -np.inf], np.nan).dropna()
                if split_date:
                    z = z[z["date"] >= pd.Timestamp(split_date)]
                    segment = "OOS_AFTER_SPLIT"
                else:
                    segment = "FULL_DESCRIPTIVE"
                daily = []
                for day, d in z.groupby("date"):
                    if len(d) < 10 or d[factor].nunique() < 3:
                        continue
                    ic = d[factor].corr(d[target], method="spearman")
                    d = d.assign(_bucket=pd.qcut(d[factor].rank(method="first"), 5, labels=False))
                    top = d.loc[d["_bucket"] == 4, target].mean()
                    bottom = d.loc[d["_bucket"] == 0, target].mean()
                    daily.append((day, ic, top-bottom, len(d)))
                if daily:
                    a = pd.DataFrame(daily, columns=["date", "rank_ic", "top_bottom_spread", "n"])
                    rows.append({
                        "family": family, "factor": factor, "horizon": horizon,
                        "segment": segment, "dates": int(len(a)),
                        "mean_rank_ic": float(a["rank_ic"].mean()),
                        "rank_ic_positive_rate": float((a["rank_ic"] > 0).mean()),
                        "mean_top_bottom_spread": float(a["top_bottom_spread"].mean()),
                        "median_cross_section": float(a["n"].median()),
                    })
            registry.append({"family": family, "factor": factor,
                             "definition": "Close-known, symbol-level factor; evaluated with next-session-open entry",
                             "selection_status": "UNSELECTED"})
    return pd.DataFrame(rows), pd.DataFrame(registry)


def run_research(input_csv: str, output_dir: str, *, split_date: str | None = None,
                 source: str = "user-supplied CSV", price_basis: str = "unknown",
                 point_in_time_universe: bool = False, delisted_coverage: bool = False,\n                 corporate_actions_reconciled: bool = False, trading_calendar_verified: bool = False,\n                 official_source_reconciled: bool = False) -> dict:
    raw = pd.read_csv(input_csv)
    audit, summary = audit_historical_panel(
        raw, source=source, price_basis=price_basis,
        point_in_time_universe=point_in_time_universe,
        delisted_coverage=delisted_coverage,
    )
    factors = build_factor_panel(raw)
    results, registry = evaluate_factor_panel(factors, split_date=split_date)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    audit.to_csv(target / "data_coverage.csv", index=False)
    factors.to_csv(target / "historical_factor_panel.csv", index=False)
    results.to_csv(target / "factor_research_results.csv", index=False)
    registry.to_csv(target / "hypothesis_registry.csv", index=False)
    (target / "data_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with symbol,date,open,high,low,close,volume")
    parser.add_argument("--output", default="artifacts/historical_research")
    parser.add_argument("--split-date", default=None, help="Predeclared OOS start date, YYYY-MM-DD")
    parser.add_argument("--source", default="user-supplied CSV")
    parser.add_argument("--price-basis", choices=["raw", "adjusted", "unknown"], default="unknown")
    parser.add_argument("--point-in-time-universe", action="store_true")
    parser.add_argument("--delisted-coverage", action="store_true")
    args = parser.parse_args()
    summary = run_research(
        args.input, args.output, split_date=args.split_date, source=args.source,
        price_basis=args.price_basis, point_in_time_universe=args.point_in_time_universe,
        delisted_coverage=args.delisted_coverage,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
