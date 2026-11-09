"""Point-in-time-safe corporate-action price adjustment helpers.

Raw OHLCV remains untouched for execution. For research signals and realized
forward returns, historical OHLC prices are back-adjusted using only verified
event ratios. A future event applies one common scale to all prices before its
effective date, so relative signals formed before that event are invariant;
returns spanning the event correctly remove the mechanical ex-right gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_COLUMNS = ("open", "high", "low", "close")


def _event_frame(frame: pd.DataFrame | None, *, source: str,
                 before_col: str, after_col: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["symbol", "date", "ratio", "source"])
    z = frame.copy()
    if "stock_id" in z and "symbol" not in z:
        z = z.rename(columns={"stock_id": "symbol"})
    if "date" not in z or "symbol" not in z:
        return pd.DataFrame(columns=["symbol", "date", "ratio", "source"])
    if before_col not in z or after_col not in z:
        return pd.DataFrame(columns=["symbol", "date", "ratio", "source"])
    z["symbol"] = z["symbol"].astype(str).str.strip()
    z["date"] = pd.to_datetime(z["date"], errors="coerce").dt.normalize()
    before = pd.to_numeric(z[before_col], errors="coerce")
    after = pd.to_numeric(z[after_col], errors="coerce")
    z["ratio"] = after / before
    z = z.loc[
        z["date"].notna() & before.gt(0) & after.gt(0)
        & np.isfinite(z["ratio"]) & z["ratio"].between(0.01, 100)
    ].copy()
    z["source"] = source
    return z[["symbol", "date", "ratio", "source"]]


def normalize_corporate_action_events(
    dividend_results: pd.DataFrame | None = None,
    split_prices: pd.DataFrame | None = None,
    capital_reductions: pd.DataFrame | None = None,
    par_value_changes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalize supported FinMind event datasets into one auditable event table."""
    parts = [
        _event_frame(dividend_results, source="TaiwanStockDividendResult",
                     before_col="before_price", after_col="reference_price"),
        _event_frame(split_prices, source="TaiwanStockSplitPrice",
                     before_col="before_price", after_col="after_price"),
        _event_frame(capital_reductions, source="TaiwanStockCapitalReductionReferencePrice",
                     before_col="ClosingPriceonTheLastTradingDay",
                     after_col="PostReductionReferencePrice"),
        _event_frame(par_value_changes, source="TaiwanStockParValueChange",
                     before_col="before_close", after_col="after_ref_close"),
    ]
    rows = pd.concat(parts, ignore_index=True)
    if rows.empty:
        return pd.DataFrame(columns=[
            "symbol", "date", "adjustment_ratio", "sources", "source_count", "conflict"
        ])
    # Multiple datasets can describe the same effective event. Identical ratios
    # are deduplicated; distinct same-day ratios are ambiguous and fail closed.
    out = []
    for (symbol, date), group in rows.groupby(["symbol", "date"], sort=True):
        unique_ratios = sorted({round(float(x), 10) for x in group["ratio"]})
        ratio = float(np.prod(unique_ratios))
        out.append({
            "symbol": str(symbol), "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "adjustment_ratio": ratio if len(unique_ratios) == 1 else np.nan,
            "sources": ";".join(sorted(set(group["source"].astype(str)))),
            "source_count": len(unique_ratios),
            "conflict": len(unique_ratios) > 1,
        })
    return pd.DataFrame(out)


def apply_back_adjustments(bars: pd.DataFrame, events: pd.DataFrame | None) -> pd.DataFrame:
    """Add adjusted OHLC columns without mutating raw execution prices.

    Each event's reference/before ratio is applied only to bars strictly before
    its effective date. Volume is not rewritten; volume-derived features are
    masked for 20 sessions after any action to avoid split-unit discontinuities.
    """
    z = bars.copy()
    for col in PRICE_COLUMNS:
        z[f"adj_{col}"] = pd.to_numeric(z[col], errors="coerce")
    z["corporate_action_window"] = False
    if events is None or events.empty:
        return z
    z["date"] = pd.to_datetime(z["date"], errors="coerce").dt.normalize()
    z = z.sort_values("date").reset_index(drop=True)
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce").dt.normalize()
    ev["adjustment_ratio"] = pd.to_numeric(ev["adjustment_ratio"], errors="coerce")
    ev = ev.loc[
        ev["date"].notna() & ev["adjustment_ratio"].gt(0)
        & np.isfinite(ev["adjustment_ratio"])
    ].sort_values("date")
    for event in ev.itertuples(index=False):
        before = z["date"] < event.date
        for col in PRICE_COLUMNS:
            z.loc[before, f"adj_{col}"] *= float(event.adjustment_ratio)
        if z.empty or event.date < z["date"].min() or event.date > z["date"].max():
            continue
        event_positions = np.flatnonzero(z["date"].to_numpy() >= np.datetime64(event.date))
        if len(event_positions):
            first = int(event_positions[0])
            z.loc[first:min(first + 20, len(z) - 1), "corporate_action_window"] = True
    return z


def audit_action_coverage(symbols: list[str], events: pd.DataFrame,
                          completed_symbols: int, expected_symbols: int) -> dict:
    valid_symbols = set(events["symbol"].astype(str)) if events is not None and not events.empty else set()
    return {
        "corporate_actions_complete": bool(completed_symbols >= expected_symbols),
        "corporate_action_symbols_with_events": len(valid_symbols.intersection(set(map(str, symbols)))),
        "corporate_action_event_rows": int(len(events)) if events is not None else 0,
        "corporate_action_conflicts": int(events["conflict"].fillna(False).sum()) if events is not None and "conflict" in events else 0,
        "corporate_action_symbols_without_reported_events": int(max(0, expected_symbols - len(valid_symbols)),
        "corporate_action_source": "FinMind TaiwanStockDividendResult, TaiwanStockSplitPrice, TaiwanStockCapitalReductionReferencePrice, TaiwanStockParValueChange",
    }
