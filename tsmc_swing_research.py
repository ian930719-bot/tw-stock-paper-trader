"""Causal, single-symbol swing turning-point research for TSMC 2330.

This is an event-labeling and lead/lag diagnostic stage, not a trading strategy.
Pivots are labeled retrospectively, but confirmation dates are recorded separately.
All predictors are computed using data available no later than each session close.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {"date", "open", "high", "low", "close", "volume"}
LEADS = (1, 3, 5, 10, 20)
THRESHOLDS = ((0.03, 2.0), (0.05, 2.0), (0.08, 2.0), (0.03, 3.0), (0.05, 3.0))


def load_tsmc(path: str) -> pd.DataFrame:
    root = Path(path)
    files = sorted((root / "bars").glob("*.csv")) if (root / "bars").is_dir() else (
        sorted(root.glob("*.csv")) if root.is_dir() else [root]
    )
    if not files:
        raise ValueError(f"No CSV history files found under {root}")
    frames = []
    for file in files:
        frame = pd.read_csv(file)
        if "symbol" in frame:
            frame = frame.loc[frame["symbol"].astype(str).str.strip() == "2330"].copy()
        elif file.stem != "2330" and root.is_dir():
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("No TSMC 2330 rows found; refusing to substitute another symbol")
    z = pd.concat(frames, ignore_index=True)
    missing = REQUIRED.difference(z.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    z["date"] = pd.to_datetime(z["date"], errors="coerce").dt.normalize()
    for c in ("open", "high", "low", "close", "volume", "turnover"):
        if c in z:
            z[c] = pd.to_numeric(z[c], errors="coerce")
    z = z.dropna(subset=list(REQUIRED)).sort_values("date")
    z = z.drop_duplicates("date", keep="last").reset_index(drop=True)
    valid = (
        (z[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (z["volume"] >= 0)
        & (z["high"] >= z[["open", "close", "low"]].max(axis=1))
        & (z["low"] <= z[["open", "close", "high"]].min(axis=1))
    )
    z = z.loc[valid].reset_index(drop=True)
    if len(z) < 260:
        raise ValueError(f"Insufficient TSMC history after quality checks: {len(z)} rows")
    z["symbol"] = "2330"
    return z


def build_causal_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Features at row t use only OHLCV through t; no centered windows."""
    z = bars.copy().sort_values("date").reset_index(drop=True)
    c, o, h, l, v = z["close"], z["open"], z["high"], z["low"], z["volume"]
    prev = c.shift(1)
    for n in (1, 2, 3, 5, 10, 20, 40, 60):
        z[f"return_{n}"] = c.pct_change(n)
    for n in (5, 10, 20, 50, 100, 200):
        ma = c.rolling(n, min_periods=n).mean()
        z[f"distance_sma_{n}"] = c / ma - 1
    z["sma20_slope_5"] = c.rolling(20, min_periods=20).mean().pct_change(5)
    z["sma50_slope_10"] = c.rolling(50, min_periods=50).mean().pct_change(10)
    z["trend_stack"] = (
        (c > c.rolling(20, min_periods=20).mean())
        & (c.rolling(20, min_periods=20).mean() > c.rolling(50, min_periods=50).mean())
        & (c.rolling(50, min_periods=50).mean() > c.rolling(200, min_periods=200).mean())
    ).astype(float)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    z["atr14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    z["atr_pct14"] = z["atr14"] / c
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    z["rsi14"] = 100 - 100 / (1 + rs)
    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    z["macd_pct"] = (ema12 - ema26) / c
    z["macd_signal_pct"] = (z["macd_pct"].ewm(span=9, adjust=False, min_periods=9).mean())
    mid = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    z["bb_width20"] = 4 * sd / mid
    z["bb_position20"] = (c - (mid - 2*sd)) / (4*sd).replace(0, np.nan)
    volbase = v.shift(1).rolling(20, min_periods=20).mean()
    z["relative_volume20"] = v / volbase.replace(0, np.nan)
    money = z["turnover"] if "turnover" in z else c * v
    z["turnover_ratio20"] = money / money.shift(1).rolling(20, min_periods=20).mean().replace(0, np.nan)
    z["realized_vol20"] = c.pct_change().rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252)
    z["realized_vol60"] = c.pct_change().rolling(60, min_periods=60).std(ddof=0) * np.sqrt(252)
    z["volatility_compression"] = z["realized_vol20"] / z["realized_vol60"].replace(0, np.nan)
    z["overnight_gap"] = o / prev - 1
    z["intraday_range_pct"] = (h-l) / prev
    z["close_location"] = (c-l) / (h-l).replace(0, np.nan)
    z["upper_wick_pct"] = (h - z[["open", "close"]].max(axis=1)) / c
    z["lower_wick_pct"] = (z[["open", "close"]].min(axis=1) - l) / c
    z["distance_prior_high20"] = c / h.shift(1).rolling(20, min_periods=20).max() - 1
    z["distance_prior_low20"] = c / l.shift(1).rolling(20, min_periods=20).min() - 1
    # Event windows supplied by the collector are excluded from event diagnostics.
    if "corporate_action_window" in z:
        z["corporate_action_window"] = z["corporate_action_window"].fillna(True).astype(bool)
    else:
        z["corporate_action_window"] = False
    return z.replace([np.inf, -np.inf], np.nan)


def detect_confirmed_pivots(features: pd.DataFrame, min_reversal_pct: float,
                            atr_multiple: float) -> pd.DataFrame:
    """Causal close-confirmed ZigZag; pivot and confirmation dates are distinct.

    The extreme is the actual intraday high/low. A pivot cannot be confirmed on
    the same bar that sets the extreme, avoiding unknown intraday path ordering.
    The reversal threshold is frozen using ATR% known on the extreme date.
    """
    z = features.reset_index(drop=True)
    if not 0 < min_reversal_pct < 1 or atr_multiple <= 0:
        raise ValueError("Invalid pivot threshold")
    events = []
    if len(z) < 3:
        return pd.DataFrame(columns=["type", "pivot_index", "confirmation_index",
                                     "pivot_date", "confirmation_date", "pivot_price",
                                     "confirmation_close", "reversal_pct", "confirmation_lag"])
    mode = "seek"
    first_valid = z["atr_pct14"].first_valid_index()
    if first_valid is None:
        return pd.DataFrame(columns=["type", "pivot_index", "confirmation_index",
                                     "pivot_date", "confirmation_date", "pivot_price",
                                     "confirmation_close", "reversal_pct", "confirmation_lag"])
    candidate_high = {"idx": int(first_valid), "price": float(z.loc[first_valid, "high"]), "atr_pct": float(z.loc[first_valid, "atr_pct14"])}
    candidate_low = {"idx": int(first_valid), "price": float(z.loc[first_valid, "low"]), "atr_pct": float(z.loc[first_valid, "atr_pct14"])}
    for i in range(int(first_valid) + 1, len(z)):
        row = z.iloc[i]
        atrp = row["atr_pct14"]
        if not np.isfinite(atrp):
            continue
        if mode == "seek":
            if row["high"] > candidate_high["price"]:
                candidate_high = {"idx": i, "price": float(row["high"]), "atr_pct": float(atrp)}
            if row["low"] < candidate_low["price"]:
                candidate_low = {"idx": i, "price": float(row["low"]), "atr_pct": float(atrp)}
            up_move = row["close"] / candidate_low["price"] - 1
            down_move = 1 - row["close"] / candidate_high["price"]
            up_threshold = max(min_reversal_pct, atr_multiple * candidate_low["atr_pct"])
            down_threshold = max(min_reversal_pct, atr_multiple * candidate_high["atr_pct"])
            if up_move >= up_threshold and candidate_low["idx"] < i and up_move >= down_move:
                mode = "up"
                events.append(("LOW", candidate_low["idx"], i, candidate_low["price"], up_move))
                window = z.loc[candidate_low["idx"] + 1:i, "high"]
                peak_idx = int(window.idxmax())
                candidate_high = {"idx": peak_idx, "price": float(z.loc[peak_idx, "high"]), "atr_pct": float(z.loc[peak_idx, "atr_pct14"])}
            elif down_move >= down_threshold and candidate_high["idx"] < i:
                mode = "down"
                events.append(("HIGH", candidate_high["idx"], i, candidate_high["price"], down_move))
                window = z.loc[candidate_high["idx"] + 1:i, "low"]
                trough_idx = int(window.idxmin())
                candidate_low = {"idx": trough_idx, "price": float(z.loc[trough_idx, "low"]), "atr_pct": float(z.loc[trough_idx, "atr_pct14"])}
        elif mode == "up":
            if row["high"] > candidate_high["price"]:
                candidate_high = {"idx": i, "price": float(row["high"]), "atr_pct": float(atrp)}
            threshold = max(min_reversal_pct, atr_multiple * candidate_high["atr_pct"])
            if i > candidate_high["idx"] and 1 - row["close"] / candidate_high["price"] >= threshold:
                rev = 1 - row["close"] / candidate_high["price"]
                events.append(("HIGH", candidate_high["idx"], i, candidate_high["price"], rev))
                mode = "down"
                candidate_low = {"idx": i, "price": float(row["low"]), "atr_pct": float(atrp)}
        else:
            if row["low"] < candidate_low["price"]:
                candidate_low = {"idx": i, "price": float(row["low"]), "atr_pct": float(atrp)}
            threshold = max(min_reversal_pct, atr_multiple * candidate_low["atr_pct"])
            if i > candidate_low["idx"] and row["close"] / candidate_low["price"] - 1 >= threshold:
                rev = row["close"] / candidate_low["price"] - 1
                events.append(("LOW", candidate_low["idx"], i, candidate_low["price"], rev))
                mode = "up"
                candidate_high = {"idx": i, "price": float(row["high"]), "atr_pct": float(atrp)}
    out = []
    for kind, pivot_i, confirm_i, price, reversal in events:
        out.append({
            "type": kind, "pivot_index": int(pivot_i), "confirmation_index": int(confirm_i),
            "pivot_date": z.loc[pivot_i, "date"].date().isoformat(),
            "confirmation_date": z.loc[confirm_i, "date"].date().isoformat(),
            "pivot_price": float(price), "confirmation_close": float(z.loc[confirm_i, "close"]),
            "reversal_pct": float(reversal), "confirmation_lag": int(confirm_i-pivot_i),
            "min_reversal_pct": float(min_reversal_pct), "atr_multiple": float(atr_multiple),
        })
    return pd.DataFrame(out)


def build_lead_lag(events: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_cols = [
        c for c in features.columns if c.startswith((
            "return_", "distance_sma_", "sma", "trend_", "atr_pct", "rsi", "macd",
            "bb_", "relative_volume", "turnover_ratio", "realized_vol",
            "volatility_", "overnight_gap", "intraday_range", "close_location",
            "upper_wick", "lower_wick", "distance_prior",
        ))
    ]
    for event_id, event in events.reset_index(drop=True).iterrows():
        pivot_i = int(event["pivot_index"])
        for lead in LEADS:
            idx = pivot_i - lead
            if idx < 0:
                continue
            row = features.iloc[idx]
            if bool(row["corporate_action_window"]):
                continue
            record = {
                "event_id": event_id, "type": event["type"], "pivot_date": event["pivot_date"],
                "lead_sessions": lead, "feature_date": row["date"].date().isoformat(),
                "confirmation_lag": int(event["confirmation_lag"]),
                "min_reversal_pct": float(event["min_reversal_pct"]),
                "atr_multiple": float(event["atr_multiple"]),
            }
            record.update({c: row[c] for c in feature_cols})
            rows.append(record)
    return pd.DataFrame(rows)


def run(input_path: str, output_dir: str) -> dict:
    bars = load_tsmc(input_path)
    features = build_causal_features(bars)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    event_frames, lead_frames = [], []
    for pct, atr_mult in THRESHOLDS:
        events = detect_confirmed_pivots(features, pct, atr_mult)
        if events.empty:
            continue
        events.insert(0, "threshold_id", f"p{pct:.3f}_atr{atr_mult:.1f}")
        event_frames.append(events)
        lead = build_lead_lag(events, features)
        if not lead.empty:
            lead.insert(0, "threshold_id", f"p{pct:.3f}_atr{atr_mult:.1f}")
            lead_frames.append(lead)
    events_all = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    lead_all = pd.concat(lead_frames, ignore_index=True) if lead_frames else pd.DataFrame()
    # No parameter is selected here. All predeclared definitions remain visible.
    if not events_all.empty:
        events_all.to_csv(out / "tsmc_2330_pivot_events.csv", index=False)
    if not lead_all.empty:
        lead_all.to_csv(out / "tsmc_2330_pivot_leadlag.csv", index=False)
    features.to_csv(out / "tsmc_2330_causal_features.csv", index=False)
    summary = {
        "symbol": "2330", "rows": int(len(bars)),
        "first_date": bars["date"].min().date().isoformat(),
        "last_date": bars["date"].max().date().isoformat(),
        "source": "input CSV; provenance must be supplied by the caller",
        "price_basis": "raw OHLCV; no retrospectively adjusted prices used",
        "thresholds": [{"min_reversal_pct": p, "atr_multiple": a} for p, a in THRESHOLDS],
        "events": int(len(events_all)),
        "high_events": int((events_all["type"] == "HIGH").sum()) if not events_all.empty else 0,
        "low_events": int((events_all["type"] == "LOW").sum()) if not events_all.empty else 0,
        "median_confirmation_lag": float(events_all["confirmation_lag"].median()) if not events_all.empty else None,
        "lead_lag_rows": int(len(lead_all)),
        "corporate_action_window_rows_excluded": int(features["corporate_action_window"].sum()),
        "strategy_selected": False, "backtest_run": False,
        "note": "Pivots are retrospective labels; confirmation dates are causal but lagged. Lead-lag rows sample features before pivot dates. This is not a strategy return test.",
    }
    (out / "tsmc_2330_research_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="TSMC 2330 causal swing pivot research")
    parser.add_argument("--input", required=True, help="CSV file or directory containing TSMC 2330 bars")
    parser.add_argument("--output", default="artifacts/tsmc_2330_swing")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
