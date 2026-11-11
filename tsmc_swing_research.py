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

from corporate_action_adjustments import apply_back_adjustments

REQUIRED = {"date", "open", "high", "low", "close", "volume"}
LEADS = (1, 3, 5, 10, 20)
THRESHOLDS = (
    # Existing reference definitions remain unchanged.
    (0.03, 2.0), (0.05, 2.0), (0.08, 2.0), (0.03, 3.0), (0.05, 3.0),
    # Predeclared lower-amplitude sample-expansion grid; each remains a separate label set.
    (0.01, 1.0), (0.02, 1.0), (0.03, 1.0), (0.04, 1.0), (0.02, 1.5),
)


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
    action_mask = z.get("corporate_action_window", pd.Series(False, index=z.index)).fillna(True).astype(bool)
    z.loc[action_mask, ["relative_volume20", "turnover_ratio20"]] = np.nan
    z["realized_vol20"] = c.pct_change().rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252)
    z["realized_vol60"] = c.pct_change().rolling(60, min_periods=60).std(ddof=0) * np.sqrt(252)
    z["volatility_compression"] = z["realized_vol20"] / z["realized_vol60"].replace(0, np.nan)
    z["overnight_gap"] = o / prev - 1
    z["intraday_range_pct"] = (h-l) / prev
    z["close_location"] = (c-l) / (h-l).replace(0, np.nan)
    z["upper_wick_pct"] = (h - z[["open", "close"]].max(axis=1)) / c
    z["lower_wick_pct"] = (z[["open", "close"]].min(axis=1) - l) / c
    prior_high20 = h.shift(1).rolling(20, min_periods=20).max()
    prior_low20 = l.shift(1).rolling(20, min_periods=20).min()
    z["distance_prior_high20"] = c / prior_high20 - 1
    z["distance_prior_low20"] = c / prior_low20 - 1

    # Causal momentum/volume divergence: compare current price extremes to
    # prior 20-session oscillator/volume extremes, never including today's bar.
    rsi_prior_high20 = z["rsi14"].shift(1).rolling(20, min_periods=20).max()
    rsi_prior_low20 = z["rsi14"].shift(1).rolling(20, min_periods=20).min()
    z["rsi_change_3"] = z["rsi14"].diff(3)
    z["macd_hist_pct"] = z["macd_pct"] - z["macd_signal_pct"]
    z["macd_hist_change_3"] = z["macd_hist_pct"].diff(3)
    macd_prior_high20 = z["macd_hist_pct"].shift(1).rolling(20, min_periods=20).max()
    macd_prior_low20 = z["macd_hist_pct"].shift(1).rolling(20, min_periods=20).min()
    rv_prior_high20 = z["relative_volume20"].shift(1).rolling(20, min_periods=20).max()
    rv_prior_low20 = z["relative_volume20"].shift(1).rolling(20, min_periods=20).min()

    high_break = z["distance_prior_high20"].ge(0)
    low_break = z["distance_prior_low20"].le(0)
    z["bearish_rsi_divergence20"] = (high_break & z["rsi14"].lt(rsi_prior_high20)).astype(float)
    z["bullish_rsi_divergence20"] = (low_break & z["rsi14"].gt(rsi_prior_low20)).astype(float)
    z["bearish_macd_divergence20"] = (high_break & z["macd_hist_pct"].lt(macd_prior_high20)).astype(float)
    z["bullish_macd_divergence20"] = (low_break & z["macd_hist_pct"].gt(macd_prior_low20)).astype(float)
    z["bearish_volume_divergence20"] = (high_break & z["relative_volume20"].lt(rv_prior_high20)).astype(float)
    z["bullish_volume_divergence20"] = (low_break & z["relative_volume20"].gt(rv_prior_low20)).astype(float)
    z["rsi_divergence_strength_high20"] = (z["rsi14"] - rsi_prior_high20).where(high_break)
    z["rsi_divergence_strength_low20"] = (z["rsi14"] - rsi_prior_low20).where(low_break)
    z["macd_divergence_strength_high20"] = (z["macd_hist_pct"] - macd_prior_high20).where(high_break)
    z["macd_divergence_strength_low20"] = (z["macd_hist_pct"] - macd_prior_low20).where(low_break)
    z["volume_divergence_strength_high20"] = (z["relative_volume20"] - rv_prior_high20).where(high_break)
    z["volume_divergence_strength_low20"] = (z["relative_volume20"] - rv_prior_low20).where(low_break)
    # Expanded technical-indicator panel. Every feature at t uses data through t only.
    for n in (5, 10, 12, 20, 26, 50, 60, 120, 200):
        ema = c.ewm(span=n, adjust=False, min_periods=n).mean()
        z[f"ema_{n}"] = ema
        z[f"distance_ema_{n}"] = c / ema - 1
    z["ema12_ema26_spread"] = (z["ema_12"] / z["ema_26"]) - 1
    z["ema20_slope_5"] = z["ema_20"].pct_change(5)
    z["ema50_slope_10"] = z["ema_50"].pct_change(10)
    z["ema200_slope_20"] = z["ema_200"].pct_change(20)

    # Stochastic oscillator, CCI and Williams %R.
    low14 = l.rolling(14, min_periods=14).min()
    high14 = h.rolling(14, min_periods=14).max()
    z["stoch_k14"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    z["stoch_d3"] = z["stoch_k14"].rolling(3, min_periods=3).mean()
    typical = (h + l + c) / 3
    typical_ma20 = typical.rolling(20, min_periods=20).mean()
    typical_mad20 = typical.rolling(20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    z["cci20"] = (typical - typical_ma20) / (0.015 * typical_mad20.replace(0, np.nan))
    z["williams_r14"] = -100 * (high14 - c) / (high14 - low14).replace(0, np.nan)

    # Directional movement and ADX (Wilder-style exponential smoothing).
    up_move, down_move = h.diff(), -l.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr_smooth = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / tr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / tr_smooth.replace(0, np.nan)
    z["plus_di14"], z["minus_di14"] = plus_di, minus_di
    z["dx14"] = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    z["adx14"] = z["dx14"].ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    # Volume / money-flow indicators. Daily OHLCV cannot reconstruct intraday VWAP;
    # these are explicitly rolling daily-bar VWAP proxies, not exchange intraday VWAP.
    direction = np.sign(c.diff()).fillna(0)
    z["obv"] = (direction * v).cumsum()
    z["obv_change_5"] = z["obv"].diff(5)
    raw_money_flow = typical * v
    positive_flow = raw_money_flow.where(typical.diff() > 0, 0.0)
    negative_flow = raw_money_flow.where(typical.diff() < 0, 0.0)
    pos14 = positive_flow.rolling(14, min_periods=14).sum()
    neg14 = negative_flow.rolling(14, min_periods=14).sum()
    money_ratio = pos14 / neg14.replace(0, np.nan)
    z["mfi14"] = 100 - 100 / (1 + money_ratio)
    money_range = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    z["chaikin_ad_line"] = (money_range.fillna(0) * v).cumsum()
    z["chaikin_money_flow20"] = (
        (money_range.fillna(0) * v).rolling(20, min_periods=20).sum()
        / v.rolling(20, min_periods=20).sum().replace(0, np.nan)
    )
    z["volume_sma5_ratio"] = v / v.shift(1).rolling(5, min_periods=5).mean().replace(0, np.nan)
    z["volume_sma20_ratio"] = v / v.shift(1).rolling(20, min_periods=20).mean().replace(0, np.nan)
    z["volume_zscore20"] = (
        (v - v.shift(1).rolling(20, min_periods=20).mean())
        / v.shift(1).rolling(20, min_periods=20).std(ddof=0).replace(0, np.nan)
    )
    for n in (5, 10, 20):
        vol_sum = v.rolling(n, min_periods=n).sum()
        z[f"rolling_vwap_proxy_{n}"] = (
            (typical * v).rolling(n, min_periods=n).sum() / vol_sum.replace(0, np.nan)
        )
        z[f"distance_vwap_proxy_{n}"] = c / z[f"rolling_vwap_proxy_{n}"] - 1

    # Trend/momentum extensions and price structure.
    for n in (5, 10, 20, 60):
        z[f"roc_{n}"] = c.pct_change(n) * 100
    z["ppo"] = 100 * (ema12 - ema26) / ema26.replace(0, np.nan)
    z["ppo_signal"] = z["ppo"].ewm(span=9, adjust=False, min_periods=9).mean()
    z["ppo_hist"] = z["ppo"] - z["ppo_signal"]
    z["donchian_position20"] = (c - prior_low20) / (prior_high20 - prior_low20).replace(0, np.nan)
    z["distance_donchian_high20"] = c / prior_high20 - 1
    z["distance_donchian_low20"] = c / prior_low20 - 1
    # Normalize signed volume flow so it is comparable across price/volume regimes.
    for n in (5, 20):
        signed_flow = (direction * v).rolling(n, min_periods=n).sum()
        volume_total = v.rolling(n, min_periods=n).sum()
        z[f"obv_flow_ratio_{n}"] = signed_flow / volume_total.replace(0, np.nan)

    # Corporate-action windows invalidate share-volume-derived comparisons.
    # The cumulative OBV level is retained for audit but is not a stationary predictor.
    volume_derived = [
        "obv_change_5", "obv_flow_ratio_5", "obv_flow_ratio_20", "mfi14",
        "chaikin_ad_line", "chaikin_money_flow20", "volume_sma5_ratio",
        "volume_sma20_ratio", "volume_zscore20", "rolling_vwap_proxy_5",
        "rolling_vwap_proxy_10", "rolling_vwap_proxy_20",
        "distance_vwap_proxy_5", "distance_vwap_proxy_10", "distance_vwap_proxy_20",
    ]
    action_mask = z.get("corporate_action_window", pd.Series(False, index=z.index)).fillna(True).astype(bool)
    z.loc[action_mask, volume_derived] = np.nan

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
            "upper_wick", "lower_wick", "distance_prior", "bearish_", "bullish_",
            "rsi_divergence_", "macd_divergence_", "volume_divergence_", "price_breakout", "price_breakdown",
            "foreign_", "trust_", "dealer_", "institutional_", "broker_", "ema_", "distance_ema_", "stoch_", "cci", "williams_", "plus_di", "minus_di", "dx", "adx", "obv", "mfi", "chaikin_", "volume_sma", "volume_zscore", "rolling_vwap", "distance_vwap", "roc_", "ppo", "donchian_", "distance_donchian_",
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


def run(input_path: str, output_dir: str, corporate_actions_path: str | None = None,
        chip_data_path: str | None = None) -> dict:
    raw_bars = load_tsmc(input_path)
    bars = raw_bars.copy()
    action_events = pd.DataFrame()
    if corporate_actions_path:
        action_path = Path(corporate_actions_path)
        if not action_path.exists():
            raise FileNotFoundError(f"Corporate-action event file not found: {action_path}")
        action_events = pd.read_csv(action_path, dtype={"symbol": str})
        required_action_cols = {"symbol", "date", "adjustment_ratio", "conflict"}
        if not required_action_cols.issubset(action_events.columns):
            raise ValueError(f"Corporate-action file missing columns: {sorted(required_action_cols - set(action_events.columns))}")
        action_events = action_events.loc[action_events["symbol"].astype(str) == "2330"].copy()
        if action_events.get("conflict", pd.Series(dtype=bool)).fillna(True).astype(bool).any():
            raise ValueError("Conflicting 2330 corporate-action ratios; refusing to label pivots")
        action_events = action_events.loc[pd.to_numeric(action_events["adjustment_ratio"], errors="coerce").gt(0)].copy()
        adjusted = apply_back_adjustments(bars, action_events)
        for col in ("open", "high", "low", "close"):
            bars[f"raw_{col}"] = bars[col]
            bars[col] = adjusted[f"adj_{col}"]
        bars["corporate_action_window"] = adjusted["corporate_action_window"].astype(bool)
    chip_manifest = {"status": "not_supplied"}
    if chip_data_path:
        chip_path = Path(chip_data_path)
        if not chip_path.exists():
            raise FileNotFoundError(f"Chip-data file not found: {chip_path}")
        chip = pd.read_csv(chip_path)
        if "date" not in chip:
            raise ValueError("Chip-data file missing date")
        chip["date"] = pd.to_datetime(chip["date"], errors="coerce").dt.normalize()
        chip = chip.dropna(subset=["date"]).drop_duplicates("date", keep="last")
        chip_columns = [c for c in chip.columns if c != "date"]
        bars = bars.merge(chip, on="date", how="left", validate="one_to_one")
        chip_manifest = {
            "status": "merged",
            "rows": int(len(chip)),
            "columns": chip_columns,
            "institutional_available_rows": int(bars.get("institutional_data_available", pd.Series(0, index=bars.index)).fillna(0).sum()),
            "broker_available_rows": int(bars.get("broker_data_available", pd.Series(np.nan, index=bars.index)).notna().sum()),
            "broker_note": "Broker fields remain null on dates without licensed branch-level data; null is not zero trading."
        }
    features = build_causal_features(bars)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    event_frames, lead_frames = [], []
    for pct, atr_mult in THRESHOLDS:
        events = detect_confirmed_pivots(features, pct, atr_mult)
        if events.empty:
            continue
        action_mask = features["corporate_action_window"].fillna(True).astype(bool)
        pivot_action = action_mask.iloc[events["pivot_index"].astype(int).to_numpy()].to_numpy()
        confirm_action = action_mask.iloc[events["confirmation_index"].astype(int).to_numpy()].to_numpy()
        events = events.loc[~(pivot_action | confirm_action)].reset_index(drop=True)
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
        "price_basis": "point-in-time-safe back-adjusted OHLC for signal labels/features; raw OHLC retained in feature output for execution reference",
        "corporate_action_events": int(len(action_events)),
        "corporate_action_window_rows_excluded": int(features["corporate_action_window"].sum()),
        "corporate_action_adjustment_source": "FinMind normalized event ratios" if corporate_actions_path else "not supplied; unadjusted research is not final-grade",
        "chip_data": chip_manifest,
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
    parser.add_argument("--corporate-actions", default=None, help="Normalized 2330 corporate-action event CSV")
    parser.add_argument("--chip-data", default=None, help="Daily 2330 institutional and broker-branch feature CSV")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.corporate_actions, args.chip_data), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# Research workflow includes the chronological 40% CAGR / win-rate constrained rule-family audit.
