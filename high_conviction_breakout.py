"""High-conviction breakout candidate for aggressive paper research.

Signals use only completed daily bars and are executed by the shared portfolio
engine at the following session open. This is an experimental, high-risk rule
set, not a return guarantee.
"""
from __future__ import annotations

from typing import Mapping
import numpy as np
import pandas as pd
from corporate_actions import point_in_time_recalculate


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    z = frame.sort_values("date").reset_index(drop=True).copy()
    close = pd.to_numeric(z["close"], errors="coerce")
    high = pd.to_numeric(z["high"], errors="coerce")
    low = pd.to_numeric(z["low"], errors="coerce")
    volume = pd.to_numeric(z["volume"], errors="coerce")
    z["mom_126"] = close.pct_change(126)
    z["mom_63"] = close.pct_change(63)
    z["sma_50"] = close.rolling(50, min_periods=50).mean()
    z["sma_200"] = close.rolling(200, min_periods=200).mean()
    z["ema_21"] = close.ewm(span=21, adjust=False, min_periods=21).mean()
    z["prior_high_55"] = high.shift(1).rolling(55, min_periods=55).max()
    z["prior_low_20"] = low.shift(1).rolling(20, min_periods=20).min()
    z["volume_avg_20"] = volume.shift(1).rolling(20, min_periods=20).mean()
    return z[["date", "close", "volume", "mom_126", "mom_63", "sma_50",
              "sma_200", "ema_21", "prior_high_55", "prior_low_20",
              "volume_avg_20"]]


def build_high_conviction_breakout_overrides(
    frames: Mapping[str, pd.DataFrame],
    *,
    corporate_actions=None,
    relative_strength_quantile: float = 0.80,
    minimum_universe: int = 15,
    volume_multiple: float = 1.50,
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Buy fresh 55-session breakouts among the strongest trending stocks."""
    if not 0.5 < float(relative_strength_quantile) < 1.0:
        raise ValueError("relative_strength_quantile 必須介於 0.5 與 1")
    if int(minimum_universe) < 3:
        raise ValueError("minimum_universe 至少為 3")
    if not np.isfinite(volume_multiple) or float(volume_multiple) < 1.0:
        raise ValueError("volume_multiple 不得低於 1")

    rows = []
    for code, frame in frames.items():
        if frame is None or frame.empty:
            continue
        events = (corporate_actions or {}).get(str(code))
        adjusted = point_in_time_recalculate(frame, events, _features)
        part = adjusted.copy()
        part["symbol"] = str(code)
        rows.append(part)
    if not rows:
        return {}

    panel = pd.concat(rows, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    panel["momentum_rank"] = panel.groupby("date")["mom_126"].rank(pct=True)
    panel["universe_count"] = panel.groupby("date")["symbol"].transform("nunique")

    outputs = {}
    for code, part in panel.groupby("symbol", sort=False):
        part = part.sort_values("date")
        valid = part["universe_count"] >= int(minimum_universe)
        trend = (
            (part["close"] > part["sma_50"])
            & (part["sma_50"] > part["sma_200"])
            & (part["mom_126"] > 0)
        )
        breakout = (
            (part["close"] > part["prior_high_55"])
            & (part["volume"] >= float(volume_multiple) * part["volume_avg_20"])
        )
        entry = (
            valid & trend
            & (part["momentum_rank"] >= float(relative_strength_quantile))
            & breakout
        ).fillna(False)
        exit_signal = (
            (~valid)
            | (part["close"] < part["ema_21"])
            | (part["close"] < part["prior_low_20"])
            | (part["mom_63"] < 0)
        ).fillna(False)
        outputs[str(code)] = (
            pd.Series(entry.to_numpy(dtype=bool)),
            pd.Series(exit_signal.to_numpy(dtype=bool)),
        )
    return outputs
