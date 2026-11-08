"""Cross-sectional momentum rotation using only information known at each close."""
from __future__ import annotations

from typing import Mapping
import numpy as np
import pandas as pd
from corporate_actions import point_in_time_recalculate


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    z = frame.sort_values("date").reset_index(drop=True).copy()
    close = pd.to_numeric(z["close"], errors="coerce")
    returns = close.pct_change()
    z["mom_63"] = close.pct_change(63)
    z["mom_126"] = close.pct_change(126)
    z["mom_252"] = close.pct_change(252)
    z["vol_63"] = returns.rolling(63, min_periods=50).std(ddof=1) * np.sqrt(252)
    z["sma_50"] = close.rolling(50, min_periods=50).mean()
    z["sma_100"] = close.rolling(100, min_periods=100).mean()
    z["sma_200"] = close.rolling(200, min_periods=200).mean()
    return z[["date", "close", "mom_63", "mom_126", "mom_252",
              "vol_63", "sma_50", "sma_100", "sma_200"]]


def build_cross_sectional_overrides(
    frames: Mapping[str, pd.DataFrame],
    *,
    corporate_actions=None,
    entry_quantile: float = 0.80,
    exit_quantile: float = 0.45,
    minimum_universe: int = 15,
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Rank 3/6/12-month momentum across stocks at each date."""
    if not 0.5 < float(entry_quantile) < 1.0:
        raise ValueError("entry_quantile 必須介於 0.5 與 1")
    if not 0.0 < float(exit_quantile) < float(entry_quantile):
        raise ValueError("exit_quantile 必須低於 entry_quantile 且大於 0")
    if int(minimum_universe) < 3:
        raise ValueError("minimum_universe 至少為 3")

    feature_rows = []
    for code, frame in frames.items():
        if frame is None or frame.empty:
            continue
        events = (corporate_actions or {}).get(str(code))
        adjusted = point_in_time_recalculate(frame, events, _features)
        part = adjusted.copy()
        part["symbol"] = str(code)
        feature_rows.append(part)
    if not feature_rows:
        return {}

    panel = pd.concat(feature_rows, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    for source, target in (
        ("mom_63", "r63"), ("mom_126", "r126"),
        ("mom_252", "r252"), ("vol_63", "rvol"),
    ):
        panel[target] = panel.groupby("date")[source].rank(pct=True)
    panel["relative_score"] = (
        0.20 * panel["r63"] + 0.35 * panel["r126"]
        + 0.45 * panel["r252"] - 0.15 * panel["rvol"]
    )
    panel["relative_rank"] = panel.groupby("date")["relative_score"].rank(pct=True)
    panel["universe_count"] = panel.groupby("date")["symbol"].transform("nunique")

    outputs = {}
    for code, part in panel.groupby("symbol", sort=False):
        part = part.sort_values("date")
        valid = part["universe_count"] >= int(minimum_universe)
        uptrend = (
            (part["close"] > part["sma_200"])
            & (part["sma_50"] > part["sma_200"])
            & (part["mom_126"] > 0)
        )
        entry = (
            valid & uptrend & (part["relative_rank"] >= float(entry_quantile))
        ).fillna(False)
        exit_signal = (
            (~valid)
            | (part["relative_rank"] < float(exit_quantile))
            | (part["close"] < part["sma_100"])
            | (part["mom_63"] < 0)
        ).fillna(False)
        outputs[str(code)] = (
            pd.Series(entry.to_numpy(dtype=bool)),
            pd.Series(exit_signal.to_numpy(dtype=bool)),
        )
    return outputs
