"""Cross-sectional scoring for latest fundamental snapshots.

Snapshot scores are descriptive only and MUST NOT be used in historical
backtests unless point-in-time historical fundamentals are supplied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


FACTOR_DIRECTIONS = {
    "ROE": 1,
    "營收成長率": 1,
    "獲利成長率": 1,
    "自由現金流": 1,
    "毛利率": 1,
    "營業利益率": 1,
    "負債／權益": -1,
    "本益比": -1,
    "股價淨值比": -1,
}


def score_fundamental_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank supplied latest-snapshot metrics cross-sectionally, with coverage."""
    result = frame.copy()
    if result.empty:
        result["基本面分數"] = pd.Series(dtype=float)
        result["基本面資料完整度"] = pd.Series(dtype=float)
        return result
    available = [name for name in FACTOR_DIRECTIONS if name in result.columns]
    if not available:
        result["基本面分數"] = np.nan
        result["基本面資料完整度"] = 0.0
        return result
    ranks = []
    for name in available:
        values = pd.to_numeric(result[name], errors="coerce")
        # Percentile ranks are computed only among securities with real values.
        rank = values.rank(pct=True, method="average")
        if FACTOR_DIRECTIONS[name] < 0:
            rank = 1.0 - rank
        ranks.append(rank.rename(name))
    rank_frame = pd.concat(ranks, axis=1)
    result["基本面資料完整度"] = rank_frame.notna().mean(axis=1)
    result["基本面分數"] = rank_frame.mean(axis=1, skipna=True) * 100.0
    result.loc[result["基本面資料完整度"] == 0, "基本面分數"] = np.nan
    result["基本面排名"] = result["基本面分數"].rank(ascending=False, method="min")
    return result
