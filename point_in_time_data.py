"""Point-in-time fundamentals and ownership data helpers.

Raw provider rows are retained. A record is usable on/after its publication
availability date, never merely because its fiscal period has ended.
"""
from __future__ import annotations

from typing import Iterable
import pandas as pd

DATE_CANDIDATES = ("available_date", "announcement_date", "publish_date",
                   "announced_date", "資料發布日", "公告日期", "發布日期")
PERIOD_CANDIDATES = ("period", "date", "營業年月", "財報年月", "資料年月")
SYMBOL_CANDIDATES = ("stock_id", "symbol", "代號", "證券代號")


def _first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((name for name in candidates if name in frame.columns), None)


def normalize_point_in_time_records(records, *, symbol: str | None = None,
                                    source: str = "unknown") -> pd.DataFrame:
    """Normalize provider data and require a genuine public-availability date."""
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "period", "available_date", "source"])
    symbol_col = _first_column(frame, SYMBOL_CANDIDATES)
    period_col = _first_column(frame, PERIOD_CANDIDATES)
    available_col = _first_column(frame, DATE_CANDIDATES)
    if available_col is None:
        raise ValueError("缺少公告／發布日期；不得以財報所屬期間代替可得日期")
    out = frame.copy()
    out["symbol"] = out[symbol_col].astype(str) if symbol_col else str(symbol or "")
    if not symbol and symbol_col is None:
        raise ValueError("缺少股票代號")
    out["period"] = pd.to_datetime(out[period_col], errors="coerce") if period_col else pd.NaT
    out["available_date"] = pd.to_datetime(out[available_col], errors="coerce")
    out["source"] = str(source)
    out = out.dropna(subset=["available_date"])
    out["available_date"] = out["available_date"].dt.normalize()
    if period_col:
        out["period"] = out["period"].dt.normalize()
    return out.sort_values(["symbol", "available_date"]).reset_index(drop=True)


def point_in_time_snapshot(records: pd.DataFrame, as_of) -> pd.DataFrame:
    """Latest published row per symbol that was knowable at the requested date."""
    if records.empty:
        return records.copy()
    if "available_date" not in records.columns or "symbol" not in records.columns:
        raise ValueError("資料必須先經 normalize_point_in_time_records 正規化")
    cutoff = pd.Timestamp(as_of).normalize()
    eligible = records.loc[pd.to_datetime(records["available_date"]) <= cutoff].copy()
    if eligible.empty:
        return eligible
    return (eligible.sort_values(["symbol", "available_date"])
            .groupby("symbol", as_index=False, sort=False).tail(1)
            .sort_values("symbol").reset_index(drop=True))


def assert_no_future_records(records: pd.DataFrame, as_of) -> None:
    if records.empty:
        return
    cutoff = pd.Timestamp(as_of).normalize()
    dates = pd.to_datetime(records["available_date"], errors="coerce")
    if (dates > cutoff).any():
        raise ValueError("資料包含回測時點之後才公開的資訊")


def fetch_finmind_dataset(dataset: str, symbol: str, start_date: str,
                          end_date: str | None = None, token: str = "",
                          timeout: int = 20) -> pd.DataFrame:
    """Fetch a named FinMind dataset without guessing its publication dates."""
    import requests
    params = {"dataset": dataset, "data_id": str(symbol), "start_date": start_date}
    if end_date:
        params["end_date"] = end_date
    if token:
        params["token"] = token
    try:
        response = requests.get("https://api.finmindtrade.com/api/v4/data",
                                params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"FinMind {dataset} 請求失敗：{exc}") from exc
    if not isinstance(payload, dict) or str(payload.get("status", "200")) not in {"200", "success", "Success"}:
        detail = payload.get("msg", "未知 API 錯誤") if isinstance(payload, dict) else "回傳格式錯誤"
        raise RuntimeError(f"FinMind {dataset} API 錯誤：{detail}")
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"FinMind {dataset} 回傳資料格式錯誤")
    return pd.DataFrame(rows)


def fetch_point_in_time_fundamentals(symbol: str, start_date: str,
                                     token: str = "") -> pd.DataFrame:
    """Fetch statements and monthly revenue; reject rows without announcement dates."""
    pieces = []
    for dataset in ("TaiwanStockFinancialStatements", "TaiwanStockMonthRevenue"):
        raw = fetch_finmind_dataset(dataset, symbol, start_date, token=token)
        if raw.empty:
            continue
        pieces.append(normalize_point_in_time_records(raw, symbol=symbol, source=dataset))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["symbol", "period", "available_date", "source"]
    )
