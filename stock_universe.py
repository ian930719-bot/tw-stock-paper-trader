"""Current Taiwan listed and OTC security master from FinMind TaiwanStockInfo.

The source includes TWSE, TPEx and emerging-board rows. A security can have
multiple historical market rows after a transfer; always keep the newest row
per stock_id before filtering to twse/tpex.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

API_URL = "https://api.finmindtrade.com/api/v4/data"
TAIPEI = ZoneInfo("Asia/Taipei")
REQUIRED_COLUMNS = {"stock_id", "stock_name", "industry_category", "type", "date"}


class StockUniverseError(RuntimeError):
    """Raised when the stock master cannot be fetched or validated."""


def normalize_stock_universe(records) -> pd.DataFrame:
    """Return one latest active TWSE/TPEx record per valid listed security."""
    if isinstance(records, pd.DataFrame):
        raw = records.copy()
    else:
        raw = pd.DataFrame(records)
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise StockUniverseError("股票主檔缺少必要欄位：" + ", ".join(sorted(missing)))
    if raw.empty:
        raise StockUniverseError("股票主檔沒有資料")

    raw = raw[list(REQUIRED_COLUMNS)].copy()
    raw["stock_id"] = raw["stock_id"].astype(str).str.strip().str.upper()
    raw["stock_name"] = raw["stock_name"].astype(str).str.strip()
    raw["industry_category"] = raw["industry_category"].fillna("未分類").astype(str).str.strip()
    raw["type"] = raw["type"].astype(str).str.strip().str.lower()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"])
    raw = raw[raw["stock_id"].str.fullmatch(r"\d{4,6}", na=False)]
    raw = raw[raw["stock_name"].ne("") & raw["stock_name"].ne("nan")]
    if raw.empty:
        raise StockUniverseError("股票主檔沒有可用的證券代號")

    # Transfer history is retained by the source; the newest row defines the
    # current market. Do not filter market type before selecting this row.
    market_rank = {"twse": 0, "tpex": 1, "emerging": 2}
    raw["_market_rank"] = raw["type"].map(market_rank).fillna(9)
    raw = raw.sort_values(
        ["stock_id", "date", "_market_rank"], ascending=[True, False, True],
        kind="mergesort",
    ).drop_duplicates("stock_id", keep="first")
    raw = raw[raw["type"].isin(["twse", "tpex"])]
    raw = raw[~raw["industry_category"].str.casefold().isin(
        {"大盤", "index", "所有證券"}
    )]
    raw = raw[~raw["stock_id"].isin({"TAIEX", "TPEX"})]
    if raw.empty:
        raise StockUniverseError("最新主檔沒有符合條件的上市／上櫃證券")

    result = raw.rename(columns={
        "stock_id": "代號",
        "stock_name": "名稱",
        "industry_category": "產業",
        "date": "市場資料日期",
    }).copy()
    result["市場"] = result["type"].map({"twse": "上市", "tpex": "上櫃"})
    result["市場資料日期"] = result["市場資料日期"].dt.strftime("%Y-%m-%d")
    result = result[["代號", "名稱", "市場", "產業", "市場資料日期"]]
    return result.sort_values(["市場", "代號"]).reset_index(drop=True)


def fetch_stock_universe(token: str | None = None, timeout: int = 20) -> pd.DataFrame:
    """Fetch the all-market security master; token is optional for free dataset."""
    api_token = (token or os.getenv("FINMIND_TOKEN", "")).strip()
    headers = {"User-Agent": "tw-stock-paper-trader/1.0"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    try:
        response = requests.get(
            API_URL, params={"dataset": "TaiwanStockInfo"},
            headers=headers, timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise StockUniverseError(f"無法取得 FinMind 股票主檔：{exc}") from exc

    if not isinstance(payload, dict):
        raise StockUniverseError("FinMind 回傳格式不是 JSON 物件")
    if str(payload.get("status", "200")) not in {"200", "success", "Success"}:
        raise StockUniverseError(
            f"FinMind API 回應錯誤：{payload.get('msg') or payload.get('message') or payload.get('status')}"
        )
    records = payload.get("data")
    if not isinstance(records, list) or not records:
        raise StockUniverseError("FinMind API 未回傳股票主檔資料")
    return normalize_stock_universe(records)


def universe_summary(universe: pd.DataFrame) -> dict:
    """Return compact counts and source timestamp for the UI."""
    if universe is None or universe.empty:
        return {"總數": 0, "上市": 0, "上櫃": 0, "資料日期": None, "更新時間": None}
    dates = pd.to_datetime(universe["市場資料日期"], errors="coerce").dropna()
    return {
        "總數": int(len(universe)),
        "上市": int(universe["市場"].eq("上市").sum()),
        "上櫃": int(universe["市場"].eq("上櫃").sum()),
        "資料日期": dates.max().strftime("%Y-%m-%d") if not dates.empty else None,
        "更新時間": datetime.now(TAIPEI).strftime("%Y-%m-%d %H:%M:%S"),
    }
