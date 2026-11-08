"""Taiwan historical fundamentals and chip data adapters.

Trading-date datasets become knowable no earlier than the next calendar day.
Monthly revenue create_time is only FinMind ingestion time (available from
2026-04-21); it is explicitly marked approximate and never treated as the
official MOPS announcement timestamp. Financial statements' fiscal date alone
is NOT a publication date and therefore cannot be used for point-in-time tests.
"""
from __future__ import annotations

from datetime import date, timedelta
import pandas as pd

CHIP_DATASETS = {
    "institutional": "TaiwanStockInstitutionalInvestorsBuySell",
    "margin_short": "TaiwanStockMarginPurchaseShortSale",
    "securities_lending": "TaiwanStockSecuritiesLending",
}
HISTORICAL_DATASETS = {
    "monthly_revenue": "TaiwanStockMonthRevenue",
    "income_statement": "TaiwanStockFinancialStatements",
    "balance_sheet": "TaiwanStockBalanceSheet",
    "cash_flow": "TaiwanStockCashFlowsStatement",
}


def _next_calendar_day(value):
    return (pd.Timestamp(value).normalize() + pd.Timedelta(days=1))


def add_conservative_availability(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Attach a conservative availability date without inventing publication dates."""
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    z = frame.copy()
    if "date" not in z:
        raise ValueError(f"{dataset} 缺少資料日期")
    z["date"] = pd.to_datetime(z["date"], errors="coerce")
    z = z.dropna(subset=["date"]).copy()
    if dataset in CHIP_DATASETS.values():
        z["available_date"] = z["date"] + pd.Timedelta(days=1)
        z["availability_quality"] = "保守估計：交易日後一個日曆日可得"
    elif dataset == HISTORICAL_DATASETS["monthly_revenue"]:
        if "create_time" not in z:
            z["create_time"] = None
        created = pd.to_datetime(z["create_time"], errors="coerce")
        eligible = created.notna() & (created >= pd.Timestamp("2026-04-21"))
        z["available_date"] = created.where(eligible)
        z["availability_quality"] = "FinMind資料庫寫入日近似值；非官方公告時間"
    else:
        z["available_date"] = pd.NaT
        z["availability_quality"] = "僅財報所屬期間；缺正式公告日，不可用於歷史回測"
    return z.sort_values("date").reset_index(drop=True)


def summarize_institutional(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    required = {"date", "name", "buy", "sell"}
    if not required.issubset(frame.columns):
        raise ValueError(f"法人資料缺少欄位：{sorted(required - set(frame.columns))}")
    z = frame.copy()
    z["買賣超"] = pd.to_numeric(z["buy"], errors="coerce") - pd.to_numeric(z["sell"], errors="coerce")
    z["法人"] = z["name"].astype(str)
    return z.pivot_table(index="date", columns="法人", values="買賣超", aggfunc="sum").sort_index()


def enrich_industry_comparison(fundamentals: pd.DataFrame, stock_master: pd.DataFrame) -> pd.DataFrame:
    """Attach industry and rank latest snapshot only; no historical backtest use."""
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame()
    if stock_master is None or stock_master.empty:
        out = fundamentals.copy()
        out["產業"] = "未知"
        return out
    master = stock_master.copy()
    code_col = next((c for c in ("代號", "stock_id", "symbol") if c in master), None)
    industry_col = next((c for c in ("產業", "industry_category", "industry") if c in master), None)
    if not code_col or not industry_col:
        raise ValueError("股票主檔缺少代號或產業欄位")
    master[code_col] = master[code_col].astype(str)
    out = fundamentals.copy()
    fund_code = next((c for c in ("代號", "stock_id", "symbol") if c in out), None)
    if not fund_code:
        raise ValueError("基本面資料缺少股票代號")
    out[fund_code] = out[fund_code].astype(str)
    out = out.merge(master[[code_col, industry_col]].drop_duplicates(code_col),
                    left_on=fund_code, right_on=code_col, how="left")
    out["產業"] = out[industry_col].fillna("未知")
    score_col = next((c for c in ("基本面分數", "factor_score", "score") if c in out), None)
    if score_col:
        out["產業內排名"] = out.groupby("產業")[score_col].rank(ascending=False, method="min")
        out["產業樣本數"] = out.groupby("產業")[score_col].transform("count")
    return out


def fetch_finmind_dataset(dataset: str, symbol: str, start_date: str,
                          end_date: str | None = None, token: str = "",
                          timeout: int = 20) -> pd.DataFrame:
    import requests
    params = {"dataset": dataset, "data_id": str(symbol), "start_date": start_date}
    if end_date:
        params["end_date"] = end_date
    if token:
        params["token"] = token
    response = requests.get("https://api.finmindtrade.com/api/v4/data",
                            params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or str(payload.get("status", "200")) not in {"200", "success", "Success"}:
        raise RuntimeError(f"FinMind {dataset} API 回傳錯誤：{payload.get('msg', '未知錯誤') if isinstance(payload, dict) else '格式錯誤'}")
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"FinMind {dataset} 回傳格式錯誤")
    return pd.DataFrame(rows)


def load_symbol_history(symbol: str, start_date: str, token: str = "") -> dict:
    """Load four chip/fundamental series; return per-dataset errors instead of hiding gaps."""
    result, errors = {}, {}
    datasets = {**CHIP_DATASETS, **HISTORICAL_DATASETS}
    for key, dataset in datasets.items():
        try:
            raw = fetch_finmind_dataset(dataset, symbol, start_date, token=token)
            result[key] = add_conservative_availability(raw, dataset)
        except Exception as exc:
            errors[key] = str(exc)
    return {"data": result, "errors": errors, "source": "FinMind API v4",
            "loaded_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
            "symbol": str(symbol), "start_date": start_date}


def point_in_time_revenue(frame: pd.DataFrame, as_of) -> pd.DataFrame:
    """Return only monthly revenue rows with recorded ingestion date known by as_of."""
    if frame is None or frame.empty or "available_date" not in frame:
        return pd.DataFrame() if frame is None else frame.iloc[0:0].copy()
    cutoff = pd.Timestamp(as_of).normalize()
    z = frame.loc[pd.to_datetime(frame["available_date"], errors="coerce") <= cutoff].copy()
    z = z.dropna(subset=["available_date"])
    return z.sort_values("available_date").reset_index(drop=True)


def next_start_date(years: int = 5) -> str:
    if years < 1 or years > 20:
        raise ValueError("years 必須介於 1 至 20")
    return (date.today() - timedelta(days=365 * years)).isoformat()
