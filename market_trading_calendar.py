"""Market-specific trading calendar: TWSE JSON and TPEx official holiday page."""
from __future__ import annotations

from datetime import date, datetime
import re
import time
from io import StringIO

import pandas as pd
import requests

from taiwan_trading_calendar import trading_day_status as twse_trading_day_status

TPEX_HOLIDAY_URL = "https://www.tpex.org.tw/web/bulletin/trading_date/trading_date.php?l=zh-tw"
_cache = {}


def _closure_text(text: str) -> bool:
    compact = str(text).replace(" ", "")
    return any(word in compact for word in ("休市", "放假", "補假", "市場無交易", "停止交易"))


def parse_tpex_holiday_html(html: str, year: int) -> tuple[frozenset[str], str]:
    if not html or not str(html).strip():
        raise ValueError("TPEx 官方休市頁面內容為空")
    tables = pd.read_html(StringIO(html))
    closures = set()
    found_year = False
    for table in tables:
        for _, row in table.iterrows():
            cells = [str(value).strip() for value in row.tolist()]
            joined = " ".join(cells)
            match = re.search(r"(\d{3,4})[/-](\d{1,2})[/-](\d{1,2})", joined)
            if not match:
                continue
            y, month, day_num = map(int, match.groups())
            if y < 1911:
                y += 1911
            if y != int(year):
                continue
            found_year = True
            parsed = date(y, month, day_num)
            if parsed.weekday() < 5 and _closure_text(joined):
                closures.add(parsed.isoformat())
    if not found_year:
        raise ValueError(f"TPEx 官方頁面未找到 {year} 年日期")
    return frozenset(closures), f"TPEx 官方休市日期頁面 {year}"


def tpex_trading_day_status(day: date | datetime, *, fetcher=None) -> tuple[bool, str]:
    target = day.date() if isinstance(day, datetime) else day
    if target.weekday() >= 5:
        return False, "週末休市"
    key = target.year
    now = time.monotonic()
    cached = _cache.get(key)
    if cached and now - cached[0] < 6 * 60 * 60:
        closures, source = cached[1], cached[2]
    else:
        try:
            if fetcher is None:
                response = requests.get(TPEX_HOLIDAY_URL, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()
                html = response.text
            else:
                html = fetcher(target.year)
            closures, source = parse_tpex_holiday_html(html, target.year)
        except Exception as exc:
            return False, f"TPEx 官方交易日曆無法確認，停止委託：{exc}"
        _cache[key] = (now, closures, source)
    if target.isoformat() in closures:
        return False, f"TPEx 官方公告休市（資料年度 {target.year}）"
    return True, source


def market_trading_day_status(day: date | datetime, market: str) -> tuple[bool, str]:
    if market == "上市":
        return twse_trading_day_status(day)
    if market == "上櫃":
        return tpex_trading_day_status(day)
    return False, f"未知市場別 {market}，停止委託"


def clear_market_calendar_cache():
    _cache.clear()
