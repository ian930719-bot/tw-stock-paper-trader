"""Official TWSE annual holiday calendar with bounded in-memory caching.

The endpoint lists special dates for the current calendar year. Weekends are
always closed; weekday closure rows are identified from the exchange's own
description (including settlement-only/no-trading and make-up holidays).
If the official schedule cannot be loaded, callers receive an unavailable
status and should fail closed for paper order creation.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from threading import RLock
import time

import requests

API_URL = "https://www.twse.com.tw/rwd/zh/holidaySchedule/holidaySchedule"
CACHE_TTL_SECONDS = 6 * 60 * 60
_lock = RLock()
_cache: dict[int, tuple[float, frozenset[str], str]] = {}


class TradingCalendarUnavailable(RuntimeError):
    """Raised when the official TWSE calendar cannot be verified."""


def _is_closure_row(name: str, description: str) -> bool:
    text = f"{name} {description}".replace(" ", "")
    closure_markers = (
        "市場無交易",
        "休市",
        "放假",
        "補假",
        "停止交易",
    )
    return any(marker in text for marker in closure_markers)


def parse_calendar_payload(payload: dict, year: int) -> tuple[frozenset[str], str]:
    if not isinstance(payload, dict) or str(payload.get("stat", "")).lower() != "ok":
        raise TradingCalendarUnavailable("TWSE 未回傳有效的開休市日期資料")
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise TradingCalendarUnavailable("TWSE 開休市日期格式錯誤")
    closures: set[str] = set()
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        raw_date, name, description = (str(row[0]).strip(), str(row[1]), str(row[2]))
        try:
            parsed = date.fromisoformat(raw_date)
        except ValueError:
            continue
        if parsed.year != int(year):
            continue
        if parsed.weekday() < 5 and _is_closure_row(name, description):
            closures.add(parsed.isoformat())
    source_date = str(payload.get("date") or "")
    return frozenset(closures), source_date


def get_official_closures(year: int, *, force_refresh: bool = False,
                          timeout: int = 12, now: float | None = None) -> tuple[frozenset[str], str]:
    current = time.monotonic() if now is None else float(now)
    with _lock:
        cached = _cache.get(int(year))
        if not force_refresh and cached and current - cached[0] < CACHE_TTL_SECONDS:
            return cached[1], cached[2]
    try:
        response = requests.get(API_URL, params={"response": "json"}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        closures, source_date = parse_calendar_payload(payload, int(year))
    except Exception as exc:
        raise TradingCalendarUnavailable(f"無法取得 TWSE 官方開休市日曆：{exc}") from exc
    with _lock:
        _cache[int(year)] = (current, closures, source_date)
    return closures, source_date


def trading_day_status(day: date | datetime, *, fetcher=get_official_closures) -> tuple[bool, str]:
    target = day.date() if isinstance(day, datetime) else day
    if target.weekday() >= 5:
        return False, "週末休市"
    try:
        closures, source_date = fetcher(target.year)
    except Exception as exc:
        return False, f"官方交易日曆無法確認，停止委託：{exc}"
    if target.isoformat() in closures:
        return False, f"TWSE 官方公告休市（資料年度 {target.year}）"
    return True, f"TWSE 官方日曆（資料標記 {source_date or '未提供'}）"


def clear_calendar_cache() -> None:
    with _lock:
        _cache.clear()
