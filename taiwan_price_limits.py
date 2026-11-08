"""Official daily price bands for Taiwan listed and OTC securities.

The module consumes exchange-published reference/limit prices. It never
estimates a price band from yesterday's close. Missing, stale, special or
unrecognized rows fail closed when the caller requires a verified band.
"""
from __future__ import annotations

from datetime import date, datetime
from threading import RLock
import time

import requests

TWSE_URL = "https://openapi.twse.com.tw/v1/exchangeReport/TWT84U"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
CACHE_TTL_SECONDS = 60
_lock = RLock()
_cache: dict[tuple[str, str], tuple[float, dict]] = {}


class PriceLimitUnavailable(ValueError):
    """Official daily price limits could not be verified."""


def _clean(value):
    return str(value).strip().replace(",", "").replace(" ", "")


def _number(row, names):
    for name in names:
        if name in row:
            raw = _clean(row[name])
            if raw and raw not in {"-", "--", "N/A", "None"}:
                try:
                    return float(raw)
                except ValueError:
                    pass
    return None


def _date_iso(value):
    raw = _clean(value)
    if not raw:
        return None
    # Parse ROC compact dates before strptime: 1151109 is a 7-digit ROC date,
    # not a Gregorian year/month/day string.
    try:
        if len(raw) == 7 and raw.isdigit():
            return date(int(raw[:3]) + 1911, int(raw[3:5]), int(raw[5:])).isoformat()
        if "/" in raw:
            parts = raw.split("/")
            if len(parts[0]) == 3:
                return date(int(parts[0]) + 1911, int(parts[1]), int(parts[2])).isoformat()
    except (ValueError, TypeError):
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _find_code(row):
    for key in ("證券代號", "股票代號", "SecuritiesCompanyCode", "SecurityCode",
                "Code", "StockNo", "股票代碼"):
        if key in row and _clean(row[key]):
            return _clean(row[key]).upper()
    return None


def _parse_rows(rows, symbol, expected_date, market):
    if not isinstance(rows, list):
        raise PriceLimitUnavailable("交易所回傳格式不是資料列")
    code_aliases = ("證券代號", "股票代號", "SecuritiesCompanyCode", "SecurityCode",
                    "Code", "StockNo", "股票代碼")
    date_aliases = ("資料日期", "日期", "Date", "TradingDate", "交易日期")
    ref_aliases = ("參考價", "當日參考價", "今日參考價", "次日參考價",
                   "ReferencePrice", "TradingReferencePrice", "NextDayReferencePrice", "NextReferencePrice", "本日開盤競價基準")
    up_aliases = ("漲停價", "次日漲停價", "漲停價格", "UpperLimitPrice",
                  "LimitUpPrice", "NextDayLimitUpPrice", "HighLimitPrice", "NextLimitUp", "NextLimitUpPrice", "本日漲停價", "次日漲停價")
    down_aliases = ("跌停價", "次日跌停價", "跌停價格", "LowerLimitPrice",
                    "LimitDownPrice", "NextDayLimitDownPrice", "LowLimitPrice", "NextLimitDown", "NextLimitDownPrice", "本日跌停價", "次日跌停價")
    matches = [row for row in rows if isinstance(row, dict) and _find_code(row) == str(symbol).upper()]
    if not matches:
        raise PriceLimitUnavailable(f"{market} 官方資料找不到證券 {symbol}")
    for row in matches:
        row_date = next((_date_iso(row[k]) for k in date_aliases if k in row), None)
        ref = _number(row, ref_aliases)
        upper = _number(row, up_aliases)
        lower = _number(row, down_aliases)
        # TPEX daily close feed explicitly publishes next-session reference and limits.
        # A source date must match the expected trading date to prevent stale fills.
        if market == "上櫃":
            # TPEx labels these as next-session limits. During today's session,
            # the latest completed daily row must therefore be from a prior day.
            if not row_date or row_date >= expected_date.isoformat():
                continue
            try:
                if (expected_date - date.fromisoformat(row_date)).days > 4:
                    continue
            except ValueError:
                continue
        elif row_date and row_date != expected_date.isoformat():
            continue
        if ref is None or upper is None or lower is None:
            continue
        if ref <= 0 or upper <= 0 or lower <= 0 or lower > upper:
            continue
        # 9999.95 / 0.01 is the exchange's sentinel for no price limit.
        no_limit = upper >= 9999.9 and lower <= 0.02
        return {
            "symbol": str(symbol).upper(),
            "market": market,
            "date": expected_date.isoformat(),
            "source_date": row_date,
            "reference_price": ref,
            "upper_limit": None if no_limit else upper,
            "lower_limit": None if no_limit else lower,
            "no_price_limit": no_limit,
            "source": "TWSE OpenAPI TWT84U" if market == "上市" else "TPEx OpenAPI daily close quotes",
        }
    raise PriceLimitUnavailable(
        f"{market} 官方資料未提供 {symbol} 可核對的參考價及漲跌停價；可能為特殊商品或無漲跌幅標的"
    )


def get_official_price_limit(symbol: str, market: str, *, expected_date: date | None = None,
                             force_refresh: bool = False, timeout: int = 15,
                             fetcher=None) -> dict:
    market = str(market).strip()
    if market not in {"上市", "上櫃"}:
        raise PriceLimitUnavailable(f"不支援的市場別：{market}")
    target = expected_date or datetime.now().date()
    key = (market, str(symbol).upper())
    now = time.monotonic()
    with _lock:
        cached = _cache.get(key)
        if not force_refresh and cached and now - cached[0] < CACHE_TTL_SECONDS:
            if cached[1]["date"] == target.isoformat():
                return dict(cached[1])
    if fetcher is None:
        url = TWSE_URL if market == "上市" else TPEX_URL
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            rows = response.json()
        except Exception as exc:
            raise PriceLimitUnavailable(f"無法取得 {market} 官方漲跌停資料：{exc}") from exc
    else:
        rows = fetcher(market)
    result = _parse_rows(rows, symbol, target, market)
    with _lock:
        _cache[key] = (now, dict(result))
    return result


def validate_limit_price(limit_price: float, band: dict) -> None:
    price = float(limit_price)
    if not band.get("no_price_limit"):
        lower, upper = band.get("lower_limit"), band.get("upper_limit")
        if lower is None or upper is None:
            raise PriceLimitUnavailable("官方漲跌停區間缺漏，拒絕委託")
        if price < float(lower) - 1e-8 or price > float(upper) + 1e-8:
            raise ValueError(
                f"委託價格超出官方漲跌停範圍：{float(lower):g} 至 {float(upper):g} 元"
            )


def clear_price_limit_cache():
    with _lock:
        _cache.clear()
