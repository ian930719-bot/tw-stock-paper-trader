"""Small, testable helpers for market-data freshness labels."""
from datetime import datetime, time
from zoneinfo import ZoneInfo

TAIPEI = ZoneInfo("Asia/Taipei")


def quote_freshness(quote_time, now=None, stale_after_minutes=20):
    """Return a conservative freshness label and age in minutes.

    Yahoo intraday timestamps are informational only. Outside Taiwan's regular
    weekday session, today's last quote is labelled as a closing/off-session
    quote rather than being incorrectly called stale.
    """
    if not quote_time or quote_time == "無報價":
        return "無報價", None

    try:
        stamp = quote_time if isinstance(quote_time, datetime) else datetime.fromisoformat(str(quote_time))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=TAIPEI)
        stamp = stamp.astimezone(TAIPEI)
    except (TypeError, ValueError, OverflowError):
        return "時間異常", None

    current = (now or datetime.now(TAIPEI))
    if current.tzinfo is None:
        current = current.replace(tzinfo=TAIPEI)
    current = current.astimezone(TAIPEI)
    age = (current - stamp).total_seconds() / 60.0

    if age < -2:
        return "時間異常", round(age, 1)

    same_day = stamp.date() == current.date()
    in_regular_session = (
        current.weekday() < 5
        and time(9, 0) <= current.time().replace(tzinfo=None) <= time(13, 30)
    )

    if not same_day:
        return "非當日報價", round(age, 1)

    if in_regular_session:
        if age <= stale_after_minutes:
            return "盤中更新", round(max(age, 0.0), 1)
        return "盤中延遲／可能休市", round(age, 1)

    return "當日最後報價（非盤中）", round(max(age, 0.0), 1)
