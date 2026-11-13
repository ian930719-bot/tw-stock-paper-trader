#!/usr/bin/env python3
"""Fetch and audit official TWSE TWT49U ex-right/ex-dividend reference events for UMC 2303."""
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

OUT = Path("research/umc2303/output")
RAW = OUT / "umc2303_twse_daily_raw.csv"
URL = "https://www.twse.com.tw/exchangeReport/TWT49U"
SYMBOL = "2303"


def number(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def iso_date(value):
    value = str(value).strip().replace("-", "/")
    parts = value.split("/")
    if len(parts) != 3:
        return None
    year, month, day = map(int, parts)
    if year < 1911:
        year += 1911
    return f"{year:04d}-{month:02d}-{day:02d}"


def fetch_year(year):
    params = urlencode({
        "response": "json",
        "strDate": f"{year}0101",
        "endDate": f"{year}1231",
    })
    request = Request(
        URL + "?" + params,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"TWT49U {year}: invalid JSON response")
    fields = payload.get("fields") or []
    data = payload.get("data") or []
    if not fields and not data:
        stat = str(payload.get("stat", ""))
        if stat and stat not in {"查無資料", "無符合條件資料"}:
            raise RuntimeError(f"TWT49U {year}: {stat}")
        return []
    records = []
    for values in data:
        if len(values) != len(fields):
            raise RuntimeError(f"TWT49U {year}: field/data width mismatch")
        row = dict(zip(fields, values))
        if str(row.get("股票代號", "")).strip() != SYMBOL:
            continue
        day = iso_date(row.get("資料日期"))
        previous = number(row.get("除權息前收盤價"))
        reference = number(row.get("除權息參考價"))
        combined = number(row.get("權值+息值"))
        if not day or not previous or not reference or previous <= 0 or reference <= 0:
            raise RuntimeError(f"TWT49U {year}: invalid UMC event row {row}")
        records.append({
            "date": day,
            "symbol": SYMBOL,
            "name": str(row.get("股票名稱", "聯電")),
            "previous_close": previous,
            "reference_price": reference,
            "combined_rights_dividend_value": combined,
            "right_dividend_type": str(row.get("權/息", "")),
            "price_adjustment_factor": reference / previous,
            "source": "TWSE TWT49U",
            "source_url": URL,
        })
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        raise RuntimeError("Official TWSE STOCK_DAY raw CSV missing")
    with RAW.open(encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))
    raw_by_date = {r["date"]: number(r["close"]) for r in raw}
    events = []
    years_checked = []
    for year in range(2010, 2027):
        events.extend(fetch_year(year))
        years_checked.append(year)
        time.sleep(0.15)
    events.sort(key=lambda x: x["date"])
    dedup = {}
    for event in events:
        if event["date"] in dedup:
            raise RuntimeError(f"Duplicate UMC TWT49U event date: {event['date']}")
        dedup[event["date"]] = event
    events = list(dedup.values())
    mismatches = []
    missing_bar_dates = []
    for event in events:
        day = event["date"]
        if day not in raw_by_date:
            missing_bar_dates.append(day)
            continue
        prior_dates = [d for d in raw_by_date if d < day]
        if not prior_dates:
            continue
        prior_day = max(prior_dates)
        prior_close = raw_by_date[prior_day]
        if prior_close is not None and abs(prior_close - event["previous_close"]) > 0.05:
            mismatches.append({
                "date": day, "raw_prior_date": prior_day,
                "raw_prior_close": prior_close,
                "official_previous_close": event["previous_close"],
            })
    path = OUT / "umc2303_official_twt49u_events.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = list(events[0]) if events else [
            "date", "symbol", "name", "previous_close", "reference_price",
            "combined_rights_dividend_value", "right_dividend_type",
            "price_adjustment_factor", "source", "source_url",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)
    report = {
        "symbol": SYMBOL,
        "source": "TWSE official TWT49U",
        "source_url": URL,
        "years_checked": years_checked,
        "events_found": len(events),
        "first_event": events[0]["date"] if events else None,
        "last_event": events[-1]["date"] if events else None,
        "missing_raw_bar_dates": missing_bar_dates,
        "previous_close_mismatches": mismatches,
        "events": events,
        "limitations": [
            "TWT49U is the official ex-right/ex-dividend reference-price record; it is not by itself a complete corporate-action ledger.",
            "TWSE notes that certain capital reductions combined with dividend reductions are excluded from TWT49U; separate special-event sources must be checked.",
            "This audit only collects and reconciles events. It does not yet claim adjusted-price backtest performance.",
        ],
    }
    (OUT / "umc2303_official_twt49u_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if missing_bar_dates or mismatches:
        raise RuntimeError(
            f"UMC TWT49U audit found missing raw bars={len(missing_bar_dates)} "
            f"or previous-close mismatches={len(mismatches)}"
        )
    print(json.dumps({k: v for k, v in report.items() if k != "events"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
