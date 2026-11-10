"""Reconstruct TSMC 2330 daily limits from official TWSE rules and references.

This avoids the unavailable FinMind paid-only TaiwanStockPriceLimit dataset.
For 2330, daily reference price is the prior traded close except on verified
cash ex-dividend dates, where the published ex-dividend reference price is
used. The historical 7% band before 2015-06-01 and 10% band thereafter are
applied with the TWSE minimum tick schedule. The latest official TWT84U
snapshot is used as a live rule cross-check when available.
"""
from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TWSE_SNAPSHOT_URL = "https://openapi.twse.com.tw/v1/exchangeReport/TWT84U"
BAND_CHANGE_DATE = pd.Timestamp("2015-06-01")


def tick_size(price: float) -> Decimal:
    p = Decimal(str(price))
    if p < Decimal("10"):
        return Decimal("0.01")
    if p < Decimal("50"):
        return Decimal("0.05")
    if p < Decimal("100"):
        return Decimal("0.1")
    if p < Decimal("500"):
        return Decimal("0.5")
    if p < Decimal("1000"):
        return Decimal("1")
    return Decimal("5")


def limit_price(reference_price: float, band: float, side: str) -> float:
    if not np.isfinite(reference_price) or reference_price <= 0:
        raise ValueError("Reference price must be positive and finite")
    if not 0 < band < 1:
        raise ValueError("Price-limit band must be between zero and one")
    if side not in {"UP", "DOWN"}:
        raise ValueError("side must be UP or DOWN")
    raw = Decimal(str(reference_price)) * (
        Decimal("1") + Decimal(str(band)) if side == "UP"
        else Decimal("1") - Decimal(str(band))
    )
    rounding = ROUND_FLOOR if side == "UP" else ROUND_CEILING
    candidate = raw
    for _ in range(10):
        tick = tick_size(float(candidate))
        units = (raw / tick).to_integral_value(rounding=rounding)
        rounded = units * tick
        if tick_size(float(rounded)) == tick:
            return float(rounded)
        candidate = rounded
    raise RuntimeError(f"Tick-size rounding did not converge for {reference_price=} {band=} {side=}")


def _field(row: dict, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in row and str(row[name]).strip() not in {"", "nan", "None", "--"}:
            return str(row[name]).replace(",", "").strip()
    return None


def validate_latest_official_snapshot() -> dict:
    result = {
        "attempted": True, "source": TWSE_SNAPSHOT_URL,
        "available": False, "symbol": "2330",
    }
    row = None
    try:
        response = requests.get(TWSE_SNAPSHOT_URL, timeout=(8, 20))
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("TWSE TWT84U response is not a JSON array")
        row = next((r for r in payload if str(
            _field(r, ("證券代號", "證券代碼", "Code", "股票代號")) or ""
        ).strip() == "2330"), None)
        if row is None:
            raise ValueError("2330 row not found in official TWT84U snapshot")
        reference = float(_field(row, ("本日開盤競價基準", "開盤競價基準", "開盤參考價", "reference_price", "TodayOpeningRefPrice")))
        actual_up = float(_field(row, ("本日漲停價", "漲停價", "limit_up", "TodayLimitUp")))
        actual_down = float(_field(row, ("本日跌停價", "跌停價", "limit_down", "TodayLimitDown")))
        band = 0.07 if pd.Timestamp.now().normalize() < BAND_CHANGE_DATE else 0.10
        calculated_up = limit_price(reference, band, "UP")
        calculated_down = limit_price(reference, band, "DOWN")
        result.update({
            "available": True, "raw_row": row, "reference_price": reference,
            "official_limit_up": actual_up, "official_limit_down": actual_down,
            "calculated_limit_up": calculated_up, "calculated_limit_down": calculated_down,
            "matches": math.isclose(actual_up, calculated_up, abs_tol=1e-8)
                       and math.isclose(actual_down, calculated_down, abs_tol=1e-8),
            "row": row,
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if row is not None:
            result["raw_row"] = row
            result["row_keys"] = sorted(row.keys())
    return result


def build(bars_path: str, dividends_path: str, output_path: str) -> dict:
    bars = pd.read_csv(bars_path, dtype={"symbol": str})
    required_bars = {"date", "open", "high", "low", "close", "volume"}
    if not required_bars.issubset(bars.columns):
        raise ValueError(f"Bars missing columns: {sorted(required_bars - set(bars.columns))}")
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
    for col in ("open", "high", "low", "close", "volume"):
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    bars = bars.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    bars = bars.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if len(bars) < 260:
        raise ValueError("Insufficient TSMC daily bars to construct historical reference prices")

    div = pd.read_csv(dividends_path, dtype={"stock_id": str})
    required_div = {"date", "stock_id", "reference_price", "stock_or_cache_dividend"}
    if not required_div.issubset(div.columns):
        raise ValueError(f"Dividend results missing columns: {sorted(required_div - set(div.columns))}")
    div = div.loc[
        (div["stock_id"].astype(str).str.strip() == "2330")
        & (div["stock_or_cache_dividend"].astype(str).str.strip() == "息")
    ].copy()
    div["date"] = pd.to_datetime(div["date"], errors="coerce").dt.normalize()
    div["reference_price"] = pd.to_numeric(div["reference_price"], errors="coerce")
    div = div.dropna(subset=["date", "reference_price"])
    dividend_reference = {
        row.date: float(row.reference_price) for row in div.itertuples(index=False)
    }

    output = []
    previous_close = None
    conflicts = []
    for row in bars.itertuples(index=False):
        day = pd.Timestamp(row.date).normalize()
        if day in dividend_reference:
            reference = dividend_reference[day]
            reference_source = "FinMind TaiwanStockDividendResult.reference_price"
        elif previous_close is not None:
            reference = previous_close
            reference_source = "previous_traded_close; no corporate-action event reported for 2330"
        else:
            reference = np.nan
            reference_source = "unavailable_first_observation"
        band = 0.07 if day < BAND_CHANGE_DATE else 0.10
        if np.isfinite(reference):
            up = limit_price(reference, band, "UP")
            down = limit_price(reference, band, "DOWN")
            if float(row.high) > up + 1e-6 or float(row.low) < down - 1e-6:
                conflicts.append({
                    "date": day.date().isoformat(), "high": float(row.high),
                    "low": float(row.low), "limit_up": up, "limit_down": down,
                    "reference_price": reference,
                })
        else:
            up, down = np.nan, np.nan
        output.append({
            "date": day.date().isoformat(), "stock_id": "2330",
            "reference_price": reference, "limit_up": up, "limit_down": down,
            "band_pct": band, "reference_source": reference_source,
            "limit_source": "TWSE official Rule 63 + official tick-size schedule",
        })
        if float(row.volume) > 0:
            previous_close = float(row.close)
        else:
            # A no-trade session may require previous bid/ask adjustment to the
            # next opening reference price. Fail closed rather than assume close.
            previous_close = None

    if conflicts:
        raise RuntimeError(
            f"Calculated official-rule limits conflict with TSMC traded OHLC on {len(conflicts)} sessions; "
            f"first conflicts={conflicts[:5]}"
        )
    result = pd.DataFrame(output)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    snapshot = validate_latest_official_snapshot()
    if snapshot.get("available") and not snapshot.get("matches"):
        raise RuntimeError(
            "TWSE TWT84U latest official limit snapshot does not match the official-rule calculation"
        )
    summary = {
        "symbol": "2330", "rows": int(len(result)),
        "first_date": result["date"].min(), "last_date": result["date"].max(),
        "cash_ex_dividend_reference_rows": int(len(dividend_reference)),
        "historical_limit_regimes": [
            {"through": "2015-05-29", "band_pct": 0.07},
            {"from": "2015-06-01", "band_pct": 0.10},
        ],
        "tick_size_schedule": [
            {"price_from": 0, "price_to": 10, "tick": 0.01},
            {"price_from": 10, "price_to": 50, "tick": 0.05},
            {"price_from": 50, "price_to": 100, "tick": 0.1},
            {"price_from": 100, "price_to": 500, "tick": 0.5},
            {"price_from": 500, "price_to": 1000, "tick": 1},
            {"price_from": 1000, "price_to": None, "tick": 5},
        ],
        "ohlc_limit_conflicts": 0,
        "latest_twse_snapshot_validation": snapshot,
        "source": "TWSE official trading rules; TWT84U current snapshot cross-check attempted; FinMind daily bars and ex-dividend reference prices",
        "note": "Historical limit prices are rule-derived from official reference-price conventions, historical 7%/10% regimes and tick sizes. Prior traded close is used only on ordinary sessions with a prior traded close; actual ex-dividend reference prices are used on all 2330 cash ex-dividend dates. No simple prior-close x 1.1 shortcut is used.",
    }
    Path(output_path).with_name("price_limits_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", required=True)
    parser.add_argument("--dividends", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.bars, args.dividends, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
