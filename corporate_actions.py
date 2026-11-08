"""Point-in-time corporate-action events for Taiwan-stock portfolio research.

Raw OHLCV remains unadjusted. Events are applied on their effective dates to
positions already held at the prior close; no future event is applied to an
earlier portfolio state. Data availability depends on the FinMind subscription.
"""
from __future__ import annotations

from datetime import date
import hashlib
import math

import pandas as pd
import requests

API_URL = "https://api.finmindtrade.com/api/v4/data"


def _number(row, *keys) -> float:
    for key in keys:
        value = row.get(key)
        try:
            number = float(str(value).replace(",", "").strip())
            if math.isfinite(number):
                return number
        except (TypeError, ValueError):
            continue
    return 0.0


def _day(value):
    stamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(stamp) else stamp.date().isoformat()


def _event(symbol, effective_date, action_type, *, share_multiplier=1.0,
           cash_per_share=0.0, payment_date=None, reference_price=None, source=""):
    effective = _day(effective_date)
    if not effective:
        return None
    ratio = float(share_multiplier)
    cash = float(cash_per_share)
    if not math.isfinite(ratio) or ratio <= 0 or not math.isfinite(cash) or cash < 0:
        return None
    payment = _day(payment_date)
    reference = None
    try:
        ref = float(reference_price)
        if math.isfinite(ref) and ref > 0:
            reference = ref
    except (TypeError, ValueError):
        pass
    identity = "|".join((str(symbol), effective, action_type, str(ratio), str(cash), payment or ""))
    return {
        "symbol": str(symbol),
        "effective_date": effective,
        "action_type": action_type,
        "share_multiplier": ratio,
        "cash_per_share": cash,
        "payment_date": payment,
        "reference_price": reference,
        "source": source,
        "event_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
    }


def _records(values):
    if values is None:
        return []
    if isinstance(values, pd.DataFrame):
        return values.to_dict("records")
    return list(values)


def normalize_corporate_actions(symbol: str, *, dividends=None, dividend_results=None,
                                splits=None, capital_reductions=None,
                                par_value_changes=None) -> pd.DataFrame:
    """Normalize FinMind corporate-action datasets into idempotent events.

    Dividend stock distributions are quoted in NTD per NTD 10 par value, so
    the share multiplier is 1 + distribution / 10. Cash dividends are recorded
    as receivables from ex-dividend date and paid on the published payment date.
    """
    events = []
    result_prices = {}
    for result in _records(dividend_results):
        key = _day(result.get("date"))
        if key:
            result_prices[key] = _number(result, "reference_price", "after_price")
    for row in _records(dividends):
        stock_rate = (
            _number(row, "StockEarningsDistribution")
            + _number(row, "StockStatutorySurplus")
        ) / 10.0
        cash_amount = (
            _number(row, "CashEarningsDistribution")
            + _number(row, "CashStatutorySurplus")
        )
        stock_date = _day(row.get("StockExDividendTradingDate"))
        cash_date = _day(row.get("CashExDividendTradingDate"))
        if stock_rate > 0 and stock_date:
            item = _event(
                symbol, stock_date, "stock_dividend",
                share_multiplier=1.0 + stock_rate,
                reference_price=result_prices.get(stock_date) or row.get("reference_price"),
                source="TaiwanStockDividend",
            )
            if item:
                events.append(item)
        if cash_amount > 0 and cash_date:
            item = _event(
                symbol, cash_date, "cash_dividend",
                cash_per_share=cash_amount,
                payment_date=row.get("CashDividendPaymentDate"),
                source="TaiwanStockDividend",
            )
            if item:
                events.append(item)

    for row in _records(splits):
        before = _number(row, "before_price")
        after = _number(row, "after_price")
        if before > 0 and after > 0:
            item = _event(
                symbol, row.get("date"), "stock_split",
                share_multiplier=before / after,
                reference_price=after,
                source="TaiwanStockSplitPrice",
            )
            if item:
                events.append(item)

    for row in _records(capital_reductions):
        before = _number(row, "ClosingPriceonTheLastTradingDay")
        after = _number(row, "PostReductionReferencePrice")
        if before > 0 and after > 0:
            item = _event(
                symbol, row.get("date"), "capital_reduction",
                share_multiplier=before / after,
                reference_price=after,
                source="TaiwanStockCapitalReductionReferencePrice",
            )
            if item:
                events.append(item)

    for row in _records(par_value_changes):
        before = _number(row, "before_close")
        after = _number(row, "after_ref_close")
        if before > 0 and after > 0:
            item = _event(
                symbol, row.get("date"), "par_value_change",
                share_multiplier=before / after,
                reference_price=after,
                source="TaiwanStockParValueChange",
            )
            if item:
                events.append(item)

    unique = {item["event_id"]: item for item in events}
    return pd.DataFrame(sorted(
        unique.values(),
        key=lambda x: (x["effective_date"], x["action_type"], x["event_id"]),
    ))


def _fetch_dataset(dataset: str, start_date: str, end_date: str | None,
                   token: str, symbol: str | None = None, timeout: int = 20) -> list[dict]:
    params = {"dataset": dataset, "start_date": start_date}
    if end_date:
        params["end_date"] = end_date
    if symbol:
        params["data_id"] = str(symbol)
    if token:
        params["token"] = token
    response = requests.get(API_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or str(payload.get("status", "200")) not in {"200", "success", "Success"}:
        message = payload.get("msg", "unknown API error") if isinstance(payload, dict) else "invalid response"
        raise RuntimeError(f"{dataset}: {message}")
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"{dataset}: invalid data payload")
    return rows


def fetch_symbol_corporate_actions(symbol: str, start_date: str, end_date: str | None = None,
                                   token: str = "", timeout: int = 20) -> pd.DataFrame:
    """Fetch one symbol's action events; raise on any missing dataset.

    The caller should fail closed or clearly label the backtest when event data
    cannot be retrieved. This function intentionally does not invent zero events.
    """
    dividends = _fetch_dataset("TaiwanStockDividend", start_date, end_date, token, symbol, timeout)
    dividend_results = _fetch_dataset("TaiwanStockDividendResult", start_date, end_date, token, symbol, timeout)
    splits = _fetch_dataset("TaiwanStockSplitPrice", start_date, end_date, token, None, timeout)
    reductions = _fetch_dataset("TaiwanStockCapitalReductionReferencePrice", start_date, end_date, token, None, timeout)
    par_changes = _fetch_dataset("TaiwanStockParValueChange", start_date, end_date, token, None, timeout)
    def only(rows):
        return [row for row in rows if str(row.get("stock_id", "")) == str(symbol)]
    return normalize_corporate_actions(
        symbol, dividends=dividends, dividend_results=dividend_results,
        splits=only(splits), capital_reductions=only(reductions),
        par_value_changes=only(par_changes),
    )


def validate_action_frame(actions: pd.DataFrame | None, symbol: str) -> pd.DataFrame:
    """Validate a supplied action frame and normalize date/order columns."""
    columns = ["symbol", "effective_date", "action_type", "share_multiplier",
               "cash_per_share", "payment_date", "reference_price", "source", "event_id"]
    if actions is None or actions.empty:
        return pd.DataFrame(columns=columns)
    frame = actions.copy()
    if "symbol" not in frame:
        frame["symbol"] = str(symbol)
    if str(symbol):
        frame = frame[frame["symbol"].astype(str) == str(symbol)].copy()
    required = {"effective_date", "action_type", "share_multiplier", "cash_per_share"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{symbol} 公司行動缺少欄位：{sorted(required - set(frame.columns))}")
    frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="coerce")
    if "payment_date" in frame:
        frame["payment_date"] = pd.to_datetime(frame["payment_date"], errors="coerce")
    else:
        frame["payment_date"] = pd.NaT
    frame["share_multiplier"] = pd.to_numeric(frame["share_multiplier"], errors="coerce")
    frame["cash_per_share"] = pd.to_numeric(frame["cash_per_share"], errors="coerce")
    frame = frame.dropna(subset=["effective_date", "share_multiplier", "cash_per_share"])
    frame = frame[
        (frame["share_multiplier"] > 0)
        & (frame["cash_per_share"] >= 0)
        & frame["action_type"].isin({
            "stock_dividend", "stock_split", "capital_reduction",
            "par_value_change", "cash_dividend",
        })
    ].sort_values(["effective_date", "action_type"]).reset_index(drop=True)
    if "event_id" not in frame:
        frame["event_id"] = [
            hashlib.sha256(
                f"{symbol}|{row.effective_date}|{row.action_type}|{row.share_multiplier}|{row.cash_per_share}".encode()
            ).hexdigest()
            for row in frame.itertuples()
        ]
    return frame.reindex(columns=columns)


def point_in_time_recalculate(frame: pd.DataFrame, actions: pd.DataFrame | None, builder):
    """Recalculate indicators as each corporate action becomes effective.

    For an ex-date at row i, only rows before i are adjusted into the new
    share/price basis. The recalculated values replace rows i onward only.
    Therefore an event never changes a signal dated before its effective date.
    The builder must be causal (rolling/expanding/EMA features using no future
    rows) and return a DataFrame with the same row count and index.
    """
    z = frame.copy().sort_values("date").reset_index(drop=True)
    output = builder(z).copy().reset_index(drop=True)
    if len(output) != len(z):
        raise ValueError("point-in-time builder 必須回傳相同列數")
    events = validate_action_frame(actions, "")
    if events.empty or z.empty:
        return output

    dates = pd.to_datetime(z["date"], errors="coerce").dt.normalize()
    adjusted = z.copy()
    for effective_date, group in events.groupby(events["effective_date"].dt.normalize(), sort=True):
        idx = int(dates.searchsorted(pd.Timestamp(effective_date).normalize(), side="left"))
        if idx <= 0 or idx >= len(adjusted):
            continue
        previous_close = float(adjusted.loc[idx - 1, "close"])
        if not math.isfinite(previous_close) or previous_close <= 0:
            continue
        price_factor = 1.0
        volume_factor = 1.0
        for event in group.itertuples(index=False):
            if event.action_type == "cash_dividend":
                cash = float(event.cash_per_share)
                if cash >= previous_close:
                    continue
                price_factor *= (previous_close - cash) / previous_close
            elif event.action_type in {
                "stock_dividend", "stock_split", "capital_reduction", "par_value_change"
            }:
                multiplier = float(event.share_multiplier)
                price_factor /= multiplier
                volume_factor *= multiplier
        if not math.isfinite(price_factor) or price_factor <= 0:
            continue
        before = adjusted.index < idx
        for column in ("open", "high", "low", "close"):
            if column in adjusted:
                adjusted.loc[before, column] = pd.to_numeric(
                    adjusted.loc[before, column], errors="coerce"
                ) * price_factor
        if "volume" in adjusted and volume_factor != 1.0:
            adjusted.loc[before, "volume"] = pd.to_numeric(
                adjusted.loc[before, "volume"], errors="coerce"
            ) * volume_factor

        recalculated = builder(adjusted).copy().reset_index(drop=True)
        if len(recalculated) != len(z):
            raise ValueError("point-in-time builder 必須回傳相同列數")
        output.iloc[idx:, :] = recalculated.iloc[idx:, :].to_numpy()
    return output
