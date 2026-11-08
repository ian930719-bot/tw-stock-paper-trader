"""Reproducible OOS performance validation using raw Yahoo daily bars.

This is a preliminary data-source run, not exchange-certified research data.
It fails closed if any requested ticker has insufficient or malformed history.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from research_engine import STRATEGIES
from continuous_walk_forward import continuous_walk_forward
from analytics import trade_statistics

SYMBOLS = {"2330.TW": "2330", "2454.TW": "2454", "2603.TW": "2603"}
INITIAL_CAPITAL = 100_000.0
FEE = 0.001425
SELL_TAX = 0.003
SLIPPAGE = 0.001
RISK_FRACTION = 0.005
MAX_POSITIONS = 3
LOOKBACK_DAYS = 756
TEST_DAYS = 126
WARMUP_BARS = 200
MIN_BARS = 1200


def fetch_history(ticker: str, symbol: str):
    last_error = None
    for attempt in range(3):
        try:
            raw = yf.Ticker(ticker).history(
                period="10y", interval="1d", auto_adjust=False,
                actions=True, repair=False, timeout=30,
            )
            if raw is None or raw.empty:
                raise RuntimeError("Yahoo Finance 回傳空資料")
            raw = raw.copy()
            raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
            raw = raw[~raw.index.duplicated(keep="last")].sort_index()
            frame = pd.DataFrame({
                "date": raw.index,
                "open": pd.to_numeric(raw["Open"], errors="coerce").to_numpy(),
                "high": pd.to_numeric(raw["High"], errors="coerce").to_numpy(),
                "low": pd.to_numeric(raw["Low"], errors="coerce").to_numpy(),
                "close": pd.to_numeric(raw["Close"], errors="coerce").to_numpy(),
                "volume": pd.to_numeric(raw["Volume"], errors="coerce").to_numpy(),
            })
            frame = frame.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)
            valid = (
                (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
                & (frame["volume"] >= 0)
                & (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
                & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
            )
            frame = frame.loc[valid].reset_index(drop=True)
            if len(frame) < MIN_BARS:
                raise RuntimeError(f"有效日線僅 {len(frame)} 根，低於最低要求 {MIN_BARS}")
            actions = []
            for idx, row in raw.iterrows():
                day = pd.Timestamp(idx).date().isoformat()
                dividend = float(row.get("Dividends", 0.0) or 0.0)
                split = float(row.get("Stock Splits", 0.0) or 0.0)
                if np.isfinite(dividend) and dividend > 0:
                    event_id = hashlib.sha256(
                        f"{symbol}|{day}|cash_dividend|{dividend}".encode()
                    ).hexdigest()
                    actions.append({
                        "symbol": symbol, "effective_date": day,
                        "action_type": "cash_dividend", "share_multiplier": 1.0,
                        "cash_per_share": dividend, "payment_date": None,
                        "reference_price": None, "price_adjustment_factor": np.nan,
                        "source": "Yahoo Finance actions", "event_id": event_id,
                    })
                if np.isfinite(split) and split > 0 and split != 1:
                    event_id = hashlib.sha256(
                        f"{symbol}|{day}|stock_split|{split}".encode()
                    ).hexdigest()
                    actions.append({
                        "symbol": symbol, "effective_date": day,
                        "action_type": "stock_split", "share_multiplier": split,
                        "cash_per_share": 0.0, "payment_date": None,
                        "reference_price": None, "price_adjustment_factor": 1.0 / split,
                        "source": "Yahoo Finance actions", "event_id": event_id,
                    })
            action_frame = pd.DataFrame(actions, columns=[
                "symbol", "effective_date", "action_type", "share_multiplier",
                "cash_per_share", "payment_date", "reference_price",
                "price_adjustment_factor", "source", "event_id",
            ])
            quality = {
                "代號": symbol, "資料來源": "Yahoo Finance（原始 OHLCV + actions）",
                "有效日線": int(len(frame)),
                "起始日": pd.Timestamp(frame["date"].iloc[0]).date().isoformat(),
                "最後交易日": pd.Timestamp(frame["date"].iloc[-1]).date().isoformat(),
                "公司行動筆數": int(len(action_frame)),
                "缺漏交易日檢查": "僅檢查資料列完整性；未與交易所官方日曆逐日比對",
            }
            return frame, action_frame, quality
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{symbol} 歷史資料下載失敗：{last_error}")


def summarize_result(name: str, result: dict):
    equity = result["equity"].copy()
    trades = result["trades"].copy()
    if equity.empty:
        raise RuntimeError(f"{name} 沒有產生任何樣本外資產曲線")
    values = pd.to_numeric(equity["資產"], errors="coerce").dropna()
    if values.empty or float(values.iloc[-1]) <= 0:
        raise RuntimeError(f"{name} 的資產曲線無效")
    peak = pd.concat([pd.Series([INITIAL_CAPITAL]), values.reset_index(drop=True)]).cummax().iloc[1:].reset_index(drop=True)
    drawdown = values.reset_index(drop=True) / peak - 1.0
    total_return = float(values.iloc[-1] / INITIAL_CAPITAL - 1.0)
    metric_values = result.get("metrics", {})
    cagr = float(metric_values.get("年化報酬", 0.0))
    sharpe = float(metric_values.get("Sharpe", 0.0))
    stats = trade_statistics(trades)
    return {
        "策略": name,
        "樣本外區段": int(len(result["windows"])),
        "樣本外交易日": int(len(values)),
        "交易筆數": int(stats["交易筆數"]),
        "勝率": float(stats["勝率"]),
        "Profit Factor": float(stats["Profit Factor"]),
        "期望值_NT": float(stats["期望值"]),
        "初始資金_NT": INITIAL_CAPITAL,
        "期末資產_NT": float(values.iloc[-1]),
        "總報酬": total_return,
        "年化報酬_CAGR": cagr,
        "最大回撤": float(metric_values.get("最大回撤", drawdown.min())),
        "Sharpe": sharpe,
        "費率": FEE,
        "賣出證交稅": SELL_TAX,
        "單邊滑價": SLIPPAGE,
        "持倉上限": MAX_POSITIONS,
        "單筆風險預算": RISK_FRACTION,
        "資料參與率上限": 0.01,
        "選擇快取命中": int(result.get("cache_hits", 0)),
        "選擇快取未命中": int(result.get("cache_misses", 0)),
    }


def main():
    os.makedirs("performance-artifacts", exist_ok=True)
    frames, actions, quality = {}, {}, []
    for ticker, symbol in SYMBOLS.items():
        frame, action_frame, row = fetch_history(ticker, symbol)
        frames[symbol] = frame
        actions[symbol] = action_frame
        quality.append(row)

    all_dates = sorted(set().union(*[
        set(pd.to_datetime(frame["date"])) for frame in frames.values()
    ]))
    expected_windows = max(0, (len(all_dates) - LOOKBACK_DAYS - 40) // TEST_DAYS + 1)
    if expected_windows < 4:
        raise RuntimeError(
            f"可用共同交易日僅 {len(all_dates)}，樣本外區段預估 {expected_windows}，不足 4 段"
        )

    strategies = list(STRATEGIES) + ["AI 自適應策略"]
    summaries, all_windows, all_trades, all_equity = [], [], [], []
    for strategy in strategies:
        print(f"開始樣本外測試：{strategy}", flush=True)
        result = continuous_walk_forward(
            frames, strategy,
            initial_capital=INITIAL_CAPITAL,
            max_positions=MAX_POSITIONS,
            risk_fraction=RISK_FRACTION,
            lookback_days=LOOKBACK_DAYS,
            test_days=TEST_DAYS,
            warmup_bars=WARMUP_BARS,
            fee=FEE, sell_tax=SELL_TAX, slippage=SLIPPAGE,
            exit_profile="baseline_atr2",
            min_test_days=40,
            corporate_actions=actions,
        )
        if len(result["windows"]) < 4:
            raise RuntimeError(f"{strategy} 僅完成 {len(result['windows'])} 個樣本外區段")
        summaries.append(summarize_result(strategy, result))
        if not result["windows"].empty:
            part = result["windows"].copy()
            part.insert(0, "策略", strategy)
            all_windows.append(part)
        if not result["trades"].empty:
            part = result["trades"].copy()
            part.insert(0, "策略", strategy)
            all_trades.append(part)
        if not result["equity"].empty:
            part = result["equity"].copy()
            part.insert(0, "策略", strategy)
            all_equity.append(part)
        print(f"完成：{strategy}；區段 {len(result['windows'])}；交易 {len(result['trades'])}", flush=True)

    summary = pd.DataFrame(summaries).sort_values("年化報酬_CAGR", ascending=False)
    summary.to_csv("performance-artifacts/summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_windows, ignore_index=True).to_csv(
        "performance-artifacts/window_metrics.csv", index=False, encoding="utf-8-sig"
    )
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(
            "performance-artifacts/closed_trades.csv", index=False, encoding="utf-8-sig"
        )
    if all_equity:
        pd.concat(all_equity, ignore_index=True).to_csv(
            "performance-artifacts/oos_equity.csv", index=False, encoding="utf-8-sig"
        )
    pd.DataFrame(quality).to_csv(
        "performance-artifacts/data_quality.csv", index=False, encoding="utf-8-sig"
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "Yahoo Finance chart data via yfinance; raw OHLCV with dividend/split actions",
        "symbols": list(SYMBOLS.values()),
        "common_calendar_days": len(all_dates),
        "settings": {
            "initial_capital_NTD": INITIAL_CAPITAL, "fee_rate": FEE,
            "sell_tax_rate": SELL_TAX, "one_way_slippage": SLIPPAGE,
            "risk_fraction": RISK_FRACTION, "max_positions": MAX_POSITIONS,
            "max_participation_rate": 0.01, "lookback_sessions": LOOKBACK_DAYS,
            "oos_window_sessions": TEST_DAYS, "warmup_sessions": WARMUP_BARS,
            "exit_profile": "baseline_atr2",
            "continuous_portfolio": True,
        },
        "limitations": [
            "Yahoo Finance is not an exchange-certified source; its historical bars and corporate-action coverage may be revised or incomplete.",
            "Dividend and split events are sourced from Yahoo Finance; cash-dividend payment dates and some capital-reduction/par-value events are not fully modeled.",
            "The current universe contains only three currently listed symbols; this does not remove survivorship bias.",
            "Portfolio cash and open positions carry continuously across OOS boundaries; adaptive candidates are selected only from pre-window history.",
            "Daily OHLCV cannot reconstruct intraday queue position, limit-up/down execution, halts, partial fills, or market impact.",
            "Results are paper research only and are not a prediction or guarantee of future returns.",
        ],
        "result_rows": len(summary),
    }
    with open("performance-artifacts/metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print("\n=== OOS PERFORMANCE SUMMARY ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nArtifacts saved to performance-artifacts/", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PERFORMANCE_VALIDATION_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
