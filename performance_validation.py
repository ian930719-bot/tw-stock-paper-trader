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
from cross_sectional_momentum import build_cross_sectional_overrides
from high_conviction_breakout import build_high_conviction_breakout_overrides
from minervini_vcp_breakout import build_minervini_vcp_overrides

# Exploratory broad universe: top 20 names from the 0050 holdings snapshot
# dated 2026-08-10, plus 2603 for the prior shipping-sector test. This is NOT
# a point-in-time historical constituent universe and retains survivorship bias.
SYMBOLS = {
    f"{code}.TW": code for code in (
        "2330", "2454", "2308", "2317", "3711", "2383", "2303",
        "3037", "2891", "2881", "2345", "2882", "1303", "2327",
        "3017", "2887", "2382", "2885", "2360", "2059", "2603",
    )
}
UNIVERSE_DESCRIPTION = "2026-08-10 0050 holdings top 20 + 2603; exploratory, survivorship-biased"
MIN_SYMBOLS = 15
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


def summarize_result(name: str, result: dict, risk_fraction: float):
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
    windows = result["windows"].copy()
    positive_window_rate = (
        float((pd.to_numeric(windows["總報酬"], errors="coerce") > 0).mean())
        if not windows.empty and "總報酬" in windows else 0.0
    )
    return {
        "策略": name,
        "樣本外區段": int(len(windows)),
        "正報酬樣本外區段比例": positive_window_rate,
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
        "單筆風險預算": float(risk_fraction),
        "資料參與率上限": 0.01,
        "選擇快取命中": int(result.get("cache_hits", 0)),
        "選擇快取未命中": int(result.get("cache_misses", 0)),
    }



def concentration_diagnostics(trades: pd.DataFrame, windows: pd.DataFrame,
                              initial_capital: float, ending_equity: float) -> dict:
    """Describe dependence on a few trades/windows; not a causal counterfactual."""
    pnl = (pd.to_numeric(trades.get("損益", pd.Series(dtype=float)), errors="coerce")
           .dropna().sort_values(ascending=False))
    net_profit = float(ending_equity - initial_capital)
    top1 = float(pnl.iloc[0]) if len(pnl) else 0.0
    top3 = float(pnl.head(3).sum()) if len(pnl) else 0.0
    window_profit = pd.Series(dtype=float)
    if not windows.empty and {"期初資產", "期末資產"}.issubset(windows.columns):
        window_profit = (
            pd.to_numeric(windows["期末資產"], errors="coerce")
            - pd.to_numeric(windows["期初資產"], errors="coerce")
        ).dropna().sort_values(ascending=False)
    top_window = float(window_profit.iloc[0]) if len(window_profit) else 0.0
    return {
        "交易筆數": int(len(pnl)), "淨利_NT": net_profit,
        "最大單筆獲利_NT": top1,
        "最大單筆佔淨利比例": top1 / net_profit if net_profit > 0 else np.nan,
        "前三筆獲利合計_NT": top3,
        "前三筆佔淨利比例": top3 / net_profit if net_profit > 0 else np.nan,
        "移除最大單筆後近似期末資產_NT": float(ending_equity - top1),
        "最大單筆樣本外區段獲利_NT": top_window,
        "最大區段佔淨利比例": top_window / net_profit if net_profit > 0 else np.nan,
        "解讀限制": "移除單筆交易為簡化敏感度指標；不重播後續資金路徑，不是可交易反事實",
    }


def benchmark_metrics_from_close(close: pd.Series, initial_capital: float = INITIAL_CAPITAL,
                               fee: float = FEE, sell_tax: float = SELL_TAX,
                               slippage: float = SLIPPAGE) -> dict:
    """Calculate a cost-adjusted 0050 buy-and-hold benchmark on total-return prices."""
    prices = pd.to_numeric(close, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    prices = prices[prices > 0].sort_index()
    if len(prices) < 2:
        raise ValueError("0050.TW 基準資料不足兩個有效交易日")
    rates = (float(fee), float(sell_tax), float(slippage))
    if any(not np.isfinite(x) or x < 0 for x in rates):
        raise ValueError("基準交易成本必須為有限非負數")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("基準初始資金必須為有限正數")

    values = prices.to_numpy(dtype=float)
    # Fractional units keep this a transparent theoretical benchmark; costs
    # match the strategy's one-way fee, sell tax and slippage assumptions.
    shares = float(initial_capital) / (values[0] * (1.0 + fee + slippage))
    marked_equity = shares * values
    ending_equity = shares * values[-1] * (1.0 - fee - sell_tax - slippage)
    marked_equity[-1] = ending_equity
    equity_path = np.r_[float(initial_capital), marked_equity]
    peaks = np.maximum.accumulate(equity_path)
    drawdowns = equity_path[1:] / peaks[1:] - 1.0
    returns = pd.Series(marked_equity).pct_change().dropna()
    first_date = pd.Timestamp(prices.index[0])
    last_date = pd.Timestamp(prices.index[-1])
    if first_date.tzinfo is not None:
        first_date = first_date.tz_localize(None)
    if last_date.tzinfo is not None:
        last_date = last_date.tz_localize(None)
    years = max((last_date - first_date).total_seconds() / (365.25 * 86400), 1 / 252)
    total_return = float(ending_equity / initial_capital - 1.0)
    cagr = float((ending_equity / initial_capital) ** (1.0 / years) - 1.0)
    daily_returns = pd.Series(marked_equity).pct_change().dropna()
    sharpe = (float(daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252))
              if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0 else 0.0)
    return {
        "基準": "0050 買進持有（Yahoo auto-adjusted；含買入及賣出成本）",
        "起始日": first_date.date().isoformat(),
        "結束日": last_date.date().isoformat(),
        "交易日數": int(len(prices)),
        "期末資產_NT": float(ending_equity),
        "總報酬": total_return,
        "年化報酬_CAGR": cagr,
        "最大回撤": float(np.nanmin(drawdowns)) if len(drawdowns) else 0.0,
        "Sharpe": sharpe,
        "買入手續費": float(fee), "賣出手續費": float(fee),
        "賣出證交稅": float(sell_tax),
        "單邊滑價": float(slippage),
        "成本處理": "已扣買入及賣出手續費、賣出證交稅及雙邊滑價；價格含 Yahoo auto-adjusted 分配調整",
        "股數假設": "理論零股，不含整股單位限制；只作成本對齊基準",
    }


def benchmark_0050(start_date, end_date):
    """Fetch Yahoo auto-adjusted 0050 prices and apply the same costs as strategies."""
    end_exclusive = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).date().isoformat()
    raw = yf.Ticker("0050.TW").history(
        start=pd.Timestamp(start_date).date().isoformat(),
        end=end_exclusive, interval="1d", auto_adjust=True,
        actions=False, repair=False, timeout=30,
    )
    if raw is None or raw.empty or "Close" not in raw:
        raise RuntimeError("0050.TW 基準資料下載失敗或為空")
    close = pd.to_numeric(raw["Close"], errors="coerce").dropna()
    return benchmark_metrics_from_close(close)

def main():
    os.makedirs("performance-artifacts", exist_ok=True)
    frames, actions, quality = {}, {}, []
    for ticker, symbol in SYMBOLS.items():
        try:
            frame, action_frame, row = fetch_history(ticker, symbol)
            frames[symbol] = frame
            actions[symbol] = action_frame
            row["資料狀態"] = "已納入"
            quality.append(row)
        except Exception as exc:
            quality.append({
                "代號": symbol, "資料來源": "Yahoo Finance",
                "資料狀態": "略過", "錯誤": str(exc),
            })
            print(f"略過資料不足或下載失敗標的 {symbol}：{exc}", flush=True)
    if len(frames) < MIN_SYMBOLS:
        raise RuntimeError(
            f"有效股票僅 {len(frames)} 檔，低於最低要求 {MIN_SYMBOLS}；停止回測避免樣本過窄"
        )

    all_dates = sorted(set().union(*[
        set(pd.to_datetime(frame["date"])) for frame in frames.values()
    ]))
    expected_windows = max(0, (len(all_dates) - LOOKBACK_DAYS - 40) // TEST_DAYS + 1)
    if expected_windows < 4:
        raise RuntimeError(
            f"可用共同交易日僅 {len(all_dates)}，樣本外區段預估 {expected_windows}，不足 4 段"
        )

    rotation_overrides = build_cross_sectional_overrides(
        frames, corporate_actions=actions,
        entry_quantile=0.80, exit_quantile=0.45,
        minimum_universe=MIN_SYMBOLS,
    )
    aggressive_overrides = build_high_conviction_breakout_overrides(
        frames, corporate_actions=actions, relative_strength_quantile=0.90,
        minimum_universe=MIN_SYMBOLS, volume_multiple=1.50,
    )
    minervini_overrides = build_minervini_vcp_overrides(
        frames, corporate_actions=actions, relative_strength_quantile=0.90,
        minimum_universe=MIN_SYMBOLS, volume_multiple=1.50,
        contraction_ratio=0.85,
    )
    strategies = list(STRATEGIES) + [
        "AI 自適應策略", "橫斷面動能輪動", "高攻擊55日突破", "Minervini SEPA-VCP 20日突破",
    ]
    summaries, all_windows, all_trades, all_equity = [], [], [], []
    robustness_rows = []
    for strategy in strategies:
        print(f"開始樣本外測試：{strategy}", flush=True)
        is_aggressive = strategy == "高攻擊55日突破"
        is_minervini = strategy == "Minervini SEPA-VCP 20日突破"
        result = continuous_walk_forward(
            frames, strategy,
            initial_capital=INITIAL_CAPITAL,
            max_positions=MAX_POSITIONS,
            risk_fraction=RISK_FRACTION,
            lookback_days=LOOKBACK_DAYS,
            test_days=TEST_DAYS,
            warmup_bars=WARMUP_BARS,
            fee=FEE, sell_tax=SELL_TAX, slippage=SLIPPAGE,
            exit_profile=("atr_2_5" if is_aggressive else "minervini_structure" if is_minervini else "baseline_atr2"),
            min_test_days=40,
            corporate_actions=actions,
            custom_signal_overrides=(
                rotation_overrides if strategy == "橫斷面動能輪動"
                else aggressive_overrides if is_aggressive
                else minervini_overrides if is_minervini else None
            ),
        )
        if len(result["windows"]) < 4:
            raise RuntimeError(f"{strategy} 僅完成 {len(result['windows'])} 個樣本外區段")
        summaries.append(summarize_result(strategy, result, RISK_FRACTION))
        if is_aggressive:
            values = pd.to_numeric(result["equity"]["資產"], errors="coerce").dropna()
            robustness_rows.append({
                "策略": strategy, "測試類型": "交易與區段集中度",
                **concentration_diagnostics(
                    result["trades"], result["windows"], INITIAL_CAPITAL,
                    float(values.iloc[-1]) if not values.empty else INITIAL_CAPITAL,
                ),
                "費率": FEE, "賣出證交稅": SELL_TAX, "單邊滑價": SLIPPAGE,
            })
            # Keep signals and parameters fixed; only increase execution costs.
            for cost_multiplier in (1.5, 2.0):
                stress = continuous_walk_forward(
                    frames, strategy, initial_capital=INITIAL_CAPITAL,
                    max_positions=MAX_POSITIONS, risk_fraction=RISK_FRACTION,
                    lookback_days=LOOKBACK_DAYS, test_days=TEST_DAYS,
                    warmup_bars=WARMUP_BARS, fee=FEE * cost_multiplier,
                    sell_tax=SELL_TAX * cost_multiplier,
                    slippage=SLIPPAGE * cost_multiplier,
                    exit_profile="atr_2_5", min_test_days=40,
                    corporate_actions=actions,
                    custom_signal_overrides=aggressive_overrides,
                )
                stress_values = pd.to_numeric(stress["equity"]["資產"], errors="coerce").dropna()
                stress_metrics = stress.get("metrics", {})
                robustness_rows.append({
                    "策略": strategy, "測試類型": f"成本壓力測試_{cost_multiplier:.1f}x",
                    "交易筆數": int(trade_statistics(stress["trades"])["交易筆數"]),
                    "淨利_NT": float(stress_values.iloc[-1] - INITIAL_CAPITAL) if not stress_values.empty else np.nan,
                    "最大單筆獲利_NT": np.nan, "最大單筆佔淨利比例": np.nan,
                    "前三筆獲利合計_NT": np.nan, "前三筆佔淨利比例": np.nan,
                    "移除最大單筆後近似期末資產_NT": np.nan,
                    "最大單筆樣本外區段獲利_NT": np.nan,
                    "最大區段佔淨利比例": np.nan,
                    "年化報酬_CAGR": float(stress_metrics.get("年化報酬", np.nan)),
                    "最大回撤": float(stress_metrics.get("最大回撤", np.nan)),
                    "費率": FEE * cost_multiplier,
                    "賣出證交稅": SELL_TAX * cost_multiplier,
                    "單邊滑價": SLIPPAGE * cost_multiplier,
                    "解讀限制": "固定訊號與參數，只提高成本；非未來實際滑價預測",
                })
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
    oos_start = pd.Timestamp(min(pd.to_datetime(x["日期"]).min() for x in all_equity))
    oos_end = pd.Timestamp(max(pd.to_datetime(x["日期"]).max() for x in all_equity))
    benchmark = benchmark_0050(oos_start, oos_end)
    summary["0050年化報酬_CAGR"] = float(benchmark["年化報酬_CAGR"])
    summary["相對0050年化超額"] = summary["年化報酬_CAGR"] - summary["0050年化報酬_CAGR"]
    pd.DataFrame([benchmark]).to_csv(
        "performance-artifacts/benchmark.csv", index=False, encoding="utf-8-sig"
    )
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
    if robustness_rows:
        pd.DataFrame(robustness_rows).to_csv(
            "performance-artifacts/robustness.csv", index=False, encoding="utf-8-sig"
        )

    source_files = [
        "performance_validation.py", "research_engine.py", "continuous_walk_forward.py",
        "high_conviction_breakout.py", "corporate_actions.py", "analytics.py",
    ]
    code_hashes = {}
    for source_path in source_files:
        if os.path.isfile(source_path):
            with open(source_path, "rb") as source_handle:
                code_hashes[source_path] = hashlib.sha256(source_handle.read()).hexdigest()
    data_digest = hashlib.sha256()
    for symbol in sorted(frames):
        data_digest.update(str(symbol).encode("utf-8"))
        data_digest.update(pd.util.hash_pandas_object(
            frames[symbol].sort_values("date").reset_index(drop=True), index=False
        ).to_numpy().tobytes())

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "unknown"),
        "git_ref": os.getenv("GITHUB_REF", "unknown"),
        "github_run_id": os.getenv("GITHUB_RUN_ID", "unknown"),
        "github_run_number": os.getenv("GITHUB_RUN_NUMBER", "unknown"),
        "python_version": sys.version,
        "code_sha256": code_hashes,
        "input_ohlcv_sha256": data_digest.hexdigest(),
        "research_grade_eligible": False,
        "eligibility_blockers": [
            "股票池使用 2026-08-10 0050 成分股快照加 2603，並非歷史時點成分股；存活者偏差未排除。",
            "Yahoo Finance 非交易所認證資料源；公司行動與退市證券覆蓋尚未完整驗證。",
        ],
        "data_source": "Yahoo Finance chart data via yfinance; raw OHLCV with dividend/split actions; 0050 benchmark uses auto-adjusted daily close and matched transaction costs",
        "benchmark": benchmark,
        "symbols": list(frames.keys()),
        "universe_description": UNIVERSE_DESCRIPTION,
        "universe_requested": len(SYMBOLS),
        "universe_loaded": len(frames),
        "common_calendar_days": len(all_dates),
        "settings": {
            "initial_capital_NTD": INITIAL_CAPITAL, "fee_rate": FEE,
            "sell_tax_rate": SELL_TAX, "one_way_slippage": SLIPPAGE,
            "risk_fraction": RISK_FRACTION, "aggressive_breakout_risk_fraction": RISK_FRACTION, "max_positions": MAX_POSITIONS,
            "max_participation_rate": 0.01, "lookback_sessions": LOOKBACK_DAYS,
            "oos_window_sessions": TEST_DAYS, "warmup_sessions": WARMUP_BARS,
            "exit_profile": "baseline_atr2",
            "continuous_portfolio": True,
            "minervini_sepa_vcp": {
                "entry": "trend template: close>SMA50>SMA150>SMA200; SMA50/SMA200 rising; price >=30% above 252-session low and within 20% of 252-session high; positive 63/126 momentum; top 10% cross-sectional 126-day momentum; ATR14/ATR50 <=0.85; 20-session breakout with volume >=1.5x prior 50-session average",
                "exit": "close below EMA21 or prior 10-session low or negative 63-session momentum; Minervini structural stop/trailing management",
                "risk_fraction_per_trade": RISK_FRACTION,
                "max_positions": MAX_POSITIONS,
                "exit_profile": "minervini_structure",
                "leverage": 1.0,
            },
            "high_conviction_breakout": {
                "entry": "55-session high breakout; top 10% 126-session momentum; positive 63/126-session momentum; rising SMA50 and SMA200 over 20 sessions; close above SMA50>SMA200; breakout extension capped at 8%; volume >= 1.5x prior 20-session average",
                "exit": "close below EMA21 or prior 20-session low or negative 63-session momentum",
                "risk_fraction_per_trade": RISK_FRACTION,
                "max_positions": MAX_POSITIONS,
                "exit_profile": "atr_2_5",
                "leverage": 1.0,
            },
        },
        "limitations": [
            "Yahoo Finance is not an exchange-certified source; its historical bars and corporate-action coverage may be revised or incomplete.",
            "Dividend and split events are sourced from Yahoo Finance; cash-dividend payment dates and some capital-reduction/par-value events are not fully modeled.",
            "Exploratory universe uses the top 20 constituents from a 2026-08-10 0050 holdings snapshot plus 2603; historical membership is not point-in-time and survivorship bias remains. This run is not research-grade acceptance evidence.",
            "0050 benchmark now deducts the same fee, sell tax and one-way slippage assumptions as the strategy; fractional units are used for benchmark comparability.",
            "CAGR is annualized using exact elapsed calendar time when dated equity observations are available.",
            "Portfolio cash and open positions carry continuously across OOS boundaries; adaptive candidates are selected only from pre-window history.",
            "Daily OHLCV cannot reconstruct intraday queue position, limit-up/down execution, halts, partial fills, or market impact.",
            "Results are paper research only and are not a prediction or guarantee of future returns.",
        ],
        "robustness_protocol": {
            "focus_strategy": "高攻擊55日突破",
            "trade_concentration": "最大單筆、前三筆及最大樣本外區段對總淨利的貢獻；移除最大單筆僅為近似敏感度",
            "cost_stress_multipliers": [1.5, 2.0],
            "stress_controls": "固定訊號、股票池、持倉上限、風險預算、區段及出場設定，只同比例提高費用、證交稅與滑價",
            "parameter_tuning": "本輪不依 OOS 結果調參",
        },
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
