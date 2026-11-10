# TSMC (2330) Single-Stock Parameter Sweep — First Pass

**Status: RESEARCH_ONLY.** This is a first reproducible diagnostic, not a claim of a deployable 200% CAGR strategy.

## Data

- Existing archive reused: `top10-adjusted-prices`, symbol `2330` (TSMC).
- Provider: Yahoo Finance `auto_adjust=True`; 4,101 daily rows, 2010-01-04 through 2026-11-10.
- Adjusted OHLC includes historical dividend/split adjustments; volume is provider-reported. The archive's turnover field is blank.
- Therefore this run uses adjusted open/high/low/close as execution-price proxies, not raw exchange fills. It cannot faithfully reproduce Taiwan price limits, suspensions, tick sizes, board lots, raw turnover capacity or actual partial fills.
- Initial capital: NT$100,000; long-only, cash-only, one position at a time, integer shares, fully reinvested when a position is open.
- Costs per round trip: buy commission 0.1425%, sell commission + tax 0.4425%, slippage 0.10% each side.
- Signal is calculated after close and entered no earlier than next session open. ATR stop uses the prior close ATR for initial stop; trailing stop updates after the completed session.

## Parameter experiment

A 576-combination grid was tested over 2010-01-04 to 2020-12-31. Parameters varied:

- Breakout lookback: 20 or 55 sessions (momentum continuation was also included).
- Exit EMA: 20 or 50 sessions.
- ATR stop/trailing distance: 2.5, 3.5, or 4.5 ATR.
- Maximum holding period: 20, 40, or 60 sessions.
- Trend filter: 150- or 200-session SMA.
- Volume ratio threshold: 1.0 or 1.2.
- RSI ceiling: 75 or 90.

The top 30 configurations by training CAGR were carried to the 2021–2023 validation period. The selected configuration was the best validation CAGR among those 30. The 2024–2026 period was then evaluated without using it to select the configuration. Note: this period has been inspected in previous project work, so it is not a pristine blind holdout.

## Selected configuration

- Entry: close above prior 20-session high; close above SMA200; SMA50 above SMA200; 60-session momentum positive; volume / prior 20-session average volume at least 1.0; RSI no greater than 75.
- Exit: close below EMA50, or 40 bars held; execution next open.
- Initial stop and trailing stop: 4.5 ATR.

## Results

| Period | Net CAGR | Max drawdown | Trades | Profit factor | Ending equity |
|---|---:|---:|---:|---:|---:|
| Training 2010–2020 | 5.96% | -22.21% | 35 | 2.66 | NT$188,964 |
| Validation 2021–2023 | 1.67% | -23.80% | 9 | 1.25 | NT$105,069 |
| Historical test 2024–2026/11/10 | 14.44% | -18.85% | 12 | 1.92 | NT$146,991 |

The strongest in-sample training configuration reached 7.68% CAGR, -15.33% max drawdown, 21 trades, and 4.74 profit factor, but it was not the best validation configuration. This is precisely why the selection is based on validation rather than the training peak.

## Conclusion

This first TSMC-only experiment does **not** approach the 200% net CAGR target. The validation CAGR is only 1.67%, with nine trades, and the post-selection historical test CAGR is 14.44%. The sample is too small to infer stable predictive power.

Do not increase leverage or loosen risk controls to manufacture a 200% result. Next iterations should expand the feature/strategy families (trend continuation, pullback/reclaim, volatility squeeze, relative strength and market-regime filters), use nested walk-forward splits and nearby-parameter stability checks, and independently reconcile raw exchange prices/corporate actions before any execution-grade claim. TSMC-specific results must remain separate from multi-stock portfolio results.
