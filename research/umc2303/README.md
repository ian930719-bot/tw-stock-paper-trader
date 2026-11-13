# UMC 2303 Swing Research

This is an independent research line. It does not modify the TSMC 2330 experiments.

## User-supplied account context

- Existing position: 300 shares at NT$148 average cost.
- Additional cash ceiling: NT$50,000.
- Earlier reference close: NT$101 on 2026-11-12; refresh official data before any live decision.
- Cost basis is NT$44,400 before fees. Any mark-to-market calculation is arithmetic, not a recommendation.

## Current research scope: pure MA5 + volume

The current core comparison uses only:
- MA5 close crossover for entry (or MA5 rising + close above MA5 in the fifth variant).
- Current-day volume compared with the prior 20 completed sessions' average, using 1.0x, 1.2x, 1.5x, or 2.0x thresholds.
- Exit when the close falls below MA5; execute no earlier than the next session's open.
- No ATR, RSI, MACD, longer moving averages, institutional flow, or fundamentals in core entry/exit rules.

The backtest starts in 2013 after using 2010 onward for indicator warm-up. Initial capital is NT$100,000; integer-share sizing is used. Commission (0.1425% each side), sell-side tax (0.3%), and slippage (0.1% each side) are explicit assumptions, not broker-confirmed actual costs.

## Execution and output

GitHub Actions runs the Python research script and uploads:
- `umc2303_research_report.json`
- `umc2303_strategy_summary.csv`
- `umc2303_twse_daily_raw.csv`

No local installation is required on the user's Windows computer.

## Critical limitations

- The downloaded TWSE STOCK_DAY history is raw and unadjusted. Cash dividends, stock dividends, splits, and other corporate actions are not incorporated. Consequently, all long-horizon performance and signals are provisional until a correctly date-aligned adjusted-price series is built.
- This is a fixed-rule exploratory comparison, not nested walk-forward validation or an untouched final holdout.
- Daily OHLCV cannot reconstruct queue priority, partial fills, limit-up/down execution, or market impact.
- No trade should be recommended from these results alone.

## Next phases

1. Audit source coverage, duplicates, missing sessions, zero values, and corporate-action events.
2. Add point-in-time corporate-action adjustment and validate price continuity.
3. Compare one-day vs two-day MA5 breakdown exits and MA5 reclaim + volume re-entry, with parameters fixed before validation.
4. Run rolling train/validation windows and preserve a final untouched holdout.
5. Report CAGR, total return, net win rate, profit factor, average win/loss, max drawdown, Sharpe/Sortino, exposure, turnover, trade count, and benchmark-relative results after costs.
6. Only then evaluate implications for the user's existing 300-share holding and additional cash ceiling.

## Official data source

TWSE historical daily trading data: https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY
