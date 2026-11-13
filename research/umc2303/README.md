# UMC 2303 Swing Research

This is a separate research line. It does not modify the TSMC 2330 experiments.

## Account facts supplied for the live-position worksheet

- Existing position: 300 shares at NT$148 average cost (user-reported).
- Additional cash ceiling: NT$50,000 (user-reported).
- Earlier reference close: NT$101 on 2026-11-12. Refresh from TWSE before any live decision.
- Existing position cost basis: NT$44,400 before fees; mark-to-market at NT$101 would be NT$30,300, or -NT$14,100 before fees. This is arithmetic, not a recommendation.

## Phase 1: price-only reproducible baseline

`run_research.py` downloads monthly batches from the TWSE official STOCK_DAY endpoint and saves the raw daily OHLCV CSV. It compares predeclared moving-average, breakout, trend-pullback, and trend-filtered mean-reversion rules. Signals are formed after the close and earliest execution is the next session's open. Estimated commission, sell-side transaction tax, and slippage are included. Sizing uses integer shares so a sub-NT$100,000 account can trade odd lots.

The workflow runs on GitHub Actions and uploads JSON/CSV results as an artifact. It does not require installing Node.js, Python, or software on the user's Windows computer.

## Important limits — do not overstate results

- Phase 1 uses raw, unadjusted TWSE prices. Corporate actions and cash dividends are not yet incorporated. Results are price-only research, not total-return results; dividend/split dates can distort indicators.
- The current strategy comparison is a fixed-rule exploratory out-of-sample run from 2013 onward, not a completed nested walk-forward parameter search.
- No institutional-flow or point-in-time fundamental features are included yet.
- Daily OHLC cannot reconstruct intraday path, queue priority, partial fills, limit-up/down execution, or market impact.
- Do not report a winning strategy, annualized return, win rate, or drawdown until the GitHub Actions artifact has been retrieved and audited.
- A previously mentioned 36-wave summary (36 reversals, 40.15% mean move, 32.39% median move, 40-session median duration) has no source dataset/code attached in this branch. Treat it as unverified historical context; do not merge it into measured results.

## Required next research phases

1. Verify data completeness, duplicates, missing sessions, zero prices, and official source dates.
2. Add corporate-action/dividend handling using event-effective dates only; validate adjusted OHLC continuity around every event.
3. Add point-in-time institutional buy/sell data only if a complete historical source is available.
4. Implement causal swing-high/swing-low labeling: a pivot is usable only after its right-side confirmation window has elapsed. Never trade at the retrospectively labeled pivot date.
5. Compare stop/target, ATR trailing, time stop, and re-entry after stop-out.
6. Run nested rolling validation: optimize only inside each training window, freeze parameters, test the next non-overlapping period; reserve the latest period as final untouched holdout.
7. Evaluate CAGR, total return, win rate, profit factor, average win/loss, exposure, turnover, max drawdown, Sharpe/Sortino, trade count, and benchmark-relative performance after costs.
8. Translate only validated rules into a position plan for the 300-share holding and NT$50,000 cash. Do not assume averaging down is optimal.

## Cost assumptions

Initial research capital NT$100,000; commission 0.1425% each side, sell tax 0.3%, slippage 0.1% each side. These are explicit research assumptions, not the user's broker-specific actual rates. Replace with the actual fee discount/minimum charge before account-level simulation.

## Data source

TWSE historical daily trading data: https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY
