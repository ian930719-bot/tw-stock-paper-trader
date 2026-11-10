# 2016–2026 Historical Strategy Research: Initial Code and Data Audit

Audit date: 2026-11-10
Branch: research/200cagr-blind-validation
Status: RESEARCH_ONLY; no strategy qualifies for the 100% net CAGR hurdle.

## 1. Scope and decision

The target is 100% net CAGR over a causal, continuously funded Taiwan-equity portfolio from 2016 through the latest verified 2026 trading date. 2026 is partial. No retrospective ZigZag turning point or future high/low label is an executable signal.

The first code/data review found that existing research components are useful building blocks but do not yet constitute a complete 2016–2026 all-market portfolio search.

## 2. Existing code findings

### `swing_turning_point_research.py`
- The symbol list is hard-coded to ten currently prominent names: 2330, 2454, 1303, 2327, 2408, 2308, 3037, 3008, 2383, 2303.
- Its configured model periods are anchored around TRAIN_END=2020-11-01, validation beginning 2021-03-01, and test beginning 2024-03-01. This is not the requested full 2016–2026 all-stock evaluation.
- It does implement a shared-cash portfolio simulator, next-session-open entries, slippage, fees/tax, ATR sizing, ADV proxy limits, stops, and trade/equity logs. These are reusable, but the universe and date design must be replaced or wrapped by a broader walk-forward experiment.
- Its ADV proxy and corporate-action-adjusted data basis require further reconciliation before capacity claims.

### `continuous_walk_forward.py`
- Provides a continuous shared-capital account across non-overlapping OOS windows, avoiding account resets at every boundary.
- It is a useful execution/evaluation foundation, but it does not itself solve historical point-in-time constituent membership or establish that every supplied frame was investable on each historical date.
- Any use with a current/delisted candidate list must retain the survivorship-bias warning until the historical universe is verified.

### `cross_sectional_momentum.py` and `high_conviction_breakout.py`
- Provide causal, close-known signal builders for cross-sectional momentum and 55-session breakout families.
- They are candidate families, not proof of profitability. Their relative ranks are only as representative as the supplied universe on each date.

### `streaming_factor_research.py`
- Produces factor IC and gross quintile-spread diagnostics.
- Its own report correctly marks results as descriptive rather than a costed portfolio backtest and explicitly lists data blockers. It cannot be used as evidence that the 100% hurdle has been met.

### `per_stock_rule_mining.py`
- Uses a per-symbol chronological 70/30 split and next-open entry assumptions.
- Forward labels overlap and results are not a shared-capital portfolio. The script itself lists these limitations; its compounded signal returns must not be reported as portfolio CAGR.

## 3. Historical data limitations

The collector `historical_data_collector.py` uses FinMind `TaiwanStockPrice` raw, unadjusted OHLCV and turnover. Its candidate list is assembled from a current security master plus historical delisting announcements. The collector explicitly sets:
- `point_in_time_universe_complete = false`
- `delisted_price_coverage_verified = false`

Therefore the current candidate list is not a verified point-in-time historical universe. Current constituents and surviving symbols cannot silently represent the investable universe in 2016–2026.

Prior collection artifacts reported 47,412 rows in the turning-point research panel, 2,189 symbols with prices from 2,220 candidates, and 31 empty candidates. Those counts are historical artifact observations, not a fresh verification of the current branch data cache. Prior collection also reported incomplete corporate-action coverage, including an unavailable capital-reduction reference-price dataset. Official exchange calendar and raw-price reconciliation were not complete.

Until these are reconciled, every performance result remains exploratory and RESEARCH_ONLY. Do not claim survivorship-safe or exchange-verified results.

## 4. Previously observed performance (not a new run)

Prior research artifacts reported:
- 2020–2021 validation CAGR: 24.22%, maximum drawdown -6.99%, 23 trades.
- 2023–2026/11 historical OOS artifact: CAGR 5.19%, cumulative return +21.5%, maximum drawdown -20.82%, 52 trades.

These results are historical references from earlier runs, not a fresh reproduction in this audit. Neither is close to the 100% net CAGR hurdle. The 2023–2026 observations have already been inspected and cannot be called an untouched blind holdout.

## 5. Required next engineering steps

1. Rebuild or verify a dated, survivorship-aware universe for each trading date, including delisted names and listing/delisting intervals. Until then, label the universe incomplete.
2. Reconcile raw OHLCV, turnover, corporate actions, price limits, suspensions, and trading dates against official TWSE/TPEx samples. Keep raw execution prices distinct from event-adjusted feature prices.
3. Create a unified all-symbol portfolio experiment using the 2016–2026 date range. If annual top-20 turnover universes are used, each year's constituents must be selected using only turnover information available before that year.
4. Evaluate multiple distinct strategy families (momentum rotation, breakout/continuation, pullback, reversal, volatility expansion, regime allocation) with predeclared hypotheses and comparable cost/execution assumptions.
5. Use chronological nested walk-forward selection, continuous shared capital, next-session executable fills, cash/position constraints, realistic commissions/tax/slippage, ADV capacity, suspensions/price-limit handling, and explicit corporate-action treatment.
6. Export a full trade ledger, daily equity curve, yearly and regime breakdowns, concentration and capacity diagnostics, and 1.5x/2.0x cost stress tests.
7. Do not optimize the already inspected 2023–2026 results and then call them blind. Freeze a version before prospective forward paper evaluation.

## 6. Pass/fail rule

A candidate does not pass merely because a single backtest reports 100% CAGR. It must reach the target on the continuous net equity curve and survive data-integrity, chronological validation, cost stress, drawdown, concentration, capacity, and cross-symbol/regime checks. No live-trading approval is implied by a historical pass.
