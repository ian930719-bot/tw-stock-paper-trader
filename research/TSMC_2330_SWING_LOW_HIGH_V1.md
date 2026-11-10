# TSMC 2330: Historical Swing-Low Buy / Swing-High Sell Research

**Status: RESEARCH_ONLY. No 200% CAGR strategy has been found.**

## Data and execution assumptions

- Existing TSMC 2330 adjusted-price archive reused; no new download.
- 4,101 daily rows, 2010-01-04 to 2026-11-10.
- Provider: Yahoo Finance `auto_adjust=True`; OHLC is split/dividend-adjusted, not raw exchange execution prices. Turnover is blank in this archive.
- Initial capital NT$100,000; cash-only, long-only, one position at a time, all-in integer shares.
- Buy commission 0.1425%; sell commission plus tax 0.4425%; slippage 0.10% per side.
- Causal fills: signal confirmed after close, enter/exit at the next available adjusted open.
- Periods: training 2010–2020; validation 2021–2023; historical test 2024-01-02 to 2026-11-10. The last period has been inspected in prior project work and is not a pristine blind holdout.

## What counts as a swing low/high

A swing low is not known on the day it occurs. In this test, the engine tracks the lowest low and confirms a low only after a closing price rebound by the buy-rebound threshold. A swing high is confirmed only after a closing price decline from the tracked high by the sell-reversal threshold. The trade is filled on the following session's open. This avoids buying an extremum with information that was unavailable at the time.

A separate retrospective oracle marks the eventual low/high extreme itself and fills at that exact historical price. That is an impossible hindsight upper bound, included only to quantify how misleading perfect-pivot backtests can be.

## Same-threshold causal swing test

Buy confirmation and sell confirmation use the same reversal threshold.

| Threshold | Train CAGR / trades | Validation CAGR / trades | Historical test CAGR / trades |
|---:|---:|---:|---:|
| 4% | 2.18% / 103 | -7.09% / 30 | 13.39% / 37 |
| 5% | 6.77% / 68 | 2.23% / 18 | 24.95% / 26 |
| 6% | 9.44% / 49 | 2.55% / 14 | 28.48% / 20 |
| 8% | 18.07% / 27 | -10.71% / 13 | 38.84% / 12 |
| 10% | 16.75% / 18 | -9.73% / 9 | 49.78% / 6 |
| 12% | 18.83% / 12 | 5.16% / 4 | 33.66% / 4 |
| 15% | 20.34% / 7 | 3.54% / 3 | 39.24% / 2 |
| 20% | 20.22% / 4 | 1.93% / 2 | 33.65% / 2 |
| 30% | 23.62% / 1 | 7.82% / 1 | 30.77% / 1 |

The larger thresholds produce very few trades; high CAGR in those rows is not statistically meaningful. At 4–6%, there are more trades but the validation results are weak. The 6% test-period max drawdown was -20.94%; the 12% test-period max drawdown was -20.59%.

## Separate buy/sell threshold search

A 17 × 17 grid of buy rebound and sell reversal thresholds (3%–20%, 1 percentage-point steps) was evaluated. Only validation configurations with at least eight trades were ranked.

- Highest validation CAGR: buy rebound 11%, sell reversal 4%.
  - Training: -0.21% CAGR, -27.10% max drawdown, 43 trades.
  - Validation: 6.98% CAGR, -17.47% max drawdown, 10 trades, PF 2.30.
  - Historical test: **-2.60% CAGR**, -25.93% max drawdown, 20 trades, PF 0.83.
- Near-tied validation alternative: buy rebound 10%, sell reversal 5%.
  - Training: 3.90% CAGR, -30.71% max drawdown, 38 trades.
  - Validation: 6.91% CAGR, -18.26% max drawdown, 11 trades, PF 2.14.
  - Historical test: 23.08% CAGR, -13.21% max drawdown, 16 trades, PF 5.39.

The 10%/5% result is a useful candidate for another forward-looking study, but selecting it because it did better in the already-inspected historical test would be test-set leakage. The strict validation winner failed in the subsequent historical test. The gap shows the instability of this parameter family.

## Perfect-pivot hindsight upper bound (not tradable)

With a 4% pivot threshold, if the system is allowed to buy at the exact eventual low and sell at the exact eventual high—even though those prices are only identifiable later—the apparent CAGR is:

| Period | Oracle CAGR |
|---|---:|
| 2010–2020 | 156.0% |
| 2021–2023 | 149.4% |
| 2024–2026/11/10 | 354.7% |

These figures are **not attainable strategy returns**. They use future information to identify the extremum and fill at its historical price. They must not be compared to a causal strategy as if they were live-tradable.

For comparison, the causal 4% rebound/decline method returned 2.18%, -7.09%, and 13.39% CAGR in those periods. This large difference is the cost of waiting for confirmation and is the central research problem.

## Conclusion and next iteration

1. The historical low/high idea has theoretical upside, but exact-pivot performance is a hindsight illusion.
2. Current causal confirmation rules do not approach 200% net CAGR.
3. Next research should compare confirmation based on close, intraday high/low, ATR-normalized reversal, volume/relative strength and market regime. Each must still fill no earlier than the next session and must be evaluated on chronological, untouched data.
4. Do not choose parameters by the 2024–2026 result. Freeze the candidate from training/validation, then reserve a genuinely unseen future paper-trading period.
5. Before execution-grade claims, replace adjusted-price fill proxies with raw exchange OHLC, official turnover, price limits, suspensions, lot sizes and verified corporate-action histories.

Reproducible runner: `research/tsmc_2330_swing_low_high.py`.
