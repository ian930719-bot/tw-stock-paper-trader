# TSMC 2330 Exit Policy Study V8

**Status:** exploratory research; not a production trading recommendation.

## Objective
Hold the V6 entry signal fixed (`p_low >= 0.65`) and change only exit management. Signals are evaluated at the close and filled at the next session open.

## Data and protocol
- Symbol: 2330; 4,101 adjusted daily bars, 2010-01-04 to 2026-11-10.
- Source: Yahoo Finance, `auto_adjust=True`.
- Walk-forward test: 2016-01-04 to 2026-11-10; expanding yearly training and matured-label purge inherited from V6.
- Initial capital NT$100,000; integer shares, cash-only, long-only.
- Costs: buy fee 0.1425%; sell fee plus tax 0.4425%; slippage 0.10% each side.

## Results

| Exit policy | Final equity | CAGR | Max drawdown | Trades | Win rate | Mean net trade |
|---|---:|---:|---:|---:|---:|---:|
| V6 baseline | NT$182,392 | 5.70% | -44.54% | 79 | 60.76% | 0.98% |
| 3 x ATR(14) trailing stop | NT$411,552 | 13.93% | -49.29% | 52 | 55.77% | 3.37% |
| 4 x ATR(14) trailing stop | NT$480,228 | 15.56% | -52.16% | 37 | 56.76% | 5.72% |
| High warning AND close below MA60; 12% close stop; 252-session cap | NT$597,023 | 17.90% | -45.05% | 58 | 56.90% | 4.43% |
| Integer-share buy-and-hold | NT$2,104,057 | 32.42% | -44.80% | — | — | — |

All strategy variants use identical V6 entries. ATR trails use the highest close since entry minus 3 or 4 times current ATR(14). The warning/MA60 exit requires both `p_high >= 0.60` and close below MA60.

## Interpretation
1. Exit management materially changes the V6 result. The warning-plus-MA60 policy raises CAGR from 5.70% to 17.90%, with drawdown close to the buy-and-hold benchmark.
2. Wider ATR trails improve returns but deepen drawdowns.
3. The best tested exit remains far below buy-and-hold (NT$597k vs NT$2.104m). This does not establish a robust edge and does not support a 200% CAGR claim.
4. The evidence suggests that acting on a high-warning probability alone is too aggressive; requiring trend deterioration before exit is a hypothesis worth testing across more stocks.

## Limitations
- This reuses the previously examined 2016-2026 OOS period. It is exploratory, not a pristine untouched holdout.
- Only one stock; no market-wide generalization.
- Adjusted Yahoo data and simplified next-open fills are not official execution-grade exchange data.
- Overnight gaps, market impact, partial fills, limit-up/down queues and broker restrictions are not modeled.
- No global parameter optimization is claimed.

## Next step
Freeze these V8 rules and test across a survivorship-bias-controlled Taiwan stock universe with point-in-time constituents, official market rules, corporate actions and realistic execution constraints. Preserve a genuinely untouched final holdout before further strategy selection.
