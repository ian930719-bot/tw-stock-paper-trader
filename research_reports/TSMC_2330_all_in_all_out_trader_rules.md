# TSMC 2330 — fixed-rule all-in/all-out execution test

## Strategy constraint
The strategy remains one position only: all available cash enters after a LOW signal and the entire position exits after a HIGH signal. No shorting, scaling, permanent trend holding, stop/time exit, or leverage is used. Signals form at session close and execute at the next session open.

## Data and assumptions
- Frozen TSMC 2330 causal feature/pivot artifact, 2010-01-04 through 2026-11-09; threshold `p0.020_atr1.0`.
- Backtest: 2016-01-04 through 2026-11-09 (2026 partial).
- Initial capital NT$100,000; all available cash; integer shares; no leverage/shorting.
- Commission 0.1425% per side with NT$20 minimum; sell tax 0.3%; slippage 0.1% each side.
- Raw OHLC fills; adjusted-scale features generate signals.
- No cash dividends/corporate-action cashflows, official price-limit lock/fill checks, liquidity impact, odd-lot auction microstructure, or broker-specific discounts. This is not an executable broker replay.
- No ATR stop or time stop: only HIGH signal exit or final test-day close.

## Fixed rule families
LOW entry: (a) relative volume <= 0.8 and close within 5% of prior 20-session low (`dryup`); (b) `dryup` plus positive 5-session OBV flow ratio (`dryup_OBV`); (c) `dryup` plus positive one-day return (`dryup_green`).

HIGH exit: (a) any causal bearish RSI, MACD histogram, or volume divergence (`bear_div`); (b) relative volume >= 1.5, negative one-day return, and close location <= 0.35 (`blowoff`); or (c) either high-side rule (`bear_or_blowoff`).

All six combinations are reported. The best-performing combination is not selected as a final strategy because the rules were inspected on the same historical period.

## Cost-aware full-period results

| LOW entry / HIGH exit | Final equity | Total return | CAGR | Max drawdown | Trades | Win rate | Exposure |
|---|---:|---:|---:|---:|---:|---:|---:|
| dryup / bear_div | NT$107,605 | +7.6% | +0.68% | -47.0% | 42 | 61.9% | 44.9% |
| dryup_OBV / blowoff | NT$207,135 | +107.1% | +6.94% | -45.6% | 36 | 36.1% | 52.5% |
| dryup_green / bear_div | NT$89,121 | -10.9% | -1.06% | -45.8% | 34 | 61.8% | 33.9% |
| dryup / bear_or_blowoff | NT$71,436 | -28.6% | -3.05% | -54.2% | 70 | 45.7% | 35.1% |
| dryup_OBV / bear_div | NT$83,289 | -16.7% | -1.67% | -41.7% | 34 | 55.9% | 33.2% |
| dryup_green / blowoff | NT$286,407 | +186.4% | +10.19% | -52.9% | 38 | 44.7% | 62.6% |
| Buy-and-hold benchmark | NT$1,556,027 | +1,456.0% | +28.79% | -45.7% | 1 | — | 100% |

All rows use the same starting capital and estimated costs, and exclude cash dividends. The highest-return tested variant is `dryup_green / blowoff`, but it materially underperforms buy-and-hold and has a larger maximum drawdown. It is exploratory, not validated.

## Daily alert quality of the highest-return tested variant

Target: a retrospectively labeled pivot occurring within the next H sessions. Precision is measured over all eligible trading days, not the balanced event/control sample.

| Side | Horizon | Signal days | Base rate | Precision | Recall | False-positive rate |
|---|---:|---:|---:|---:|---:|---:|
| LOW | 3 | 124 | 13.5% | 15.3% | 3.4% | 2.9% |
| LOW | 5 | 124 | 21.9% | 33.9% | 4.7% | 2.5% |
| LOW | 10 | 123 | 36.7% | 63.4% | 5.2% | 1.7% |
| HIGH | 3 | 84 | 13.4% | 26.2% | 4.0% | 1.7% |
| HIGH | 5 | 82 | 21.8% | 41.5% | 3.8% | 1.5% |
| HIGH | 10 | 82 | 36.4% | 73.2% | 4.0% | 0.8% |

Low recall is the central failure: even when precision rises for longer horizons, the signals catch only a small share of actual pivots. A system that misses most lows and highs cannot reliably capture every wave.

## Conclusion and next gate
The event/control results identified plausible ingredients, but the full all-in/all-out simulation shows the first combinations are not good enough. The best tested pair returned +10.19% CAGR, far below price-only buy-and-hold at +28.79%, with -52.9% maximum drawdown. The 200% annualized goal is not demonstrated.

Next research should improve LOW/HIGH timing and coverage without replacing the all-in/all-out design: separate large and small wave labels; improve low-side early warning with volume contraction plus OBV/price reversal confirmation; improve high-side exits with divergence and blow-off exhaustion without blindly combining every alert; measure missed-pivot distance and false-alert cost. Thresholds learned from inspected history must be frozen before future forward paper trading.
