# TSMC 2330 — fixed-rule all-in/all-out execution test v2

## Strategy constraint

The strategy remains one position only: all available cash enters after a LOW signal and the entire position exits after a HIGH signal. No shorting, scaling, permanent trend holding, stop/time exit, or leverage is used. Signals form at session close and execute at the next session open.

The earlier volume-dry-up / bearish-divergence combinations are retained for comparison, but the corrected event/control audit no longer supports them as robust pivot predictors.

## Corrected event/control context

The first matched-control sample was biased because corporate-action windows were not excluded. After rebuilding controls and using control-date cluster inference, no positive simple rule survives the 55-test multiple-comparison correction at q < 0.05. The expanded feature scan then identified a different LOW capitulation / HIGH climax-extension profile. The new combinations below are still exploratory because their thresholds were derived from the same historical sample.

## Data and assumptions

- Frozen TSMC 2330 causal feature/pivot artifact, 2010-01-04 through 2026-11-09; threshold `p0.020_atr1.0`.
- Backtest: 2016-01-04 through 2026-11-09 (2026 partial).
- Initial capital NT$100,000; all available cash; integer shares; no leverage/shorting.
- Commission 0.1425% per side with NT$20 minimum; sell tax 0.3%; slippage 0.1% each side.
- Raw OHLC fills; adjusted-scale features generate signals.
- Corporate-action signal dates are excluded.
- No cash dividends/corporate-action cashflows, official price-limit lock/fill checks, liquidity impact, odd-lot auction microstructure, or broker-specific discounts. This is not an executable broker replay.
- No ATR stop or time stop: only HIGH signal exit or final test-day close.

## New candidate rules

LOW capitulation:
- relative volume >= 1.2
- one-day return <= -1%
- stochastic K <= 30
- close location <= 0.25
- higher-confidence variant adds OBV 5-session flow ratio <= 0

HIGH climax:
- one-day return >= +1%
- close location >= 0.70
- intraday range >= 1.6%
- distance above SMA200 >= 10%

HIGH extension/momentum fade:
- distance above SMA200 >= 10%
- 3-session MACD histogram change <= 0
- SMA50 slope over 10 sessions > 0

The best candidate requires the LOW capitulation + negative OBV condition and the same-day HIGH climax + extension/momentum-fade conjunction.

## Cost-aware full-period results

| LOW entry / HIGH exit | Final equity | Total return | CAGR | Max drawdown | Trades | Win rate |
|---|---:|---:|---:|---:|---:|---:|
| Capitulation + OBV / climax + extension | NT$1,512,473 | +1,412.5% | 28.46% | -45.64% | 23 | 87.0% |
| Capitulation / climax + extension | NT$1,450,757 | +1,350.8% | 27.96% | -45.67% | 24 | 83.3% |
| High-volume selloff / climax + extension | NT$1,322,879 | +1,222.9% | 26.88% | -45.66% | 32 | 75.0% |
| Capitulation + OBV / climax | NT$700,434 | +600.4% | 19.66% | -45.64% | 31 | 77.4% |
| Capitulation / climax | NT$672,715 | +572.7% | 19.21% | -45.61% | 32 | 75.0% |
| High-volume selloff / climax | NT$653,815 | +553.8% | 18.90% | -45.68% | 45 | 71.1% |
| Capitulation + OBV / extension fade | NT$376,147 | +276.1% | 12.99% | -41.37% | 21 | 85.7% |
| Dry-up + positive close / blow-off | NT$286,407 | +186.4% | 10.19% | -52.87% | 38 | 44.7% |
| Dry-up + OBV / blow-off | NT$207,135 | +107.1% | 6.94% | -45.60% | 36 | 36.1% |
| Dry-up / bearish divergence | NT$118,909 | +18.9% | 1.61% | -47.20% | 39 | 61.5% |
| Dry-up + positive close / bearish divergence | NT$97,745 | -2.3% | -0.21% | -46.53% | 32 | 59.4% |
| Dry-up + OBV / bearish divergence | NT$87,539 | -12.5% | -1.22% | -42.36% | 33 | 54.5% |
| Dry-up / bearish divergence or blow-off | NT$76,607 | -23.4% | -2.43% | -54.44% | 68 | 45.6% |
| Buy-and-hold benchmark | NT$1,556,027 | +1,456.0% | 28.79% | -45.68% | 1 | — |

The best candidate is close to but still below buy-and-hold. It does not demonstrate a 200% annualized return.

## Pivot proximity and coverage of the best candidate

For the 23 completed trades:
- 21/23 entries were within 3 sessions of the nearest retrospective LOW pivot; 22/23 within 5 sessions.
- 14/23 exits were within 3 sessions of the nearest retrospective HIGH pivot; 18/23 within 5 sessions.
- Median holding time was 41 calendar days; mean 122 days due to one long hold.

At a five-session target horizon, daily alert precision/recall were:
- LOW: 61.8% precision, 4.7% recall.
- HIGH: 72.1% precision, 3.5% recall.

This is a high-confidence, low-coverage candidate. It captures a small subset of pivots and misses most turns. Pivot proximity is measured against retrospective labels on the same historical sample, not an independent validation.

## Subperiod check

| Period | Candidate CAGR | Buy-and-hold CAGR |
|---|---:|---:|
| 2016–2020 | 30.38% | 29.90% |
| 2021–2026-11-09 | 23.04% | 27.72% |

It slightly exceeded buy-and-hold in the earlier subperiod but underperformed in the later period. Stable outperformance is not demonstrated.

## Conclusion and next gate

Keep the all-in/all-out design. The strongest current hypothesis is **capitulation buying near LOWs and a climax-plus-extension confirmation near HIGHs**, but recall is too low for an all-wave system and full-period CAGR remains below buy-and-hold.

Next research should separate small-wave and large-wave labels, improve recall without losing precision, report missed-pivot distance and false-alert drawdown, and freeze rules before forward paper trading. The 200% annualized target remains unmet.
