# TSMC 2330 frozen-rule robustness audit — preliminary

## Scope

This audit holds the candidate rule fixed and tests whether its reported performance survives chronological segmentation, delayed execution, higher slippage, and small one-factor threshold perturbations. It is a sensitivity check, not a new parameter search and not an independent holdout.

Data: frozen TSMC 2330 causal feature set and pivot artifact, 2010-01-04 through 2026-11-09. Executable backtest: 2016-01-04 through 2026-11-09 (2026 partial). Starting capital NT$100,000. All available cash, integer shares, no shorting, no leverage. Signal after close, execution next session open. Commission 0.1425% each side (NT$20 minimum), sell tax 0.3%, slippage 0.1% each side. No cash dividends, price-limit lock/fill simulation, market impact, odd-lot auction effects, or broker-specific discounts.

## Frozen candidate

LOW entry requires all:
- Relative volume 20 sessions >= 1.2
- One-day return <= -1%
- Stochastic K <= 30
- Close location <= 0.25
- OBV 5-session flow ratio <= 0

HIGH exit requires both:
- Climax: one-day return >= +1%, close location >= 0.70, intraday range >= 1.6%, distance above SMA200 >= 10%
- Extension/momentum fade: distance above SMA200 >= 10%, 3-session MACD histogram change <= 0, SMA50 10-session slope > 0

Corporate-action signal dates are suppressed. The candidate was discovered after inspecting the same historical sample; all results below remain exploratory.

## Reproduced full-period result

| Metric | Candidate |
|---|---:|
| Final equity | NT$1,512,473 |
| Total return | +1,412.5% |
| CAGR | 28.46% |
| Maximum drawdown | -45.64% |
| Closed trades | 23 |
| Winning trades | 87.0% |
| Exposure | 70.3% |

Matched buy-and-hold benchmark: NT$1,556,027 final equity, 28.79% CAGR, -45.68% maximum drawdown. The candidate underperforms buy-and-hold by 0.33 percentage points annualized while using 23 round trips instead of one continuous holding period. It does not demonstrate superior risk-adjusted performance or the 200% CAGR research target.

## Chronological segments

| Segment | Candidate CAGR | Candidate max DD | Buy-and-hold CAGR |
|---|---:|---:|---:|
| 2016–2020 | 30.38% | -23.89% | 29.90% |
| 2021–2026-11-09 | 23.04% | -45.55% | 27.72% |
| 2023–2026-11-09 | 39.50% | -24.09% | descriptive only |

The candidate slightly beats buy-and-hold in 2016–2020 but underperforms in 2021–2026. The 2023–2026 segment is not a clean untouched holdout because the candidate was selected after reviewing the full sample. It must not be presented as out-of-sample proof.

Annual candidate returns are uneven: 2017 -0.8%, 2022 -25.8%, while 2019 +40.6%, 2020 +50.8%, and 2025 +57.6%. The strategy is exposed to regime dependence and concentrated large trades.

## Execution and cost sensitivity

| Test | Final equity | CAGR | Max DD | Trades |
|---|---:|---:|---:|---:|
| Frozen next-open execution | NT$1,512,473 | 28.46% | -45.64% | 23 |
| One extra session delay | NT$1,298,122 | 26.66% | -45.66% | 23 |
| Two extra session delay | NT$1,449,207 | 27.95% | -45.67% | 23 |
| Slippage 0.2% each side | NT$1,444,458 | 27.91% | -45.67% | 23 |
| Slippage 0.3% each side | NT$1,379,747 | 27.37% | -45.65% | 23 |

Execution delay and cost stress do not reverse the conclusion: the candidate remains near, but below, buy-and-hold. It does not create an edge large enough to support deployment.

## One-factor neighbouring threshold checks

| Parameter | Tested values | CAGR results |
|---|---|---|
| LOW relative volume | 1.1 / 1.2 / 1.3 | 27.45% / 28.46% / 28.46% |
| LOW one-day return | -0.8% / -1.0% / -1.2% | 30.44% / 28.46% / 24.12% |
| LOW stochastic K | 25 / 30 / 35 | 28.69% / 28.46% / 28.46% |
| LOW close location | 0.20 / 0.25 / 0.30 | 28.46% / 28.46% / 27.44% |
| HIGH distance above SMA200 | 8% / 10% / 12% | 26.71% / 28.46% / 27.68% |
| HIGH one-day return | 0.8% / 1.0% / 1.2% | 24.18% / 28.46% / 28.46% |
| HIGH close location | 0.65 / 0.70 / 0.75 | 28.46% / 28.46% / 31.34% |
| HIGH intraday range | 1.4% / 1.6% / 1.8% | 24.65% / 28.46% / 27.28% |

This is one-factor-at-a-time sensitivity only. It is not a grid search, and the highest neighbouring result must not be promoted as a new best rule. Several thresholds produce identical results because the candidate fires infrequently; this is not evidence that the thresholds are intrinsically optimal.

## Signal evidence from 500 pivot/control audit

The corrected 500-pivot study contains 250 HIGH and 250 LOW pivots. After matching controls by time context, excluding corporate-action and pivot windows, clustering by control date, and applying Benjamini-Hochberg correction across rule/lead tests, none of the 55 simple trader-rule tests survives at q < 0.05.

The expanded feature audit shows a more interpretable pattern:

- LOW, one session before pivot: lower one-day return, lower close location, lower stochastic K, lower OBV flow, and price further below SMA5; relative volume is higher. These directions are consistent with a capitulation / selling-pressure profile.
- HIGH, one session before pivot: higher one-day return, higher close location, wider intraday range, and greater distance above SMA200. The MACD histogram change is more negative three sessions before the high, suggesting momentum deterioration can be useful as context.
- Year-by-year feature direction is more consistent for the immediate LOW pressure features than for the HIGH extension/fade features. The HIGH extension conjunction therefore remains a hypothesis, not a settled rule.

Foreign-investor and investment-trust features do not provide a stable improvement in the current TSMC turning-point models. In the small/large-wave OOF summary, adding foreign/trust features often lowers AUC or does not improve the corresponding metric. Keep foreign and trust flows as separate candidate features for further testing, but do not force them into the entry/exit rule. Dealer data remain excluded.

## Research decisions

1. Do not deploy this candidate as a live strategy. It is close to buy-and-hold, has a -45.6% drawdown, and has not passed a clean independent holdout.
2. Do not select a new threshold from the neighbour table. Freeze the current candidate as a benchmark hypothesis.
3. Next work should improve low/high pivot coverage and small/large wave discrimination without using retrospective pivot confirmation as a tradable signal.
4. Any new rule must be selected only on past data, then evaluated on later untouched years; report event recall, precision, missed-pivot distance, false-alert loss, CAGR, drawdown, exposure, and cost sensitivity together.
5. Before any paper-trading gate, add official Taiwan price-limit and suspension/locked-limit execution handling, cash dividends/corporate-action cashflows, and liquidity-aware fill constraints.

**Conclusion:** The data support a plausible LOW capitulation feature family. The current HIGH climax/extension exit is promising as a research hypothesis but not stable enough to prove persistent alpha. The current strategy has not achieved 200% annualized return and has not beaten buy-and-hold over the full test period.
