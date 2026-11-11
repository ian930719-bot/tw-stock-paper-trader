# TSMC 2330 — swing LOW/HIGH signal discovery v2

## Strategy requirement

The research target remains unchanged: **buy all-in near a genuine swing LOW, then sell the entire position near a genuine swing HIGH**. No trend-following core, partial positions, shorting or leverage.

This report supersedes the first event/control findings, which used a biased control sample.

## 1. Corrected event/control sample

- TSMC 2330, frozen pivot threshold `p0.020_atr1.0`.
- 500 selected pivots: 250 HIGH and 250 LOW.
- 2,215 valid event/lead observations across 1, 3, 5, 10 and 20 sessions before the pivot.
- Event and control feature dates in corporate-action windows are excluded.
- Controls exclude the +/-3-session neighborhood of any frozen pivot and all sampled event feature dates. Matching prioritizes calendar year and `trend_stack` at the feature date; if fewer than five eligible controls exist, the window widens.
- 446 unique control dates; maximum reuse 32 observations. Statistical inference uses a control-date cluster bootstrap and cluster sign test. P-values are BH-adjusted across the rule/lead family.
- The 63-feature scan includes technical indicators plus foreign-investor and investment-trust features. Dealer, broker and combined institutional fields are excluded. Flow tests use complete event/control pairs from 2012-05-02 onward.

**Correction:** the earlier control sample did not exclude corporate-action windows and was disproportionately composed of those dates. The earlier claim that low-volume dry-up was a strong LOW signal is withdrawn.

## 2. What the full feature scan found

A scan of 63 technical/foreign/trust features across two pivot directions and five lead times produced 630 feature/side/lead tests. Sixty-two passed the exploratory BH q < 0.05 threshold. Many are correlated versions of the same underlying information; they are not 62 independent discoveries.

### LOW profile

One to five sessions before LOW pivots, the sample more often showed:
- Negative short-term return and weak close location.
- Low stochastic K and weak short-term OBV flow.
- Higher relative volume and volume z-score, consistent with a capitulation sell-off rather than quiet volume contraction.

Examples at one session before LOW:
- `return_1`: event-control standardized difference about -0.44.
- `close_location`: about -0.52.
- `stoch_k14`: about -0.52.
- `relative_volume20`: about +0.41.
- `obv_flow_ratio_5`: about -0.37.

Direction stability across the 17 calendar years was 15/17 years for negative `return_1`, 15/17 for weak `close_location`, 14/17 for higher `relative_volume20`, and 13/17 for lower `stoch_k14`.

### HIGH profile

Before HIGH pivots, the profile differs:
- At lead 1, returns and close location are still strong, and price is extended above the 200-day average.
- At lead 3, MACD histogram momentum change is more often negative while long-term extension and intraday range remain elevated.
- At leads 5–20, trend/volatility extension features remain elevated; this looks more like an advancing move losing momentum than an obvious bearish candle at the exact top.

Examples:
- HIGH lead 1: `return_1` standardized difference about +0.31; `close_location` +0.34; `distance_sma_200` +0.28.
- HIGH lead 3: `macd_hist_change_3` about -0.27.
- HIGH lead 10: `sma50_slope_10` is positive on average, but year-to-year consistency is weaker.

`bearish_divergence` alone is not a sufficiently robust high-point rule in the corrected comparison. A simple “foreign/trust net selling = top” rule is also unsupported.

## 3. New candidate rule combinations

These combinations were derived after inspecting the same feature sample. Their q-values are exploratory and do not constitute independent validation.

### LOW candidate: capitulation sell-off

Base form:
- `relative_volume20 >= 1.2`
- `return_1 <= -1%`
- `close_location <= 0.25`

Higher-confidence form adds:
- `stoch_k14 <= 30`
- `obv_flow_ratio_5 <= 0`

In the 40-test combination scan, the base high-volume sell-off rule appeared on 16.2% of LOW event observations at lead 1 versus 2.5% of matched controls; at lead 5, 7.0% versus 0.4%. The stricter capitulation rule appeared on 11.6% versus 1.7% at lead 1.

### HIGH candidate: climax plus extension/momentum fade

Climax:
- `return_1 >= +1%`
- `close_location >= 0.70`
- `intraday_range_pct >= 1.6%`
- `distance_sma_200 >= 10%`

Extension/momentum fade:
- `distance_sma_200 >= 10%`
- `macd_hist_change_3 <= 0`
- `sma50_slope_10 > 0`

The same-day conjunction of these two high-side conditions is rare, but showed a lead-1 event hit rate of 5.3% versus 0% of matched controls. The broader extension/momentum-fade rule showed enrichment at leads 5, 10 and 20. Thresholds were derived from this same sample, so these results are hypothesis generation only.

## 4. All-in/all-out execution test

The best tested candidate is:
- **Buy:** capitulation + negative OBV flow.
- **Sell:** climax + extension/momentum fade conjunction.

Execution assumptions:
- Signal at session close; execute next session open.
- All available cash; integer shares; no leverage or shorting.
- Commission 0.1425% per side, NT$20 minimum; sell tax 0.3%; slippage 0.1% per side.
- Raw OHLC fills; corporate-action signal dates are excluded.
- No stop/time exit; only the HIGH signal or final test-day close.
- Price-only accounting; cash dividends, official price-limit fills, liquidity impact and odd-lot microstructure are not modeled.

| Strategy | Final equity from NT$100k | CAGR | Max drawdown | Trades | Win rate |
|---|---:|---:|---:|---:|---:|
| Capitulation + climax/extension | NT$1,512,473 | 28.46% | -45.64% | 23 | 87.0% |
| Buy-and-hold benchmark | NT$1,556,027 | 28.79% | -45.68% | 1 | — |

The candidate is close to, but still below, buy-and-hold. It is not a 200% CAGR strategy.

### Pivot proximity and coverage

Among its 23 completed trades:
- 21/23 entries were within 3 sessions of the nearest retrospective LOW pivot; 22/23 within 5 sessions.
- 14/23 exits were within 3 sessions of the nearest retrospective HIGH pivot; 18/23 within 5 sessions.
- Median holding time was 41 calendar days; mean 122 days, reflecting one long hold.

However, the signals are rare:
- At a 5-session horizon, LOW alert precision was 61.8% but recall only 4.7%.
- HIGH alert precision was 72.1% but recall only 3.5%.

This is a high-confidence, low-coverage candidate. It catches a small subset of pivots and misses most small and large turns. The trade proximity is measured against retrospective pivot labels on the same historical sample and is therefore not out-of-sample proof.

### Subperiod check

| Period | Candidate CAGR | Buy-and-hold CAGR |
|---|---:|---:|
| 2016–2020 | 30.38% | 29.90% |
| 2021–2026-11-09 | 23.04% | 27.72% |

The candidate slightly exceeded buy-and-hold in the earlier subperiod but underperformed in the later period. It does not demonstrate stable outperformance.

## 5. Decision and next research gate

Keep the all-in/all-out design. Do not deploy these rules as a finished strategy yet.

The next objective is to improve **coverage without destroying precision**:
1. Separate small-wave and large-wave pivots using reversal amplitude, then discover distinct LOW/HIGH profiles.
2. Build a calibrated daily probability/score from the full technical feature set, with training data strictly earlier than each test year.
3. Compare a high-confidence trigger with a broader confirmation trigger, measuring missed-pivot distance and false-alert cost.
4. Freeze rules before any future forward paper-trading evaluation.

The current candidate's 28.46% CAGR is only an exploratory result, below the same-period buy-and-hold benchmark and far below the 200% annualized target.
