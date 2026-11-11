# TSMC 2330 — corrected Taiwan swing-trader event/control audit

## Strategy boundary

The core remains **all-in at a buyable swing low and fully out at a swing high**. No trend-held core, partial positions, shorting or leverage is introduced.

This is an exploratory diagnostic on 500 retrospectively labeled pivots (250 HIGH, 250 LOW), not a live strategy or P&L backtest. It tests whether publicly described Taiwan swing-trading ideas appear more often before pivots than on matched non-pivot dates.

## Important control-set correction

The first control sample did not exclude corporate-action windows, and its control dates were disproportionately drawn from those windows. That made the original event/control differences unreliable. Those earlier differences are superseded.

The corrected sample was rebuilt from the frozen TSMC causal feature and pivot files:
- 2,215 valid event/lead rows across 1, 3, 5, 10 and 20 sessions before pivots.
- Event and control feature dates inside corporate-action windows are excluded.
- Controls are excluded within 3 sessions of any frozen-threshold pivot and cannot reuse any sampled event feature date.
- Matching prioritizes calendar year and trend regime at the feature date. If that pool has fewer than five eligible dates, the matching window widens.
- 446 unique control dates are used; the maximum reuse is 32 rows. Inference therefore uses a control-date cluster bootstrap and a control-date cluster sign test, not an independent-pair McNemar test.
- Benjamini–Hochberg correction is applied across all 55 rule × lead tests.
- Foreign-investor and investment-trust rules use only complete event/control pairs dated on or after 2012-05-02. Dealer data is excluded.

## Taiwan trading ideas referenced

1. **蔣明誠 / volume-price swing idea**: published guidance describes low-volume accumulation and buying a pullback near the quarterly line; it also describes a high-volume bearish engulfing after a large advance as a distribution warning. The frozen sample lacks monthly revenue and an exact 60-day average, so the tested low-side rule (volume contraction near the prior 20-day low) and high-side blow-off rule are proxies, not exact reproductions.
2. **廖繼弘 / positive-bias exit**: the published 15% positive-bias idea is tested as `distance_sma_20 >= 15%`. It occurs too rarely to evaluate in this sample.
3. **RSI/stochastic oversold and RSI/MACD/volume divergence**: fixed conventional thresholds and causal divergence flags are tested separately.
4. **Foreign-investor and investment-trust flows**: five-day combined buying/selling rules are tested separately.

References:
- ScanTrader, 蔣明誠, “利用成交量看出主力動向”: https://scantrader.com/article/01855d225129000005f6000000000000
- 廖繼弘 positive-bias discussion: https://my.stust.edu.tw/sys/read_attach.php?id=508340
- Taiwan swing-trading indicator overview: https://stock.epoch.tw/glossary/swing-trading

## Corrected findings

| Rule before pivot | Side | Lead | Event hit rate | Matched-control rate | Difference | 95% cluster-bootstrap CI | BH q |
|---|---|---:|---:|---:|---:|---:|---:|
| Volume dry-up near prior 20-day low | LOW | 1 | 11.2% | 20.3% | -9.1 pp | -17.5 to -1.2 pp | 0.375 |
| Volume dry-up near prior 20-day low | LOW | 3 | 14.0% | 19.6% | -5.5 pp | -13.9 to +2.2 pp | 1.000 |
| Volume dry-up near prior 20-day low | LOW | 5 | 17.5% | 18.0% | -0.4 pp | -8.3 to +7.4 pp | 1.000 |
| RSI <= 35 and stochastic K <= 25 | LOW | 1 | 9.1% | 4.2% | +5.0 pp | 0.0 to +9.8 pp | 0.172 |
| Any bearish RSI/MACD/volume divergence | HIGH | 1 | 20.5% | 16.4% | +4.1 pp | -3.9 to +12.0 pp | 0.375 |
| Volume blow-off + negative return + weak close | HIGH | 3 | 6.3% | 1.3% | +5.1 pp | +1.7 to +8.8 pp | 0.172 |

**No positive rule survives the 55-test multiple-comparison correction at q < 0.05.**

Interpretation:
- The earlier apparent low-point advantage for volume dry-up disappears after correcting the control group. In this corrected sample, it is no more common before lows than in controls; at lead 1 it is actually less common.
- A short-horizon high-volume blow-off pattern is the strongest high-side candidate, and oversold RSI/stochastic is a possible near-low clue. Both remain exploratory because their adjusted q-values are 0.172.
- Bearish divergence alone is not statistically robust in this corrected comparison.
- Foreign + trust both buying before lows and both selling before highs are not supported as simple directional filters.
- The 50-day moving-average pullback/positive-close proxy is strongly less common before lows than controls. This is not a faithful test of the published quarterly-line strategy because the exact 60-day line and monthly revenue conditions are unavailable.
- The 15% positive-bias rule is too rare to evaluate.

## Consequence for the all-in/all-out system

Do not promote volume dry-up, divergence, or institutional flow as confirmed buy/sell rules. The separate full-period next-open backtest of the six fixed rule combinations remains exploratory: the best tested pair returned about +10.2% CAGR, while price-only buy-and-hold returned about +28.8% CAGR over the same period and assumptions. The best pair is not validated and does not meet the 200% annualized target.

The next research task is to improve pivot labels and signal coverage, especially small-wave and large-wave separation, then evaluate candidate alerts over every trading day and run a frozen chronological test. The final independent validation must be forward paper trading because this historical sample has already been inspected.
