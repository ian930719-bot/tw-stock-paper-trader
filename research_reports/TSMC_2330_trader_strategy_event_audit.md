# TSMC 2330 — Taiwan swing-trader rule event/control audit

## Strategy boundary

The core remains **all-in at a buyable swing low and fully out at a swing high**. This study does not replace it with trend-following, partial positions, or a permanent holding.

This is a rule-level diagnostic using a frozen sample of 500 retrospectively labeled pivots (250 HIGH, 250 LOW), not a live strategy or P&L backtest. It tests whether publicly described Taiwan swing-trading ideas leave measurable footprints in features observed before pivots.

## Data and matching

- Symbol: TSMC 2330; frozen threshold: `p0.020_atr1.0`.
- 250 HIGH and 250 LOW events, evenly spaced through the available event chronology.
- Feature observations: 1, 3, 5, 10 and 20 sessions before each pivot.
- Each event/lead pair has one non-pivot control, matched by calendar year and `trend_stack` regime where possible. Controls were excluded within 10 sessions of any pivot under the frozen threshold.
- The source feature builder explicitly uses OHLCV through the current session close; divergence flags compare current values with shifted prior 20-session extrema. The pivot label is retrospective, but it is not used as a feature.
- Rates below are **event-hit rates versus matched-control hit rates**, not live alert precision. The sample is case/control-balanced and the denominator is selected pivot events, not every trading day.
- Confidence intervals use 5,000 paired event-row bootstrap resamples. Exact paired McNemar p-values are adjusted across all 55 rule × lead comparisons using Benjamini–Hochberg. Year-by-year differences are stability diagnostics, not independent replications.
- Institutional variables use foreign investors and investment trusts only. Dealer features are excluded.

## Taiwan trading ideas referenced

1. **蔣明誠 / volume-price swing idea**: published guidance describes low-volume accumulation/box consolidation and buying a pullback near the quarterly line; it also describes a high-volume bearish engulfing after a large advance as a distribution warning. The frozen snapshot lacks monthly revenue and an exact 60-day average, so the low-side proxy is volume contraction near the prior 20-day low (`relative_volume20 <= 0.8` and `distance_prior_low20 <= 5%`). The high-side blow-off proxy is relative volume >= 1.5, negative one-day return, and weak close location. These are proxies, not exact reproductions.
2. **廖繼弘 / positive-bias exit**: the published 15% positive-bias exit idea is tested as `distance_sma_20 >= 15%`. It is too rare in this sample to evaluate.
3. **RSI/stochastic oversold and divergence**: fixed conventional thresholds and precomputed causal divergence flags are tested separately.
4. **Foreign-investor and investment-trust flows**: five-day combined buying/selling rules are tested separately. No dealer data is used.

References:
- ScanTrader, 蔣明誠, “利用成交量看出主力動向”: https://scantrader.com/article/01855d225129000005f6000000000000
- 廖繼弘 positive-bias discussion: https://my.stust.edu.tw/sys/read_attach.php?id=508340
- Taiwan swing-trading indicator overview: https://stock.epoch.tw/glossary/swing-trading

## Main findings

### Candidate low-point clues

| Rule before LOW pivot | Lead | Event hit rate | Matched-control rate | Paired difference | 95% bootstrap CI | BH q |
|---|---:|---:|---:|---:|---:|---:|
| Volume dry-up near prior 20-day low | 1 | 11.2% | 4.4% | +6.8 pp | +2.4 to +11.6 pp | 0.0299 |
| Volume dry-up near prior 20-day low | 3 | 13.2% | 2.0% | +11.2 pp | +7.2 to +15.6 pp | <0.001 |
| Volume dry-up near prior 20-day low | 5 | 16.0% | 2.0% | +14.0 pp | +9.6 to +18.8 pp | <0.001 |
| Volume dry-up near prior 20-day low | 10 | 14.4% | 3.2% | +11.2 pp | +6.4 to +16.4 pp | 0.0002 |
| Volume dry-up near prior 20-day low | 20 | 12.8% | 1.6% | +11.2 pp | +7.2 to +15.6 pp | <0.001 |

This is the strongest and most consistent low-side clue in this sample. At the 5-session lead, 16% of selected lows showed the rule versus 2% of matched controls; the paired difference was +14 percentage points. The difference was positive in 16 of 17 calendar years at this lead. It still misses 84% of selected lows, so it cannot be used alone as a full-entry signal.

Other low-side observations:
- RSI <= 35 plus stochastic K <= 25 has a small near-pivot effect at lead 1, but it does not persist at longer leads.
- Bullish RSI/MACD/volume divergence alone is not consistently different from controls.
- Foreign + trust both net-buying over five days does not reliably precede lows.
- A 50-day-average pullback/positive-close proxy performs worse than controls at short leads. It is not evidence for the published quarterly-line setup because the exact 60-day average and monthly revenue conditions are absent.

### Candidate high-point clues

| Rule before HIGH pivot | Lead | Event hit rate | Matched-control rate | Paired difference | 95% bootstrap CI | BH q |
|---|---:|---:|---:|---:|---:|---:|
| Any bearish RSI/MACD/volume divergence | 1 | 21.2% | 9.2% | +12.0 pp | +5.6 to +18.4 pp | 0.0036 |
| Volume blow-off + negative return + weak close | 3 | 6.0% | 0.4% | +5.6 pp | +2.8 to +8.8 pp | 0.0036 |
| Volume blow-off + negative return + weak close | 5 | 4.4% | 0.8% | +3.6 pp | +0.8 to +6.4 pp | 0.0772 |

Bearish divergence is a near-pivot warning, not a reliable 3–20 session early signal: its difference shrinks sharply after lead 1. The blow-off proxy is rare but shows a lead-3 enrichment. The exact 15% bias rule produces at most one event in these snapshots and cannot be evaluated.

Foreign and trust both net-selling over five days was less common before highs than in matched controls at lead 1 (16.8% vs 26.4%). A simple “institutions sell = top” rule is rejected for this sample.

## Implications for all-in / all-out

1. **Low side:** volume contraction near a prior low is the best current candidate. Pair it with a separate reversal-confirmation condition (e.g. positive close, regain of a short-term level, or improving OBV flow); it alone catches only a minority of lows.
2. **High side:** bearish divergence is strongest very close to the high; the blow-off proxy is a rarer lead-3 warning. Test them as independent exit triggers and as a conjunction; do not assume both must occur.
3. **Do not force institutional-flow filters.** Foreign/trust flows did not produce a robust simple directional rule here.
4. **Do not change the live trading system yet.** Event/control enrichment is not trade performance. A rule can be more common near pivots and still lose money after next-open execution, costs, missed pivots and false alerts.

## Next research gate

Freeze the candidate definitions before trying combinations. Build a daily all-market-day signal panel, measure alert precision/recall against the frozen pivot labels without case/control sampling, then run chronological walk-forward evaluation. Only after thresholds and combination logic are frozen should the strategy be simulated with all available capital, integer shares, next-session executable prices, Taiwan price limits, commission, tax, slippage, corporate actions and dividends. Report missed-pivot distance and false-alert drawdown. Since historical years and this pivot sample have already been inspected, the final independent check must be forward paper trading.

No annualized-return target is claimed by this event study.
