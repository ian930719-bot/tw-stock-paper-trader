# TSMC 2330 Swing Turning-Point Event Study V2

Date: 2026-11-10

## Purpose

This study tests whether indicators observed at date t help identify a swing-low or swing-high event over the following 40 sessions. It is a predictive-label study, not a backtest and not a claim that the future turning point is executable at its exact price.

## Data and limitations

- Yahoo Finance auto-adjusted daily OHLCV for 2330, 2010-01-04 through 2026-11-10 (4,101 rows).
- Indicators use information available through date t only.
- The source has no turnover field. Adjusted OHLCV is not official raw exchange execution data.
- Event windows overlap. Daily observations are therefore not independent.
- This report uses a simple 20-session signal de-duplication rule: after accepting a signal, ignore further signals for the next 20 sessions. This reduces repeated counting within the same decline/rally but is not a perfect event-clustering algorithm.

## Forward event labels

For each date t:
- Low event: the lowest low in sessions t+1 to t+20 is followed by a maximum high within the next 20 sessions that is at least 10% above that low.
- High event: the highest high in sessions t+1 to t+20 is followed by a minimum low within the next 20 sessions that is at least 10% below that high.

These labels intentionally use future data only to define the outcome. They must never be used as contemporaneous trading signals.

## Base rates

| Period | Eligible dates | Low-event rate | High-event rate |
|---|---:|---:|---:|
| Train 2010-2020 | 2,623 | 45.2% | 15.4% |
| Validation 2021-2023 | 728 | 38.6% | 35.2% |
| Historical test 2024-2026-11-10 | 691 | 69.6% | 22.3% |

The event base rate is regime-dependent. Comparing a signal rate with the base rate of the same period is essential.

## Low-event signal rates after 20-session de-duplication

| Condition | Train rate (signals) | Validation rate (signals) | Historical test rate (signals) |
|---|---:|---:|---:|
| 20-session return < -8% | 44.4% (18) | 27.3% (11) | 66.7% (6) |
| Close below MA20 by >5% | 47.4% (19) | 12.5% (8) | 75.0% (8) |
| 20-session return < -8% AND volume ratio >1.5 | 70.0% (10) | 25.0% (8) | 60.0% (5) |
| 20-session return < -8% AND volume ratio >1.5 AND close below MA20 by >5% | 62.5% (8) | 16.7% (6) | 60.0% (5) |

Volume ratio is current volume divided by trailing 20-session average volume.

## High-event signal rates after 20-session de-duplication

| Condition | Train rate (signals) | Validation rate (signals) | Historical test rate (signals) |
|---|---:|---:|---:|
| RSI >70 | 17.7% (62) | 18.2% (11) | 26.3% (19) |
| 20-session return >8% | 22.2% (45) | 30.0% (10) | 31.8% (22) |
| Close above MA20 by >5% | 19.0% (42) | 18.2% (11) | 29.4% (17) |
| 20-session return >8% AND RSI >65 | 24.3% (37) | 37.5% (8) | 33.3% (15) |
| 20-session return >8% AND close above MA20 by >5% | 23.3% (30) | 28.6% (7) | 33.3% (15) |

## Interpretation

1. Earlier raw daily hit rates overstated evidence because many adjacent dates described the same market episode. De-duplicated results are the more conservative view.
2. The low-event condition combining a sharp 20-session decline and high volume had 70.0% in train, but only 25.0% in validation and 60.0% in the later historical test. It is not stable enough to call a robust rule.
3. For highs, momentum/extension combinations show only modest lift and remain inconsistent across periods.
4. The event base rate itself changes substantially by period. A raw probability without same-period baseline and event count is misleading.
5. The historical test period has already been examined during prior research; it is not a pristine blind holdout.

## Next research

- Replace the simple de-duplication rule with explicit turning-point episode clustering and report confidence intervals.
- Use ATR-normalized drawdown/reversal sizes and separate trend regimes.
- Test RSI/MACD divergence, Bollinger width, volume climax followed by contraction, and volatility compression.
- Select rules only on training data, freeze them, and evaluate validation. Do not choose parameters based on the historical test result.
- Run a prospective paper-signal log after parameters are frozen.
- Re-run execution backtests with official raw exchange data, corporate-action handling, trading limits, tick sizes, suspensions, and realistic order fills.

## Conclusion

No high-confidence, cross-period swing-low/high formula has been established yet. The next objective is not to maximize a single historical hit rate; it is to find a repeatable increase over the same-period base rate with sufficient independent episodes and acceptable drawdown.
