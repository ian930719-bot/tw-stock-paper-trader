# 2016–2026 Existing Historical Dataset: Preliminary Annual Top-20 Portfolio Diagnostic

**Status: RESEARCH_ONLY — not a research-grade result and not a live-trading claim.**

## Dataset reused (no fresh full-history download)

- Source: FinMind API v4, `TaiwanStockPrice`, raw/unadjusted OHLCV execution bars.
- Period present in the archive: 2010-01-04 to 2026-11-09.
- Candidate symbols: 2,220.
- Symbols with price rows: 2,189; empty files: 31.
- Price rows: 6,903,229.
- Turnover nulls in non-empty bar files: 0.
- Duplicate dates: 0.
- OHLC/positive-price/negative-turnover validation failures: 0.
- Per-file SHA-256 checks against `download_manifest.json`: 0 mismatches.
- Corporate-action event rows: 22,624; conflicts: 0.

Archive checked locally: `taiwan-history-229.zip`, SHA-256
`2b77cd075c82d5443ac221f6938314a053622750179bbeb562f9d303696984e1`.

## Important data limitations

The manifest says the candidate bar collection completed, but this is not equivalent to a complete point-in-time exchange universe:

- `point_in_time_universe_complete=false`.
- `delisted_price_coverage_verified=false`.
- The candidate list contains 2,162 current common-equity candidates and 58 delisted-common-equity candidates; it is not a verified historical membership census.
- The global `TaiwanStockCapitalReductionReferencePrice` dataset failed to download, so `corporate_actions_complete=false`.
- The available corporate-action event file does not establish complete cash-dividend adjustment coverage.
- 31 candidates have no usable price file.
- 2026 is partial through 2026-11-09.

Consequently the annual top-20 lists below are **candidate-universe rankings**, not certified survivorship-bias-free historical constituents.

## Annual top-20 universe

For year Y, rank symbols by total turnover in calendar year Y-1; no current-year turnover is used. The ranking source years are 2015–2025 for portfolio years 2016–2026. Each of the 11 years produced 20 candidates (220 rows total). The ranking code excludes symbols with fewer than 100 prior-year turnover observations.

## Preliminary shared-capital diagnostic

Local diagnostic used the same causal outline as the new research runner: signal known at close, entry no earlier than next available open, one shared TWD 100,000 account, integer shares, max five positions, 0.5% risk budget, 20% max position, 1% of lagged 20-day average turnover participation, ATR trailing stop, 60-bar maximum holding period, 0.1425% buy fee, 0.4425% sell fee plus tax, and 0.1% slippage per side. Stress cases multiply fees/tax and slippage by 1.5 and 2.0.

Period: 2016-01-04 through 2026-11-09 (2026 partial).

| Strategy | Cost | Ending equity (TWD) | Net CAGR | Max drawdown | Trades | Avg exposure |
|---|---:|---:|---:|---:|---:|---:|
| Trend breakout | 1.0x | 105,141 | 0.46% | -14.79% | 275 | 10.84% |
| Momentum rotation | 1.0x | 104,964 | 0.45% | -22.93% | 1,200 | 21.23% |
| Trend pullback | 1.0x | 99,687 | -0.03% | -3.32% | 35 | 0.73% |
| Mean reversion | 1.0x | 95,753 | -0.40% | -5.23% | 33 | 0.44% |
| Squeeze breakout | 1.0x | 98,110 | -0.18% | -12.54% | 259 | 8.68% |
| Trend breakout | 1.5x | 98,214 | -0.17% | -17.75% | 275 | 10.85% |
| Momentum rotation | 1.5x | 83,349 | -1.67% | -30.61% | 1,196 | 21.15% |
| Trend pullback | 1.5x | 98,811 | -0.11% | -3.63% | 35 | 0.73% |
| Mean reversion | 1.5x | 95,080 | -0.46% | -5.85% | 33 | 0.44% |
| Squeeze breakout | 1.5x | 92,301 | -0.74% | -15.63% | 258 | 8.67% |
| Trend breakout | 2.0x | 91,446 | -0.82% | -20.80% | 275 | 10.82% |
| Momentum rotation | 2.0x | 64,495 | -3.96% | -44.81% | 1,194 | 20.84% |
| Trend pullback | 2.0x | 96,618 | -0.32% | -3.96% | 35 | 0.61% |
| Mean reversion | 2.0x | 94,403 | -0.53% | -6.48% | 33 | 0.44% |
| Squeeze breakout | 2.0x | 86,507 | -1.33% | -19.25% | 257 | 8.59% |

These results do **not** meet the 100% net CAGR target. Even the best base-cost diagnostic is below 1% CAGR. Higher cost stress worsens most strategy results, particularly momentum rotation.

## Interpretation and next work

1. Do not increase risk or tune thresholds on the full 2016–2026 result to force the target; that would overfit.
2. First resolve the point-in-time universe and delisted-price coverage, capital-reduction reference prices, and dividend/corporate-action completeness.
3. Reproduce this diagnostic from the committed runner inside GitHub Actions using the same archived dataset and compare trade ledgers/equity curves.
4. Then implement nested chronological walk-forward selection, frozen candidates, annual subperiod reporting, and parameter-neighbourhood robustness. Since 2023–2026 has already been inspected in earlier work, it is not a clean blind holdout for any newly tuned version.
5. Only a fully costed, reproducible, continuous shared-capital result with verified data and no leakage can be considered against the 100% net CAGR hurdle.

This report records a preliminary diagnostic from the existing downloaded archive. It does not certify the dataset as complete, nor does it claim the results have passed the repository's full CI/reproducibility pipeline.
