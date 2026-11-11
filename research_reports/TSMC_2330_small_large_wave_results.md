# TSMC 2330 — small-wave / large-wave walk-forward results

## Run and reproducibility

- GitHub Actions run: [run 17](https://github.com/ian930719-bot/tw-stock-paper-trader/actions/runs/43023902206)
- Result artifact: [tsmc-2330-daily-walkforward-17](https://github.com/ian930719-bot/tw-stock-paper-trader/actions/runs/43023902206/artifacts/13856806362)
- Run conclusion: success.
- Source data: frozen TSMC 2330 causal feature and pivot artifact; 4,128 daily rows, 2010-01-04 through 2026-11-09; 317 HIGH and 319 LOW pivots at `p0.020_atr1.0`.
- Rebuilt 500-pivot control sample and all unit tests passed before model execution.
- The workflow completed corrected event/control audit, all-feature cluster audit, prior fixed-rule simulation, small/large wave walk-forward, and the earlier daily walk-forward.

## Method

- Wave magnitude = `abs(reversal_pct)` from the retrospective pivot event file.
- In each fold, for HIGH and LOW separately, the 35th percentile of prior pivot magnitudes defines small waves; the 65th percentile defines large waves. The middle 30% is omitted from class-specific positive labels.
- The cutoffs are computed only from pivots before the two-year validation window and held fixed through that fold's validation and test year.
- Four independent targets: HIGH-small, HIGH-large, LOW-small, LOW-large.
- Horizons: 1, 3, 5, 10, 20 sessions.
- Expanding training set, previous two years for score-threshold calibration, next calendar year held out. Training labels are purged by forecast horizon.
- Fixed score threshold: 97th percentile of validation scores. Validation labels do not tune the threshold.
- Models: technical features; technical plus foreign-investor and investment-trust features. Dealer and broker features are excluded. Institutional data begins 2012-05-02.
- Execution policies: small-only, large-only, and union of both wave classes. All-in/all-out; signal at close, execute next open; integer shares; estimated 0.1425% commission each side (NT$20 minimum), 0.3% sell tax, 0.1% slippage each side; corporate-action signal dates excluded; final position liquidated at final close.
- Backtest period: 2016-01-04 through 2026-11-09; 2026 is partial.

## Walk-forward model quality

Values are means across annual held-out folds. Precision, recall and false-positive rate are daily event-proximity classification metrics, not trade win rates.

| Side | Wave | Horizon | Model | Folds | Mean AUC | Precision | Recall | FPR | Base rate |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| HIGH | Large | 1 | Technical | 11 | 0.703 | 7.4% | 8.7% | 3.8% | 3.0% |
| HIGH | Large | 1 | Technical + foreign/trust | 10 | 0.579 | 18.4% | 9.6% | 3.5% | 3.2% |
| HIGH | Large | 5 | Technical | 11 | 0.566 | 22.8% | 6.4% | 4.5% | 14.6% |
| HIGH | Large | 10 | Technical | 11 | 0.543 | 55.3% | 6.7% | 3.6% | 26.3% |
| HIGH | Small | 1 | Technical | 11 | 0.678 | 8.3% | 19.5% | 4.6% | 2.0% |
| HIGH | Small | 5 | Technical | 11 | 0.525 | 5.1% | 6.7% | 7.4% | 10.0% |
| HIGH | Small | 10 | Technical | 11 | 0.538 | 20.9% | 5.9% | 6.8% | 18.7% |
| LOW | Large | 1 | Technical | 11 | 0.720 | 8.5% | 7.6% | 4.7% | 2.7% |
| LOW | Large | 1 | Technical + foreign/trust | 10 | 0.713 | 7.4% | 14.0% | 4.6% | 2.9% |
| LOW | Large | 5 | Technical | 11 | 0.597 | 21.0% | 7.1% | 4.7% | 13.1% |
| LOW | Large | 10 | Technical | 11 | 0.593 | 46.1% | 8.2% | 4.6% | 23.9% |
| LOW | Small | 1 | Technical | 11 | 0.626 | 1.0% | 4.1% | 5.7% | 1.8% |
| LOW | Small | 5 | Technical | 11 | 0.513 | 17.0% | 8.8% | 9.1% | 9.1% |
| LOW | Small | 10 | Technical | 11 | 0.530 | 19.4% | 7.9% | 9.4% | 17.4% |

### Interpretation

- The most discriminative models are **LOW-large at 1 session (AUC 0.720)** and **HIGH-large at 1 session (AUC 0.703)**. This suggests large swing turns have more detectable short-horizon precursors than small turns in this sample.
- HIGH-small at 1 session also has useful ranking signal (AUC 0.678) and the strongest small-wave recall shown (19.5%), but precision is only 8.3%.
- LOW-small is weak: AUC is near chance at 3–10 sessions, and short-horizon precision is very low.
- Large-wave models at longer horizons often show high precision because the target base rate rises substantially with horizon. Recall remains low (generally 5–14%), so they still miss most pivots.
- Adding foreign-investor and investment-trust features does **not** consistently improve performance. In several folds/horizons it lowers AUC or recall. It is not a universal filter.
- AUC and alert precision must not be confused with trading returns. They are retrospective pivot-proximity diagnostics.

## Cost-aware all-in/all-out results

Starting capital NT$100,000. Every strategy uses the same period and estimated transaction costs.

| Model | Horizon | Policy | Final equity | CAGR | Max drawdown | Trades | Win rate |
|---|---:|---|---:|---:|---:|---:|---:|
| Technical | 3 | Small only | NT$853,244 | 21.85% | -45.63% | 19 | 73.7% |
| Technical | 5 | Small only | NT$787,831 | 20.96% | -45.65% | 21 | 71.4% |
| Technical | 10 | Small only | NT$824,318 | 21.47% | -45.67% | 16 | 75.0% |
| Technical | 3 | Large only | NT$623,654 | 18.38% | -45.64% | 14 | 85.7% |
| Technical | 3 | All waves | NT$678,055 | 19.30% | -45.64% | 35 | 74.3% |
| Technical + foreign/trust | 5 | Small only | NT$603,243 | 18.02% | -30.81% | 7 | 85.7% |
| Technical + foreign/trust | 3 | Large only | NT$509,779 | 16.20% | -45.66% | 15 | 80.0% |
| Technical + foreign/trust | 20 | All waves | NT$426,931 | 14.32% | -37.41% | 17 | 70.6% |
| Buy-and-hold | — | Benchmark | NT$1,556,027 | 28.79% | -45.68% | 1 | — |

The table shows the leading fixed policies, not an independently selected winner. The highest historical CAGR among the tested policies is technical small-only at horizon 3, but it is still below buy-and-hold by about 6.94 percentage points annually. All wave policies tested underperform buy-and-hold.

## Decision

1. **Keep small and large waves separate.** Large pivots are easier to rank than small pivots in the one-session models, while small-wave signals have different precision/recall behavior.
2. **Do not make foreign/trust flow mandatory.** Its addition did not consistently improve model or trading performance.
3. **Do not deploy this model yet.** The best tested all-in/all-out CAGR is 21.85%, below the 28.79% buy-and-hold benchmark. Several model variants have low recall and small trade counts.
4. Next improve small-wave coverage and large-wave exit timing, add pivot-distance and false-alert drawdown diagnostics, then freeze the model and perform forward paper trading. Earlier feature discovery has already inspected this historical period, so this walk-forward is not an untouched independent validation.
5. No 200% annualized return has been achieved or established.

## Reproducibility

- Model: [`tsmc_2330_small_large_wave_walkforward.py`](https://github.com/ian930719-bot/tw-stock-paper-trader/blob/research/200cagr-blind-validation/tsmc_2330_small_large_wave_walkforward.py)
- Tests: [`tests/test_tsmc_2330_small_large_wave_walkforward.py`](https://github.com/ian930719-bot/tw-stock-paper-trader/blob/research/200cagr-blind-validation/tests/test_tsmc_2330_small_large_wave_walkforward.py)
- Workflow: [`.github/workflows/tsmc-2330-daily-walkforward.yml`](https://github.com/ian930719-bot/tw-stock-paper-trader/blob/research/200cagr-blind-validation/.github/workflows/tsmc-2330-daily-walkforward.yml)
