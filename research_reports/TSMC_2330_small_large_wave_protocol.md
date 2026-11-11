# TSMC 2330 — small-wave / large-wave research protocol

## Research objective

Preserve the user's required trading structure: buy the entire position near a swing LOW and sell the entire position near a swing HIGH. The new experiment separates small and large swing amplitudes so one signal definition is not forced to predict both.

## Wave-size definition

Use the frozen pivot event file at threshold `p0.020_atr1.0`. The wave magnitude is `abs(reversal_pct)` for each confirmed pivot. For every chronological test fold and separately for HIGH and LOW pivots:

- Calculate the 35th percentile of pivot magnitudes using only events dated before the validation window; pivots at or below it are classified as **small**.
- Calculate the 65th percentile from that same prior-only sample; pivots at or above it are classified as **large**.
- The middle 30% is retained as neither small nor large for class-specific model targets.
- The cutoffs are frozen for the fold and applied to its validation and test events. If fewer than 30 historical pivots exist for either side, or the cutoffs collapse due to ties, that fold is skipped.

This avoids using future pivots to define the magnitude thresholds. The magnitude is retrospective and only becomes known after the reversal is confirmed; it is a target label, never a model input.

## Chronological modelling

- Test years: 2016 onward, including partial 2026.
- Training: all eligible earlier daily rows before the validation window.
- Calibration: the preceding two calendar years.
- Test: the next calendar year.
- Purge training observations whose forward labels could overlap the validation window by the target horizon.
- Horizons: 1, 3, 5, 10 and 20 sessions.
- Separate binary classifiers for LOW-small, LOW-large, HIGH-small and HIGH-large.
- Two feature sets: technical indicators only; technical indicators plus foreign-investor and investment-trust features. Dealer, broker and proprietary-dealer fields are excluded. Institutional models use observations from 2012-05-02 onward.
- Model scores are calibrated to a fixed 97th-percentile alert-rate threshold using the validation score distribution. Validation labels are not used to select the threshold.

## Execution policies

For each horizon and feature set, test three policies:
1. **Small only:** LOW-small buys all-in; HIGH-small sells all.
2. **Large only:** LOW-large buys all-in; HIGH-large sells all.
3. **All waves:** either LOW class buys all-in; either HIGH class sells all.

Signals form at the close and execute at the next session's open. Same-day conflicting HIGH and LOW alerts are ignored. The simulation uses all available cash, integer shares, no leverage or shorting, estimated commission 0.1425% per side (NT$20 minimum), sell tax 0.3%, and 0.1% slippage per side. Corporate-action signal dates are excluded. Remaining positions are liquidated at the final test close.

## Outputs

The workflow creates:
- `tsmc_2330_small_large_wave_fold_metrics.csv`
- `tsmc_2330_small_large_wave_model_summary.csv`
- `tsmc_2330_small_large_wave_predictions.csv`
- `tsmc_2330_small_large_wave_backtest_summary.csv`
- `tsmc_2330_small_large_wave_trades.csv`
- `tsmc_2330_small_large_wave_equity.csv`
- `tsmc_2330_small_large_wave_summary.json`

## Interpretation rules

- Report precision, recall, false-positive rate, AUC, and base rate separately for each pivot side, wave class, horizon, model, and test year.
- Report CAGR, final capital, max drawdown, trade count, win rate, exposure and benchmark for every fixed execution policy.
- Do not select the best policy and report it as an independent result: all historical years have been inspected in earlier research.
- A model that only catches a few large turns is not a complete swing system. Coverage and missed-pivot distance matter alongside precision.
- The final independent validation must be forward paper trading after the feature set, class cutoffs, model, and execution policy are frozen.

## Limitations

The project remains a single-stock exploratory study. Feature selection in previous research has already inspected the full history, so this walk-forward does not erase selection bias. Pivot size is retrospective, 2026 is partial, and the raw-price simulation omits cash dividends, corporate-action cashflows, official price-limit fill rules, liquidity impact, odd-lot microstructure and broker-specific charges. No 200% CAGR or future profitability is claimed.
