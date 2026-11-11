# TSMC 2330 Research Consolidated Report — Run 32

- Research branch: `research/200cagr-blind-validation`
- Tested commit: `b94f56987c585963288611ab073cae27f22579c0`
- GitHub Actions run: [43029433299](https://github.com/ian930719-bot/tw-stock-paper-trader/actions/runs/43029433299)
- Run status: **success**; all 14 substantive workflow steps passed, including compilation/unit tests, event/control audit, feature-cluster audit, small/large-wave walk-forward, daily walk-forward, frozen-rule robustness and artifact upload.
- Artifact: `tsmc-2330-daily-walkforward-32` (artifact ID `13859218154`)
- Data window: 2010-01-04 to 2026-11-09 (4,128 sessions); 2026 is partial.

## Executive conclusion

**No strategy in this run meets the project target of 300% annualized return with maximum drawdown no worse than 10%. No strategy is approved for live automatic trading.**

The strongest frozen LOW/HIGH rule returned 28.46% CAGR over the selected 2016-11-09 research window, but maximum drawdown was 45.64%; buy-and-hold returned 28.79% CAGR with 45.68% drawdown. The frozen rule therefore did not beat the benchmark. This is exploratory evidence, not a clean independent holdout: feature selection and candidate construction used prior inspection of the same historical period.

## Run 32 corrections verified

1. The first test-day decision can use the signal from the immediately preceding completed session; it does not incorrectly drop the boundary signal.
2. Annual returns are calculated from one continuous portfolio equity curve, not by resetting capital to NT$100,000 each January.
3. The annual opening/closing equity chain reconciles continuously. Compounding the 2016–2026 annual returns reproduces the reported full-period cumulative return to rounding; the final annual closing equity is NT$1,512,473.31.
4. The boundary regression test passed in the workflow.

## Frozen LOW/HIGH rule — full-period robustness

Execution model: all available cash, integer shares, next-session open, commission 0.1425% each side (NT$20 minimum), sell tax 0.3%, slippage 0.1% each side, no leverage/shorting.

| Metric | Result |
|---|---:|
| Initial capital | NT$100,000 |
| Final equity | NT$1,512,473 |
| Total return | +1,412.47% |
| CAGR | 28.46% |
| Maximum drawdown | -45.64% |
| Closed trades | 23 |
| Win rate | 86.96% |
| Market exposure | 70.33% |

Benchmark buy-and-hold: final equity NT$1,556,027; CAGR 28.79%; maximum drawdown -45.68%. The rule underperformed buy-and-hold by approximately 0.33 percentage points CAGR.

Subperiods (fresh-start simulations, descriptive and not independent validation):
- 2016–2020: CAGR 30.38%, max drawdown -23.89%.
- 2021–2026-11-09: CAGR 23.04%, max drawdown -45.55%.
- 2023–2026-11-09: CAGR 39.50%, max drawdown -24.09%; explicitly descriptive only.

Execution stress:
- One-session additional delay: CAGR 26.66%.
- Two-session additional delay: CAGR 27.95%.
- Slippage 0.2% each side: CAGR 27.91%.
- Slippage 0.3% each side: CAGR 27.37%.

The annual returns file is now continuous and chained: 2016 opens at NT$100,000 and 2026 closes at NT$1,512,473.31. Annual results include a 2022 decline of 27.04%; annual returns must not be read as independent annual strategy tests.

## Chronological daily walk-forward

The model used expanding chronological training, preceding two years for fixed 95th-percentile alert calibration, and the next year as held-out evaluation; training labels were purged by forecast horizon. It evaluated 317 HIGH and 319 LOW pivot events.

Mean AUC across 11 annual folds:

| Pivot | Horizon | Technical | Technical + foreign/trust |
|---|---:|---:|---:|
| HIGH | 1 session | 0.6767 | 0.6651 |
| LOW | 1 session | 0.6087 | 0.6505 |
| HIGH | 3 sessions | 0.5621 | 0.5530 |
| LOW | 3 sessions | 0.5337 | 0.5669 |

Signal discrimination weakens materially at longer horizons and does not reliably translate into net trading returns.

Cost-aware all-in/all-out simulation (initial NT$100,000):

| Model | Holding horizon | CAGR | Max drawdown | Final equity |
|---|---:|---:|---:|---:|
| Technical | 1 | -3.94% | -53.18% | NT$64,639 |
| Technical | 3 | -4.89% | -57.75% | NT$58,060 |
| Technical | 5 | -9.53% | -71.57% | NT$33,743 |
| Technical | 10 | -2.70% | -44.98% | NT$74,299 |
| Technical | 20 | +1.17% | -26.86% | NT$113,502 |
| Technical + foreign/trust | 1 | -1.70% | -51.11% | NT$83,055 |
| Technical + foreign/trust | 3 | -3.11% | -43.88% | NT$70,964 |
| Technical + foreign/trust | 5 | -6.01% | -61.10% | NT$51,050 |
| Technical + foreign/trust | 10 | +1.94% | -32.54% | NT$123,131 |
| Technical + foreign/trust | 20 | -0.23% | -29.51% | NT$97,510 |
| Buy-and-hold | Full period | 28.79% | -45.68% | NT$1,556,027 |

Adding foreign-investor and investment-trust features did not create a stable profitable edge. Dealers remain excluded. Institutional coverage begins 2012-05-02.

## Small/large wave study

The best no-stop technical small-wave 3-session configuration returned 21.85% CAGR with -45.63% maximum drawdown. Technical all-wave 10-session returned 17.58% CAGR with -20.41% drawdown. Buy-and-hold remained stronger on CAGR (28.79%).

One-session model AUCs:
- Large LOW: 0.7200 (technical); technical + foreign/trust 0.7132.
- Large HIGH: 0.7031 (technical); technical + foreign/trust 0.5787.
- Small HIGH: 0.6777 (technical).
- Small LOW: 0.6258 (technical).

Pivot amplitude class is retrospective: a pivot's eventual reversal size is known only after the reversal. It is an evaluation label, not a tradable real-time signal. Precision is low for rare large-pivot events, so AUC alone is not a sufficient trading criterion.

## ATR stop and fresh-LOW re-entry grid

The grid tested 1.5/2.0/2.5/3.0 ATR trailing stops and 1/3/5/10-session cooldowns; a stop invalidates the old LOW alert and requires a new LOW alert after cooldown.

Best CAGR among stop-enabled configurations:
- Technical all-waves, 1-session horizon, 2.0 ATR stop, 1-session cooldown.
- CAGR 9.90%; max drawdown -28.21%; 82 trades; win rate 53.66%; exposure 28.05%; 47 stops and 46 re-entries.

No stop-enabled combination achieved max drawdown of 10% or less. The strongest CAGR among combinations with drawdown no worse than 12% was a small-wave 1-session model with 2.0 ATR stop and 1-session cooldown: CAGR 6.35%, max drawdown -11.39%, 44 trades, 22 stops and 21 re-entries. This demonstrates the return/risk trade-off; it does not approach the 300% CAGR target.

## What the evidence says about turning-point features

The event/control audit found statistically distinguishable feature patterns around pivots after multiple-testing correction. LOW precursors tended to show weaker short-term returns, lower stochastic readings, weaker OBV flow and lower close-location values. HIGH precursors tended to show stronger one-day returns, higher close-location values, larger intraday ranges and greater extension above long-term averages.

These are conditional associations, not sufficient entry/exit rules. Prior simple rule audits did not establish a robust profitable rule. Significant feature differences must not be represented as a proven trading edge.

## Market mechanics and unresolved execution fidelity

The current simulator models next-session open fills, fees, tax and estimated slippage, but does **not** yet fully model:
- Official daily price-limit rounding and locked limit-up/limit-down no-fill conditions.
- New-listing exemptions and special trading-status exceptions.
- Cash dividends and all corporate-action cashflows.
- Market impact, liquidity/participation limits, odd-lot microstructure and broker-specific fee discounts.
- A complete reconciliation of every simulated order to official exchange executions.

TWSE rules generally limit stocks to ±10% from that session's opening auction reference price; newly listed common stocks have no price limit for the first five trading sessions (subject to the rule's stated exceptions). See the [TWSE Operating Rules, Article 63](https://twse-regulation.twse.com.tw/TW/law/DOC01.aspx?FLCODE=fl007304&FLNO=63) and [TWSE trading system explanation](https://www.twse.com.tw/zh/products/system/trading.html?hl=zh-TW). The simulator must implement the applicable TWSE/TPEx rule and actual order-fill constraints before any result is considered execution-ready.

## Research decision and next gates

1. **Do not promote the current candidate to live trading.** Keep it frozen as a research benchmark; do not optimize its thresholds on the same 2010–2026 history.
2. Implement exchange-rule-aware fills, price-limit queue/no-fill behavior, corporate actions/dividends, liquidity constraints and explicit data-quality checks.
3. Expand from TSMC to a point-in-time, delisted-inclusive universe. Avoid survivorship bias and ensure each feature is available at the decision timestamp.
4. Run pre-registered cross-stock walk-forward tests with training, calibration and untouched forward test windows; reserve a truly unseen future period for final confirmation.
5. Compare against buy-and-hold, cash and simple baseline strategies using net returns, CAGR, max drawdown, exposure, turnover, trade count, bootstrap confidence intervals and multiple-testing correction.
6. Only consider paper trading after the above gates pass; require a long enough forward paper-trading period and reconciliation before enabling real orders.

**Final status:** Run 32 completed successfully and its corrected annual-equity accounting is verified. The research pipeline is reproducible for this single-stock dataset, but the investment objective is not met and the strategy is not ready for automatic or live execution.
