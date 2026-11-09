# 200% CAGR Reverse-Research and Blind Validation Protocol

Status: research specification; no strategy is approved for live capital.
Target: investigate whether a robust, executable Taiwan-equity strategy can approach 200% net CAGR. This is a research hurdle, not an expected or guaranteed result.

## Non-negotiable rules

1. Never tune on the blind period. Record the cutoff date, code commit, data digest, universe rules, parameters, execution model, and hypotheses before opening blind results.
2. Separate discovery, inner validation, outer walk-forward, and final blind forward paper trading. The final blind period is evaluated once per pre-registered version. Any change creates a new version and a new future blind period.
3. Use only information available at each decision timestamp. Signals computed from a completed daily bar may fill no earlier than the next tradable session. No same-bar close/high/low look-ahead.
4. Use point-in-time constituents, delisted securities, ticker changes, suspensions, corporate actions, and historical listing dates wherever verifiable. If historical universe coverage is incomplete, label survivorship bias as unresolved and prohibit a research-grade pass.
5. Keep latest fundamental snapshots out of historical signals. Historical fundamentals require publication/availability timestamps and point-in-time vintages.
6. Include commissions, transaction tax by instrument and date, bid/ask spread or conservative slippage, market impact, board-lot constraints, price limits, suspensions, liquidity and partial-fill constraints. Stress costs at 1.5x and 2.0x.
7. Never use future data to fill missing bars or corporate actions. Missing or suspicious records are excluded or flagged with a reproducible reason.
8. Do not optimize solely for CAGR. Report drawdown, volatility, Sharpe/Sortino, Calmar, profit factor, expectancy, turnover, exposure, capacity, trade concentration, tail losses, and year/regime/sector dispersion.

## Research design

### A. Data and universe

- Build a timestamped daily OHLCV panel for listed and OTC Taiwan equities, including delisted names when obtainable.
- Preserve raw source payloads, retrieval timestamps, adjustment/action tables, and per-symbol quality reports.
- Build each date's eligible universe using listing status and information actually known on that date; do not use today's constituents retrospectively.
- Keep separate coverage flags for delisted names, suspensions, limit-up/down sessions, missing prices, and corporate actions.
- Exclude securities with insufficient warm-up history, invalid prices, stale bars, or inadequate liquidity under rules declared before testing.
- Document every source gap. A contemporary Yahoo-derived constituent snapshot is not a point-in-time universe.

### B. Factor discovery (training data only)

Explore interpretable features in predeclared families:
- Trend and price structure: multi-horizon returns, moving-average slope/order, breakout distance, distance to highs/lows.
- Momentum and relative strength: 1/3/6/12-month momentum, cross-sectional ranks, industry-relative strength.
- Volume and liquidity: relative volume, turnover, dollar volume, volume shock, participation capacity.
- Volatility and compression: ATR, realized volatility, range compression, volatility expansion.
- Market/industry regime: broad-market trend, breadth, dispersion, industry breadth and relative momentum.
- Risk and execution: gap risk, limit proximity, liquidity, volatility-scaled stops and portfolio heat.

Use interpretable feature combinations and bounded symbolic regression. Penalize formula complexity and unstable coefficients. Keep a complete registry of every tested hypothesis and parameter configuration to measure multiple-testing risk. No factor may be selected from final blind results.

### C. Nested validation

- Discovery/training: earliest chronological segment only.
- Inner validation: select feature families and parameters using purged/embargoed time splits.
- Outer validation: rolling or expanding walk-forward with non-overlapping OOS windows and continuous portfolio capital.
- Group results by year, bull/bear/sideways regime, industry, market-cap/liquidity bucket, and individual symbol.
- Use block/bootstrap uncertainty estimates and multiple-testing-aware statistics (e.g. Deflated Sharpe and Probability of Backtest Overfitting where sample design permits).
- Compare against matched-cost 0050 buy-and-hold and simple trend/momentum baselines.
- Reject a candidate that relies on one symbol, one sector, one year, a handful of outsized trades, or one narrow parameter setting.

### D. Portfolio and execution model

- Rank candidates using only prior-close data; execute next session with conservative price assumptions.
- Position size is the minimum of risk-budget sizing, cash/slot allocation, board-lot constraints, and ADV participation capacity.
- Enforce portfolio-level risk limits: aggregate stop risk, sector concentration, single-name concentration, gross exposure, cash reserve, and drawdown circuit breakers.
- Model gaps through stops at the first executable price; never assume a stop fills at its trigger when the market gaps through it.
- Include open positions and receivables correctly in marked-to-market equity; report realized and unrealized P&L separately.
- Run baseline, 1.5x-cost, and 2.0x-cost cases without changing signals or parameters.

### E. Frozen blind forward test

Before the blind period begins, write an immutable manifest containing:
- git commit SHA and strategy version;
- training/validation cutoff and blind start date;
- exact symbol eligibility and delisting policy;
- factor formula, parameter values, and portfolio rules;
- fees, taxes, slippage, participation and fill assumptions;
- data source/version and input hashes;
- pass/fail criteria.

During the blind period, do not retune, change symbols, or revise fills retrospectively. Log every signal, intended order, simulated fill, rejected order, position, cash movement, corporate action, and daily equity. Evaluate the frozen version at predeclared checkpoints only.

## Proposed acceptance gates

These are initial research gates and may be tightened before blind testing; they must not be relaxed after seeing blind results.

- Data integrity: no known look-ahead; corporate-action reconciliation; point-in-time universe coverage documented. Unresolved survivorship bias blocks research-grade approval.
- Breadth: positive net expectancy across multiple independent years and industries; results not dominated by a single symbol or a small set of trades.
- Robustness: positive OOS net expectancy after baseline costs and no catastrophic collapse at 1.5x costs; 2.0x costs reported even if it fails.
- Stability: performance survives nearby parameter perturbations and alternative reasonable execution assumptions.
- Risk: predeclared maximum drawdown, exposure, concentration, and liquidity constraints respected.
- Blind evidence: at least 12 months of frozen forward paper trading before considering any live deployment; longer is preferable for low-frequency strategies.
- Target reporting: show whether net CAGR reaches 200%, but do not waive any risk, data, or generalization gate to achieve that number.

## Required outputs

1. data_coverage.csv and data_quality.csv
2. point_in_time_universe_audit.csv
3. hypothesis_registry.csv and factor_stability.csv
4. nested_walk_forward_windows.csv
5. symbol_sector_regime_breakdown.csv
6. trade_ledger.csv and daily_equity.csv
7. cost_stress.csv and concentration_diagnostics.csv
8. frozen_blind_manifest.json and forward_paper_log.csv
9. final_report.md with limitations and reproducible commands

## Decision labels

- REJECT: invalid data, look-ahead, non-positive net OOS expectancy, or unacceptable risk.
- RESEARCH_ONLY: promising historical results but incomplete universe, weak robustness, insufficient independent OOS evidence, or insufficient blind duration.
- BLIND_PAPER: all historical gates pass and the version is frozen for forward paper testing.
- LIVE_CANDIDATE: only after the blind period and all pre-registered gates pass; this label is not a recommendation to deploy capital.

No result may be described as a universal formula or 200% strategy until independently reproduced across symbols, sectors, regimes, and a frozen future period.
