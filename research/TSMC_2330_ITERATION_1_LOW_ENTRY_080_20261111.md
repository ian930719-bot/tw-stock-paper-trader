# TSMC 2330 strategy iteration 1 — low-entry threshold sensitivity

Date: 2026-11-11  
Branch: research/200cagr-blind-validation  
Baseline: Run #32 frozen all-in/all-out candidate, 2016-01-04 through 2026-11-09.

## Objective
Improve the existing 28.4557% CAGR while preserving or improving the 86.9565% trade win rate. No claim of guaranteed future performance is made.

## Existing baseline
- CAGR: 28.4557%
- Maximum drawdown: -45.6375%
- Closed trades: 23
- Win rate: 86.9565%
- Final equity from NT$100,000: NT$1,512,473

## First candidate found in the existing one-factor sensitivity audit
Change only the LOW-entry daily return threshold:
- Previous: return_1 <= -1.00%
- Candidate: return_1 <= -0.80%
- All other LOW and HIGH conditions remain frozen.

Recorded Run #32 neighbour-sensitivity result:
- CAGR: 30.4402%
- Total return: +1,686.1%
- Final equity from NT$100,000: approximately NT$1,786,115
- Maximum drawdown: -45.6761%
- Closed trades: 25
- Win rate: 88.00%

This clears the requested 30% historical CAGR threshold and raises the measured win rate by 1.04 percentage points, with two additional trades. Drawdown is effectively unchanged and slightly worse by 0.04 percentage points.

## Interpretation
The candidate allows a slightly earlier/less severe capitulation entry. The hypothesis is that requiring a full -1.0% down day misses some valid reversal setups; -0.8% captures more of them without materially diluting the other capitulation filters (relative volume, stochastic oversold, close near session low, negative OBV flow).

## Critical validation status
This is a one-factor neighbour result from the same historical sample used to develop the baseline. It is a promising in-sample candidate, NOT independent out-of-sample validation. It must not be called production-ready or guaranteed. The next required test is to freeze -0.8% and re-run the exact full chronological simulation with:
1. 2016–2020 and 2021–2026 subperiod metrics;
2. full trade ledger and exact signal/fill dates;
3. same transaction cost, slippage, integer-share, next-session-open assumptions;
4. delayed execution and higher-slippage stress;
5. untouched forward validation after the rule is frozen.

Do not tune more thresholds against the same test segment. Reject the candidate if the chronological holdout fails to retain the target return or win-rate floor.

## Decision
Promote LOW return threshold -0.8% to a frozen validation candidate only. Do not replace the baseline in the live/paper-trading configuration until the independent chronological validation passes.
