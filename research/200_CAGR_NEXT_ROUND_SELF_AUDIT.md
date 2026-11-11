# 200% Net CAGR Research Round — Strategy Logic Self-Audit

Date: 2026-11-11 (Asia/Taipei)
Repository: `ian930719-bot/tw-stock-paper-trader`
Branch: `research/200cagr-blind-validation`

## Hard acceptance rule

The only acceptable research result is **net CAGR >= 200%** over the declared full evaluation window, after realistic fees, tax, slippage and executable-fill constraints. Any result below 200% is a failed candidate and must not be promoted, softened, relabelled, or averaged with a stronger subperiod. This is a research acceptance threshold, not a promise that such a strategy exists.

No strategy tested in this exploratory round met the threshold.

## Data used and evidence limitations

Used the frozen Taiwan-history artifact from workflow run 42941077075:
- Requested candidate symbols: 2,220.
- Price histories with rows: 2,189.
- Valid OHLCV rows in audit: 6,905,263.
- Delisted candidates: 58; 57 had some price bars.
- Data range in manifest: 2010-01-04 through 2026-11-10.

The same audit explicitly says point-in-time universe completeness, delisted-price coverage, official TWSE/TPEx price reconciliation, official calendar reconciliation, and complete corporate actions are **not established**. Capital-reduction reference prices were blocked without a token. This dataset is useful for exploratory screening but cannot support a research-grade pass or a claim of survivorship-bias-free validation.

For the exploratory screen, each calendar year's universe was selected using the previous calendar year's turnover ranking (up to 20 names). This reduced the unique selected-symbol set to 89 across 2016–2026. The 2026 period is partial.

## Exploratory first-pass strategy screen

A deliberately small set of interpretable long-only, no-leverage, one-position-at-a-time signals was screened:
- 20-session and 55-session breakout with a long-term trend filter and relative-volume confirmation.
- Trend pullback / EMA reclaim.
- Cross-sectional momentum rotation using 63/126/252-session ranks.
- Exits based on EMA21, SMA50 or negative 20-session momentum.
- 2.5 ATR initial/trailing stop; 40-session maximum holding period.
- Next-session open entry/exit, estimated 0.1% slippage per side, 0.1425% buy commission, and 0.4425% sell commission plus tax.

The exploratory screen produced the following full-window diagnostics:

| Signal | Exit | Final equity from NT$100,000 | CAGR | Max drawdown | Trades |
|---|---|---:|---:|---:|---:|
| Momentum rotation | 20-session momentum < 0 | NT$1,307 | -32.95% | -73.72% | 50 |
| Trend pullback | 20-session momentum < 0 | NT$1,122 | -33.89% | -16.21% | 18 |
| Momentum rotation | SMA50 | NT$959 | -34.83% | -61.86% | 31 |
| Momentum rotation | EMA21 | NT$695 | -36.75% | -80.73% | 52 |
| Trend pullback | EMA21 | NT$218 | -43.16% | -20.47% | 17 |
| Trend pullback | SMA50 | NT$146 | -45.21% | -33.74% | 15 |
| 55-session breakout | 20-session momentum < 0 | NT$44 | -50.94% | -62.50% | 110 |
| 55-session breakout | SMA50 | NT$34 | -52.16% | -63.08% | 109 |
| 55-session breakout | EMA21 | NT$28 | -52.97% | -62.87% | 113 |
| 20-session breakout | EMA21 | NT$12 | -56.47% | -96.16% | 139 |
| 20-session breakout | SMA50 | NT$8 | -57.98% | -89.02% | 133 |
| 20-session breakout | 20-session momentum < 0 | NT$7 | -58.73% | -94.34% | 135 |

These are **diagnostic exploratory outputs, not a validated performance claim**. They are not an independent holdout, and execution/corporate-action fidelity is incomplete. No configuration comes remotely close to the 200% acceptance hurdle. The segment rerun did not complete; therefore no discovery/validation/forward split result is claimed.

## Self-audit: what was wrong with my own logic

I am not attributing this failure to market conditions or data availability. The logic and implementation need to be corrected.

1. **I used weak entry timing.** Breakout confirmation after a 20/55-session high often enters after much of the move has already happened. The strategy then suffers from false breakouts and gap risk. A high/low turning-point strategy must distinguish a genuine accumulation/reversal from a late momentum chase.
2. **I used exits that give back too much.** A negative 20-session return or a slow moving-average break reacts late. In a sharp reversal, a large portion of open profit can disappear before the exit. A trailing stop alone does not solve this because ATR expands during volatility shocks.
3. **I did not model regime selection.** A trend strategy should not be active with the same rules in a broad-market downtrend, low-breadth regime, or high-volatility selloff. Conversely, a mean-reversion entry should not be used indiscriminately in a strong falling market.
4. **I over-concentrated the portfolio.** Putting all capital into the single highest-ranked name exposes the account to company-specific gaps, limit-down no-fill risk and sudden liquidity loss. The user's all-in/all-out requirement is retained at the portfolio level, but candidate ranking and a market-regime gate must be stronger before committing the whole account.
5. **I did not enforce fresh-signal re-entry.** In the exploratory simulator, an exit at today's open could be followed by a same-day re-entry using yesterday's still-active signal. This can create churn and violates the intended rule that a stop/exit invalidates the old signal and a new completed-session signal is required after a cooldown. The simulator must track `closed_today`, invalidate the old signal, and enforce a fresh-signal/cooldown state machine.
6. **I used adjusted prices as execution prices.** The research transformation back-adjusted historical OHLC for corporate actions, then used those adjusted prices as if they were executable fills. Actual fills must use raw tradable prices; share changes, cash dividends, capital reductions and splits must be booked as explicit portfolio cash/share events. Incomplete corporate-action data makes the current fill model unsuitable for acceptance testing.
7. **The terminal liquidation was not reconciled into the final equity-curve observation.** The reported terminal cash after liquidation and the last marked equity observation can diverge. The corrected engine must record liquidation in the equity curve and reconcile every daily equity value to cash plus marked holdings.
8. **I did not finish independent segment validation.** A full-window ranking is not proof of generalization. Any next candidate must be selected on discovery data, frozen, then assessed on later non-overlapping validation and forward windows. Previously inspected periods cannot be called blind.

## Next algorithm design — change the hypothesis, not merely tune thresholds

The next round must test a different causal hypothesis instead of repeatedly changing moving-average or ATR thresholds.

### Entry point model
1. **Market regime gate:** only allow full-risk long exposure when broad-market trend, market breadth and volatility regime are supportive. Define these using information available at the prior close.
2. **Compression-to-expansion:** identify volatility contraction, declining true range and volume dry-up, then require a confirmed expansion/breakout with turnover expansion. Enter at the next session's executable open; do not buy a signal bar's close.
3. **Pullback re-entry:** after a valid expansion, permit a controlled pullback toward the breakout level or EMA21 only if the prior breakout level holds and relative strength remains high. Require a fresh confirmation close before re-entry.
4. **Relative strength leadership:** rank candidates by multi-horizon relative strength against both the broad market and industry, not only against the current top-20 turnover list.
5. **False-breakout rejection:** reject signals with excessive opening gaps, weak close location, abnormal spread/illiquidity, price-limit proximity, or failed follow-through.
6. **Wave structure:** identify a first impulse, controlled retracement and renewed expansion. Do not use the eventual pivot size as an entry feature; that is future information.

### Exit and re-entry state machine
- Initial stop is volatility-scaled and fixed from the entry-time ATR; it must not be widened after entry.
- Profit protection should combine a structure-based invalidation level and a volatility-adjusted trailing exit; test partial-profit logic only in a separate experimental family, while the primary portfolio remains all-in/all-out.
- A stop or thesis invalidation changes the position state to `COOLDOWN`; the old signal is discarded.
- Re-entry is allowed only after cooldown and a newly formed, newly confirmed signal timestamped after the exit. Never re-enter from a stale prior-day alert.
- Include gap-through-stop and limit-down no-fill logic; if a legal executable fill is unavailable, carry the position and mark the risk conservatively.

### Required evaluation gates
- Freeze source SHA, dataset hashes, universe definition, parameters and execution assumptions before validation.
- Use chronological discovery, validation and untouched prospective forward evaluation. Do not retune on the forward result.
- Report net CAGR, max drawdown, annual returns, trade count, win rate, profit factor, exposure, turnover, worst gap, capacity and benchmark comparisons.
- Stress commissions/slippage at 1.5x and 2.0x.
- Reconcile every fill and daily equity value; no unexplained cash or share changes.
- The sole candidate pass condition is net CAGR >= 200% for the entire declared evaluation window **and** positive results in the locked later validation/forward windows. A 199.99% result is a fail. A high in-sample result with a failed later window is also a fail.
- Do not claim a pass until point-in-time universe, delisted coverage, corporate actions and TWSE/TPEx execution rules are sufficiently reconciled.

## Decision

**Round failed. No strategy achieved the 200% hurdle. No candidate is approved for paper/live automatic execution.**

The next research iteration should first repair the simulator state machine, raw-fill/corporate-action accounting and equity reconciliation, then test the new regime + compression/expansion + relative-strength hypothesis. It must not simply optimize the failed thresholds on the same historical sample.
