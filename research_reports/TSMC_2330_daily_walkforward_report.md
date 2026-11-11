# TSMC 2330 daily walk-forward study

## Reproducibility

- Branch: `research/200cagr-blind-validation`
- Frozen source artifact: workflow run `42982871652`, artifact `tsmc-2330-cpcv-foreign-trust-ablation-42`
- Input period: 2010-01-04 to 2026-11-09
- Daily feature rows: 4,128
- Frozen pivot threshold: `p0.020_atr1.0`
- Retrospective pivot events: 317 HIGH, 319 LOW
- Institutional features start: 2012-05-02
- Dealer features: excluded; only foreign-investor and investment-trust flow features are used.

## Validation design

For each direction and 1/3/5/10/20-session horizon, the model is trained on earlier years, the preceding two calendar years are used only to calibrate a fixed 95th-percentile alert threshold, and the next calendar year is scored. Training rows whose forward labels could cross into the calibration period are purged. Signals are generated from completed daily features and the simulated order is not executed before the next session open.

The daily target is whether a frozen retrospective pivot occurs within the next H sessions. Classification metrics are calculated only where the full forward horizon is observable. The 2026 test year is partial.

## Local reproducible results

The local run on the frozen source artifact produced 220 annual folds (11 test years × 2 directions × 5 horizons × 2 feature sets).

Mean AUC examples:
- HIGH, 1 session: technical 0.677; technical + foreign/trust 0.665.
- LOW, 1 session: technical 0.609; technical + foreign/trust 0.650.
- HIGH, 10 sessions: technical 0.535; technical + foreign/trust 0.516.
- LOW, 10 sessions: technical 0.493; technical + foreign/trust 0.475.

At the 1-session horizon, HIGH technical alerts had mean precision 17.1%, mean recall 14.0%, and mean control-day false-positive rate 5.0%, against a 7.0% event base rate. LOW technical alerts had mean precision 10.1%, recall 10.1%, and false-positive rate 7.5%, against a 7.1% base rate. These results show a weak short-horizon high-pivot signal, but no comparably reliable low-pivot signal.

## Cost-aware price-only simulation

The long-only simulation was run over 2016-01-04 to 2026-11-09 with TWD 100,000 initial capital. It buys after a LOW alert at the close (next open execution), exits after a HIGH alert (next open), or uses a 2 ATR stop and 20-session maximum holding period. It tests all five alert horizons.

Estimated costs: commission 0.1425% each side, sell tax 0.3%, slippage 0.1% each side. Positions use integer shares, all available cash, no leverage and no shorting. Execution uses raw OHLC prices; ATR is rescaled to the raw price level. The price-only accounting excludes cash dividends and corporate-action cashflows, broker minimum fees, liquidity impact, and odd-lot auction microstructure.

| Model | Horizon | Final net equity | Total return | Price-only CAGR | Max drawdown | Closed trades |
|---|---:|---:|---:|---:|---:|---:|
| Technical | 1 | 64,639 | -35.4% | -3.94% | -53.2% | 99 |
| Technical | 3 | 58,060 | -41.9% | -4.89% | -57.8% | 123 |
| Technical | 5 | 33,743 | -66.3% | -9.53% | -71.6% | 93 |
| Technical | 10 | 74,299 | -25.7% | -2.70% | -45.0% | 49 |
| Technical | 20 | 113,502 | +13.5% | +1.17% | -26.9% | 42 |
| Technical + foreign/trust | 1 | 83,055 | -17.0% | -1.70% | -51.1% | 74 |
| Technical + foreign/trust | 3 | 70,964 | -29.0% | -3.11% | -43.9% | 75 |
| Technical + foreign/trust | 5 | 51,050 | -48.9% | -6.01% | -61.1% | 77 |
| Technical + foreign/trust | 10 | 123,131 | +23.1% | +1.94% | -32.5% | 40 |
| Technical + foreign/trust | 20 | 97,510 | -2.5% | -0.23% | -29.5% | 36 |
| Buy and hold | — | 1,556,027 | +1,456.0% | +28.79% | -45.7% | 1 |

All amounts are TWD. Buy-and-hold uses the same starting capital and estimated entry/exit costs; all rows exclude cash dividends. The strategy variants materially underperformed buy-and-hold. The highest-return tested variant is not promoted as a selected strategy because choosing it from the same test period would introduce selection bias.

## Interpretation and next gate

1. The short-horizon HIGH alert has some discrimination, but recall is low; most actual pivots are not flagged.
2. LOW alerts are weak. The existing feature combinations do not reliably identify buyable lows.
3. Adding foreign/trust flow does not consistently improve classification or net returns.
4. A strategy that buys every predicted low and sells every predicted high is not competitive with holding TSMC over this period.
5. The research should pivot to a trend-following core with swing alerts as risk/position-sizing overlays, then be tested on a newly frozen period. Do not tune against these same OOS years and report them as independent validation.

## Research limitations

This is a single-stock exploratory study. Candidate features were preselected after inspecting the broader pivot study, so the chronological folds are not a fully untouched final blind test. Retrospective pivots are outcome labels, not directly observable live turning points. Historical security-universe survivorship is not relevant to this single-symbol study, but vendor price reconciliation and official exchange execution reconciliation are not established. The results do not establish research-grade readiness, future profitability, or the 200% annualized-return goal.
