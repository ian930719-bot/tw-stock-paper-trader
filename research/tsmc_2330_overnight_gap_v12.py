"""V12: point-in-time walk-forward prediction of TSMC overnight gaps.
Feature date i is known at its close. A signal schedules a buy at the next
session close (i+1) and sells at the following open (i+2). All labels mature
before the annual test window; costs and integer-share sizing are included.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CAPITAL = 100_000.0
BUY_FEE, SELL_FEE_TAX, SLIPPAGE = 0.001425, 0.004425, 0.001
COST_HURDLE = (1 + SLIPPAGE) * (1 + BUY_FEE) / ((1 - SLIPPAGE) * (1 - SELL_FEE_TAX)) - 1
FEATURES = [
    "rsi", "macdh_pct", "atrp", "ret5", "ret10", "ret20", "ret40", "ret60", "ret120",
    "ma5dev", "ma10dev", "ma20dev", "ma40dev", "ma60dev", "ma120dev", "ma200dev",
    "vol5", "vol20", "gap", "intraday", "bbpct", "bbwidth",
]
PERIODS = {
    "research": ("2016-01-01", "2019-12-31"),
    "validation": ("2020-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2026-11-10"),
}


def prepare(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    c, h, l, o, v = d.close, d.high, d.low, d.open, d.volume
    change = c.diff()
    up = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / down.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macdh_pct"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atrp"] = tr.rolling(14).mean() / c
    for n in (5, 10, 20, 40, 60, 120, 200):
        d[f"ret{n}"] = c.pct_change(n)
        d[f"ma{n}dev"] = c / c.rolling(n).mean() - 1
    d["gap"] = o / c.shift(1) - 1
    d["intraday"] = c / o - 1
    d["vol5"] = c.pct_change().rolling(5).std()
    d["vol20"] = c.pct_change().rolling(20).std()
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["bbpct"] = (c - mid) / (2 * sd)
    d["bbwidth"] = 4 * sd / mid
    # At feature date i, predict the overnight gap after next session's close:
    # enter at close[i+1], exit at open[i+2].
    d["target_ret"] = o.shift(-2) / c.shift(-1) - 1
    d["target_positive_net"] = (d.target_ret > COST_HURDLE).astype(float)
    d.loc[d.index >= len(d) - 2, ["target_ret", "target_positive_net"]] = np.nan
    return d


def model_factories():
    return {
        "logistic_probability": lambda: Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=2000, random_state=2330)),
        ]),
        "ridge_return": lambda: Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()), ("model", Ridge(alpha=10.0)),
        ]),
        "random_forest_return": lambda: Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=25,
                max_features=0.75, n_jobs=-1, random_state=2330)),
        ]),
        "extra_trees_return": lambda: Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=25,
                max_features=0.75, n_jobs=-1, random_state=2330)),
        ]),
        "hist_gradient_return": lambda: Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=7,
                min_samples_leaf=30, l2_regularization=5, learning_rate=0.04, random_state=2330)),
        ]),
    }


def walk_forward_predictions(d: pd.DataFrame, model_name: str, factory):
    predictions = np.full(len(d), np.nan)
    yearly = []
    entry_year = d.date.shift(-1).dt.year
    for year in sorted(d.date.dt.year.unique()):
        if year < 2016:
            continue
        test_mask = (entry_year == year).fillna(False).to_numpy()
        test_idx = np.where(test_mask & (d.index.to_numpy() <= len(d) - 3))[0]
        if not len(test_idx):
            continue
        first_test_signal = int(test_idx[0])
        train_mask = (d.index.to_numpy() + 2 < first_test_signal) & d.target_ret.notna().to_numpy()
        train = d.loc[train_mask].dropna(subset=["target_ret", "target_positive_net"])
        if len(train) < 500:
            yearly.append({"model": model_name, "year": int(year), "status": "insufficient", "train_rows": len(train)})
            continue
        model = factory()
        target = "target_positive_net" if model_name == "logistic_probability" else "target_ret"
        model.fit(train[FEATURES], train[target].astype(int) if target.endswith("net") else train[target])
        predictions[test_idx] = model.predict_proba(d.loc[test_idx, FEATURES])[:, 1] if model_name == "logistic_probability" else model.predict(d.loc[test_idx, FEATURES])
        yearly.append({"model": model_name, "year": int(year), "status": "fit", "train_rows": len(train),
                       "train_positive_rate": float(train.target_positive_net.mean())})
    return predictions, yearly


def period_masks(d: pd.DataFrame):
    entry_date, exit_date = d.date.shift(-1), d.date.shift(-2)
    masks = {}
    for name, (lo, hi) in PERIODS.items():
        masks[name] = ((entry_date >= lo) & (entry_date <= hi) &
                       (exit_date >= lo) & (exit_date <= hi)).fillna(False).to_numpy()
    return masks


def simulate_gap_trades(d: pd.DataFrame, predictions: np.ndarray, threshold: float,
                        mask: np.ndarray, keep_trades: bool = False):
    eligible = np.where(mask & np.isfinite(predictions))[0]
    cash = CAPITAL
    trade_rows, equity_after_trades = [], [CAPITAL]
    net_returns = []
    for i in eligible:
        if predictions[i] < threshold:
            continue
        buy_px = float(d.close.iloc[i + 1]) * (1 + SLIPPAGE)
        sell_px = float(d.open.iloc[i + 2]) * (1 - SLIPPAGE)
        qty = int(cash / (buy_px * (1 + BUY_FEE)))
        if qty <= 0:
            continue
        cost = qty * buy_px * (1 + BUY_FEE)
        proceeds = qty * sell_px * (1 - SELL_FEE_TAX)
        net_return = proceeds / cost - 1
        cash += proceeds - cost
        net_returns.append(net_return)
        equity_after_trades.append(cash)
        if keep_trades:
            trade_rows.append({
                "signal_date": str(d.date.iloc[i].date()),
                "entry_date": str(d.date.iloc[i + 1].date()),
                "entry_close": float(d.close.iloc[i + 1]),
                "exit_date": str(d.date.iloc[i + 2].date()),
                "exit_open": float(d.open.iloc[i + 2]),
                "shares": qty, "net_return_pct": net_return * 100,
                "probability_or_predicted_return": float(predictions[i]),
            })
    if len(eligible):
        years = (d.date.iloc[eligible[-1] + 2] - d.date.iloc[eligible[0] + 1]).days / 365.25
    else:
        years = 0
    cagr = ((cash / CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0.0
    eq = np.asarray(equity_after_trades, dtype=float)
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1) * 100)
    return {
        "final_equity": float(cash), "cagr_pct": float(cagr),
        "max_drawdown_trade_end_pct": mdd, "trades": len(net_returns),
        "win_rate_pct": float(np.mean(np.asarray(net_returns) > 0) * 100) if net_returns else None,
        "average_net_trade_pct": float(np.mean(net_returns) * 100) if net_returns else None,
        "coverage_pct": float(np.mean(predictions[eligible] >= threshold) * 100) if len(eligible) else 0.0,
        "eligible_signal_days": int(len(eligible)),
    }, trade_rows


def benchmark_close_to_open(d: pd.DataFrame, mask: np.ndarray):
    idx = np.where(mask)[0]
    if not len(idx):
        return {}
    first_entry, last_exit = idx[0] + 1, idx[-1] + 2
    buy_px = float(d.close.iloc[first_entry]) * (1 + SLIPPAGE)
    qty = int(CAPITAL / (buy_px * (1 + BUY_FEE)))
    cash = CAPITAL - qty * buy_px * (1 + BUY_FEE)
    final = cash + qty * float(d.open.iloc[last_exit]) * (1 - SLIPPAGE) * (1 - SELL_FEE_TAX)
    years = (d.date.iloc[last_exit] - d.date.iloc[first_entry]).days / 365.25
    return {"final_equity": float(final), "cagr_pct": float(((final / CAPITAL) ** (1 / years) - 1) * 100)}


def run(input_csv: str, output_dir: str) -> dict:
    d = prepare(input_csv)
    masks = period_masks(d)
    factories = model_factories()
    model_predictions, training_rows, result_rows, calibration_rows = {}, [], [], []
    for model_name, factory in factories.items():
        pred, yearly = walk_forward_predictions(d, model_name, factory)
        model_predictions[model_name] = pred
        training_rows.extend(yearly)
        for period, mask in masks.items():
            valid = mask & np.isfinite(pred) & d.target_positive_net.notna().to_numpy()
            if model_name == "logistic_probability" and valid.sum():
                y = d.loc[valid, "target_positive_net"].astype(int)
                calibration_rows.append({"model": model_name, "period": period,
                    "n": int(valid.sum()), "auc": float(roc_auc_score(y, pred[valid])),
                    "brier": float(brier_score_loss(y, pred[valid])),
                    "base_rate": float(y.mean()), "mean_prediction": float(np.mean(pred[valid]))})
        thresholds = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60) if model_name == "logistic_probability" else (0.002, 0.004, 0.006, 0.008, 0.010, 0.015)
        for threshold in thresholds:
            row = {"model": model_name, "threshold": threshold}
            for period, mask in masks.items():
                metrics, _ = simulate_gap_trades(d, pred, threshold, mask)
                for key, value in metrics.items():
                    row[f"{period}_{key}"] = value
            result_rows.append(row)
    results = pd.DataFrame(result_rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "overnight_gap_threshold_results.csv", index=False)
    pd.DataFrame(training_rows).to_csv(out / "yearly_training.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(out / "logistic_calibration.csv", index=False)
    selected_pred = model_predictions["logistic_probability"]
    selected_metrics, trades = simulate_gap_trades(d, selected_pred, 0.40, masks["test"], True)
    pd.DataFrame(trades).to_csv(out / "logistic_040_test_trades.csv", index=False)
    summary = {
        "version": "V12", "symbol": "2330", "history_rows": len(d),
        "history_start": str(d.date.min().date()), "data_end": str(d.date.max().date()),
        "source": str(d.provider.iloc[0]) if "provider" in d else "adjusted OHLCV",
        "cost_hurdle_before_rounding": COST_HURDLE,
        "execution": "Feature date close i; signal schedules buy at close i+1 and sell at open i+2; both sides include fees/tax and 0.10% slippage.",
        "periods": {k: {"start": lo, "end": hi} for k, (lo, hi) in PERIODS.items()},
        "buy_and_hold_close_to_open_benchmark": {k: benchmark_close_to_open(d, v) for k, v in masks.items()},
        "logistic_threshold_040_test_diagnostic": selected_metrics,
        "logistic_calibration": calibration_rows,
        "model_families": list(factories),
        "limitations": [
            "Annual expanding walk-forward; training labels must mature before each year's first test signal.",
            "2016-2019 research period is used for threshold screening; 2020-2021 validation and 2022-2026 test are not used for selection.",
            "The 2022-2026 historical test period has been examined in prior project iterations; it is not a pristine never-viewed holdout.",
            "Yahoo adjusted daily OHLCV is not official exchange execution data; market-on-close fill is hypothetical.",
            "No market impact, auction imbalance, queue priority, partial fills or broker-specific MOC cut-off simulation.",
            "Model discrimination does not imply profitable trading after costs; this study is exploratory, not a live signal.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/top10_adjusted_prices/bars/2330.csv")
    parser.add_argument("--output", default="artifacts/tsmc_v12_overnight_gap")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), ensure_ascii=False, indent=2, default=str))
