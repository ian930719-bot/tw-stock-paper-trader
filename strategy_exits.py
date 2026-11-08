"""Auditable exit-rule primitives; candidate parameters require Taiwan-market OOS validation.

These functions do not place orders and deliberately separate price risk from
portfolio risk. O'Neil/Minervini values are research references, not guarantees.
"""
from __future__ import annotations

import math

PROFILES = {
    "oneil_growth": {
        "name": "O'Neil 成長股",
        "description": "7–8% 初始最大損失參考；20–25% 獲利區間進行檢視，不是無條件硬性賣出。符合強勢領先股條件時需另行評估延長持有。",
        "initial_stop_reference": (0.07, 0.08),
        "profit_review_reference": (0.20, 0.25),
    },
    "minervini_sepa": {
        "name": "Minervini SEPA",
        "description": "依技術結構失效點設停損；至少 2R、偏好 3R 作為候選風險報酬門檻。沒有合理失效點就不交易。",
        "minimum_reward_risk": 2.0,
        "preferred_reward_risk": 3.0,
    },
    "atr_dynamic": {
        "name": "ATR 動態風控",
        "description": "依 ATR 倍數與技術失效點建立初始風險；倍數是待驗證參數，不宣稱有通用最佳值。",
        "candidate_atr_multipliers": (1.5, 2.0, 2.5, 3.0),
    },
}


def initial_stop(entry_price: float, profile: str, *, technical_invalidation: float | None = None,
                 atr: float | None = None, atr_multiple: float = 2.0,
                 oneil_loss_cap: float = 0.08) -> dict:
    entry = _positive(entry_price, "entry_price")
    if profile == "oneil_growth":
        cap = _fraction(oneil_loss_cap, "oneil_loss_cap")
        stop = entry * (1.0 - cap)
        if technical_invalidation is not None:
            support = _positive(technical_invalidation, "technical_invalidation")
            stop = max(stop, support) if support < entry else stop
        return {"profile": profile, "stop_price": stop, "method": "O'Neil 7–8% 參考上限＋技術位置檢視"}
    if profile == "minervini_sepa":
        if technical_invalidation is None:
            raise ValueError("Minervini 策略必須提供技術失效價；沒有合理位置就不交易")
        stop = _positive(technical_invalidation, "technical_invalidation")
        if stop >= entry:
            raise ValueError("技術失效價必須低於進場價")
        return {"profile": profile, "stop_price": stop, "method": "技術結構失效點"}
    if profile == "atr_dynamic":
        volatility = _positive(atr, "atr")
        multiple = _positive(atr_multiple, "atr_multiple")
        stop = entry - volatility * multiple
        if stop <= 0:
            raise ValueError("ATR 停損計算結果不合理")
        if technical_invalidation is not None:
            support = _positive(technical_invalidation, "technical_invalidation")
            if support >= entry:
                raise ValueError("技術失效價必須低於進場價")
            stop = max(stop, support)
        return {"profile": profile, "stop_price": stop, "method": f"ATR × {multiple:g}（候選參數）"}
    raise ValueError(f"未知策略：{profile}")


def risk_position_size(equity: float, risk_fraction: float, entry_price: float,
                       stop_price: float, *, max_shares: int | None = None) -> dict:
    """Shares from account risk budget; excludes cash, fees, lot and liquidity caps."""
    capital = _positive(equity, "equity")
    fraction = _fraction(risk_fraction, "risk_fraction")
    entry = _positive(entry_price, "entry_price")
    stop = _positive(stop_price, "stop_price")
    if stop >= entry:
        raise ValueError("停損價必須低於進場價")
    risk_budget = capital * fraction
    per_share_risk = entry - stop
    shares = math.floor(risk_budget / per_share_risk)
    if max_shares is not None:
        if isinstance(max_shares, bool) or int(max_shares) < 0:
            raise ValueError("max_shares 必須是非負整數")
        shares = min(shares, int(max_shares))
    return {
        "risk_budget": risk_budget,
        "risk_per_share": per_share_risk,
        "shares": shares,
        "estimated_price_risk": shares * per_share_risk,
        "risk_fraction": fraction,
        "note": "尚須再受現金、單筆／單股曝險、交易單位、費稅、流動性及漲跌停限制",
    }


def trailing_stop(high_water: float, atr: float, atr_multiple: float,
                  previous_stop: float | None = None) -> float:
    high = _positive(high_water, "high_water")
    volatility = _positive(atr, "atr")
    multiple = _positive(atr_multiple, "atr_multiple")
    candidate = high - volatility * multiple
    if candidate <= 0:
        raise ValueError("移動停損計算結果不合理")
    # Long-position protective stops may rise but never loosen.
    return max(candidate, float(previous_stop)) if previous_stop is not None else candidate


def _positive(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必須是正數")
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} 必須是正數")
    return number


def _fraction(value, name):
    number = _positive(value, name)
    if number >= 1:
        raise ValueError(f"{name} 必須小於 1")
    return number
