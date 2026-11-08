"""Conservative staged-trading defaults for the NT$100,000 pilot account."""
from __future__ import annotations

from dataclasses import dataclass

INITIAL_CAPITAL_NTD = 100_000.0


@dataclass(frozen=True)
class TradingConfig:
    initial_capital: float = INITIAL_CAPITAL_NTD
    max_order_notional: float = 20_000.0
    max_symbol_exposure: float = 25_000.0
    max_portfolio_exposure: float = 80_000.0
    max_daily_loss: float = 2_000.0
    max_open_orders: int = 3
    max_order_count_per_day: int = 8
    risk_budget_per_trade: float = 500.0
    max_quote_age_seconds: int = 60
    allow_short: bool = False
    require_regular_session: bool = True

    def __post_init__(self):
        positive = (
            self.initial_capital, self.max_order_notional,
            self.max_symbol_exposure, self.max_portfolio_exposure,
            self.max_daily_loss, self.risk_budget_per_trade,
            self.max_quote_age_seconds,
        )
        if any(float(value) <= 0 for value in positive):
            raise ValueError("交易資金及風控參數必須大於零")
        if self.max_order_notional > self.initial_capital:
            raise ValueError("單筆委託上限不得超過初始資金")
        if self.max_symbol_exposure > self.initial_capital:
            raise ValueError("單一標的曝險不得超過初始資金")
        if self.max_portfolio_exposure > self.initial_capital:
            raise ValueError("總曝險不得超過初始資金")
        if self.max_daily_loss > self.initial_capital * 0.05:
            raise ValueError("每日停損不得高於初始資金 5%")
        if self.max_open_orders < 1 or self.max_order_count_per_day < 1:
            raise ValueError("委託筆數限制必須大於零")

    def risk_limits(self):
        from execution_engine import RiskLimits
        return RiskLimits(
            max_order_notional=self.max_order_notional,
            max_symbol_exposure=self.max_symbol_exposure,
            max_portfolio_exposure=self.max_portfolio_exposure,
            max_daily_loss=self.max_daily_loss,
            max_open_orders=self.max_open_orders,
            max_order_count_per_day=self.max_order_count_per_day,
            allow_short=self.allow_short,
            require_regular_session=self.require_regular_session,
        )
