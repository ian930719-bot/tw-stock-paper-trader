"""Broker integration contract.

No concrete broker is selected or connected in the staged release. This module
contains no network calls and cannot submit real orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BrokerOrderRequest:
    symbol: str
    side: str
    quantity: int
    price: float
    price_type: str = "LIMIT"
    time_in_force: str = "ROD"
    client_order_id: str = ""


@dataclass(frozen=True)
class BrokerOrderResult:
    accepted: bool
    broker_order_id: str | None
    status: str
    message: str


class BrokerGateway(Protocol):
    """Interface every future broker adapter must implement."""

    def account_status(self) -> dict:
        ...

    def submit_order(self, request: BrokerOrderRequest) -> BrokerOrderResult:
        ...

    def cancel_order(self, broker_order_id: str) -> BrokerOrderResult:
        ...

    def list_open_orders(self) -> list[dict]:
        ...

    def list_fills(self, since: str | None = None) -> list[dict]:
        ...


class LiveTradingDisabled(RuntimeError):
    """Raised whenever live routing is requested before staged validation."""


class UnconfiguredBroker:
    """Explicit placeholder; never simulates a successful broker response."""

    broker_name = None
    is_live = False

    def account_status(self) -> dict:
        raise LiveTradingDisabled("尚未指定並設定券商，禁止讀取實盤帳戶")

    def submit_order(self, request: BrokerOrderRequest) -> BrokerOrderResult:
        raise LiveTradingDisabled("實盤交易尚未開放：請先完成券商介接、驗證及人工啟用")

    def cancel_order(self, broker_order_id: str) -> BrokerOrderResult:
        raise LiveTradingDisabled("尚未連接券商，無法撤銷真實委託")

    def list_open_orders(self) -> list[dict]:
        raise LiveTradingDisabled("尚未連接券商，無法讀取真實委託")

    def list_fills(self, since: str | None = None) -> list[dict]:
        raise LiveTradingDisabled("尚未連接券商，無法讀取真實成交")
