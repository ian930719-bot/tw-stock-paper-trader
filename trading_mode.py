"""Fail-closed staged/live mode policy."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from broker_gateway import UnconfiguredBroker, LiveTradingDisabled


class TradingMode(str, Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"


@dataclass(frozen=True)
class Readiness:
    broker_selected: bool = False
    broker_adapter_implemented: bool = False
    credentials_configured: bool = False
    account_verified: bool = False
    paper_validation_passed: bool = False
    operator_live_approval: bool = False
    kill_switch_clear: bool = False


def live_trading_blockers(readiness: Readiness) -> list[str]:
    checks = (
        (readiness.broker_selected, "尚未選定券商"),
        (readiness.broker_adapter_implemented, "券商 API 介接尚未實作"),
        (readiness.credentials_configured, "券商憑證尚未安全設定"),
        (readiness.account_verified, "尚未核對實盤帳戶及交易權限"),
        (readiness.paper_validation_passed, "模擬交易驗證尚未通過"),
        (readiness.operator_live_approval, "尚未取得人工實盤啟用確認"),
        (readiness.kill_switch_clear, "緊急停止開關未解除"),
    )
    return [message for passed, message in checks if not passed]


class StagedTradingController:
    def __init__(self, broker=None):
        self.broker = broker or UnconfiguredBroker()
        self.mode = TradingMode.PAPER

    def enable_live(self, readiness: Readiness) -> None:
        blockers = live_trading_blockers(readiness)
        if blockers:
            raise LiveTradingDisabled("實盤模式仍被鎖定：" + "；".join(blockers))
        # A future concrete adapter must explicitly identify itself as live.
        if not getattr(self.broker, "is_live", False):
            raise LiveTradingDisabled("目前沒有通過驗證的實盤券商 Adapter")
        self.mode = TradingMode.LIVE

    def submit_live_order(self, request):
        if self.mode is not TradingMode.LIVE:
            raise LiveTradingDisabled("目前為 staged 模式，只允許模擬交易")
        return self.broker.submit_order(request)
