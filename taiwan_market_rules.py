"""Taiwan listed/OTC common-stock order increments and standard board lots.

Scope is ordinary listed/OTC stocks and domestic listed ETFs using 1,000-share
board lots. This module deliberately does not model special securities,
auction-only instruments, IPO no-limit sessions, odd-lot sessions, or price
limits without an authoritative daily reference price.
"""
from decimal import Decimal, InvalidOperation


def stock_tick_size(price) -> Decimal:
    value = Decimal(str(price))
    if not value.is_finite() or value <= 0:
        raise ValueError("價格必須是有限正數")
    if value < 10:
        return Decimal("0.01")
    if value < 50:
        return Decimal("0.05")
    if value < 100:
        return Decimal("0.10")
    if value < 500:
        return Decimal("0.50")
    if value < 1000:
        return Decimal("1.00")
    return Decimal("5.00")


def is_valid_stock_tick(price) -> bool:
    try:
        value = Decimal(str(price))
        tick = stock_tick_size(value)
        return value % tick == 0
    except (InvalidOperation, ValueError, TypeError):
        return False


def floor_board_lot(quantity: int, board_lot: int = 1000) -> int:
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0:
        raise ValueError("股數必須是非負整數")
    if board_lot <= 0:
        raise ValueError("交易單位必須大於零")
    return (quantity // board_lot) * board_lot
