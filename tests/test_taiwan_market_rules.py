import unittest
from decimal import Decimal

from taiwan_market_rules import stock_tick_size, is_valid_stock_tick, floor_board_lot


class TaiwanMarketRulesTests(unittest.TestCase):
    def test_stock_tick_price_bands(self):
        cases = [
            (9.99, "0.01"), (10, "0.05"), (49.95, "0.05"),
            (50, "0.10"), (99.9, "0.10"), (100, "0.50"),
            (499.5, "0.50"), (500, "1.00"), (999, "1.00"),
            (1000, "5.00"), (1500, "5.00"),
        ]
        for price, expected in cases:
            with self.subTest(price=price):
                self.assertEqual(stock_tick_size(price), Decimal(expected))

    def test_valid_and_invalid_ticks(self):
        self.assertTrue(is_valid_stock_tick(10.05))
        self.assertFalse(is_valid_stock_tick(10.02))
        self.assertTrue(is_valid_stock_tick(99.9))
        self.assertFalse(is_valid_stock_tick(99.95))
        self.assertTrue(is_valid_stock_tick(1000))
        self.assertFalse(is_valid_stock_tick(1001))

    def test_floor_board_lot(self):
        self.assertEqual(floor_board_lot(2999), 2000)
        self.assertEqual(floor_board_lot(999), 0)
        with self.assertRaises(ValueError):
            floor_board_lot(-1)


if __name__ == "__main__":
    unittest.main()
