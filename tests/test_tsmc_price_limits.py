import unittest

from tsmc_2330_price_limits import tick_size, limit_price


class TSMCPriceLimitTests(unittest.TestCase):
    def test_official_tick_bands(self):
        self.assertEqual(float(tick_size(9.99)), 0.01)
        self.assertEqual(float(tick_size(10.0)), 0.05)
        self.assertEqual(float(tick_size(50.0)), 0.1)
        self.assertEqual(float(tick_size(100.0)), 0.5)
        self.assertEqual(float(tick_size(500.0)), 1.0)
        self.assertEqual(float(tick_size(1000.0)), 5.0)

    def test_twse_official_example_rounding(self):
        self.assertAlmostEqual(limit_price(40.60, 0.10, "UP"), 44.65)
        self.assertAlmostEqual(limit_price(40.60, 0.10, "DOWN"), 36.55)

    def test_rounding_when_limit_crosses_tick_band(self):
        self.assertAlmostEqual(limit_price(49.90, 0.10, "UP"), 54.80)
        self.assertAlmostEqual(limit_price(49.90, 0.10, "DOWN"), 44.95)

    def test_historical_seven_percent_band(self):
        self.assertAlmostEqual(limit_price(40.60, 0.07, "UP"), 43.40)
        self.assertAlmostEqual(limit_price(40.60, 0.07, "DOWN"), 37.80)

    def test_invalid_price_is_rejected(self):
        with self.assertRaises(ValueError):
            limit_price(0, 0.10, "UP")


if __name__ == "__main__":
    unittest.main()
