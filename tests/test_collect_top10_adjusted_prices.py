import unittest
import pandas as pd
from collect_top10_adjusted_prices import normalize, normalize_yahoo


class AdjustedPriceNormalizationTests(unittest.TestCase):
    def test_finmind_adjusted_schema_is_normalized(self):
        rows=[{"date":"2024-01-02","stock_id":"2330","open":100,"max":102,"min":99,"close":101,"Trading_Volume":123456,"Trading_money":12000000}]
        frame=normalize(rows,"2330")
        self.assertEqual(frame.loc[0,"high"],102)
        self.assertEqual(frame.loc[0,"volume"],123456)
        self.assertEqual(frame.loc[0,"name"],"台積電")
        self.assertEqual(frame.loc[0,"turnover"],12000000)

    def test_yahoo_auto_adjust_schema_is_normalized_and_keeps_actions(self):
        history=pd.DataFrame({"Open":[95.0],"High":[97.0],"Low":[94.0],"Close":[96.0],"Volume":[1000],"Dividends":[1.0],"Stock Splits":[0.0]},index=pd.DatetimeIndex(["2024-01-02"],name="Date"))
        frame=normalize_yahoo(history,"2330")
        self.assertEqual(frame.loc[0,"close"],96.0)
        self.assertEqual(frame.loc[0,"dividends"],1.0)
        self.assertEqual(frame.loc[0,"stock_splits"],0.0)
        self.assertTrue(pd.isna(frame.loc[0,"turnover"]))

    def test_invalid_ohlc_is_rejected(self):
        rows=[{"date":"2024-01-02","stock_id":"2330","open":100,"max":98,"min":99,"close":101,"Trading_Volume":123456}]
        with self.assertRaises(RuntimeError): normalize(rows,"2330")


if __name__=="__main__": unittest.main()
