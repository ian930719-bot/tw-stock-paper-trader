import unittest
import pandas as pd

from fundamental_factors import score_fundamental_snapshot


class FundamentalFactorTests(unittest.TestCase):
    def test_ranks_quality_and_penalizes_expensive_and_leveraged(self):
        data = pd.DataFrame({
            "代號": ["A", "B", "C"],
            "ROE": [0.20, 0.10, 0.05],
            "營收成長率": [0.30, 0.10, -0.05],
            "負債／權益": [20, 80, 150],
            "本益比": [15, 25, 40],
        })
        result = score_fundamental_snapshot(data)
        self.assertGreater(result.loc[0, "基本面分數"], result.loc[2, "基本面分數"])
        self.assertEqual(result.loc[0, "基本面排名"], 1)

    def test_missing_data_is_not_scored_as_zero(self):
        result = score_fundamental_snapshot(pd.DataFrame({
            "代號": ["A", "B"], "ROE": [None, None]
        }))
        self.assertTrue(result["基本面分數"].isna().all())
        self.assertTrue((result["基本面資料完整度"] == 0).all())

    def test_coverage_is_explicit(self):
        result = score_fundamental_snapshot(pd.DataFrame({
            "ROE": [0.2, None], "營收成長率": [0.3, 0.1]
        }))
        self.assertEqual(result.loc[0, "基本面資料完整度"], 1.0)
        self.assertEqual(result.loc[1, "基本面資料完整度"], 0.5)


if __name__ == "__main__":
    unittest.main()
