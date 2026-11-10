import unittest
import numpy as np
import pandas as pd

from tsmc_2330_walk_forward import annual_fold_years, event_target, fold_masks, fold_events


class AnnualWalkForwardTests(unittest.TestCase):
    def test_expanding_year_sequence(self):
        dates = pd.Series(pd.to_datetime(["2010-01-04", "2011-01-03", "2012-01-03"]))
        self.assertEqual(annual_fold_years(dates), [2011, 2012])

    def test_training_fold_ends_before_validation_year(self):
        dates = pd.Series(pd.bdate_range("2010-01-01", "2012-12-31"))
        valid = np.ones(len(dates), dtype=bool)
        fold = fold_masks(dates, valid, 2012, 3)
        self.assertLess(pd.Timestamp(fold["training_end"]), pd.Timestamp("2012-01-01"))
        self.assertGreaterEqual(pd.Timestamp(fold["validation_start"]), pd.Timestamp("2012-01-01"))
        self.assertLessEqual(fold["train_mask"].nonzero()[0].max(), fold["train_end_idx"] - 3)

    def test_event_target_marks_only_pre_pivot_sessions(self):
        dates = pd.Series(pd.bdate_range("2010-01-01", periods=10))
        events = pd.DataFrame([{"type": "LOW", "pivot_date": dates.iloc[5]}])
        target = event_target(dates, events, 3, "LOW")
        self.assertEqual(np.flatnonzero(target).tolist(), [2, 3, 4])

    def test_training_events_must_be_confirmed_before_cutoff(self):
        dates = pd.Series(pd.bdate_range("2010-01-01", "2012-12-31"))
        cutoff = int(np.flatnonzero(dates.dt.year.to_numpy() == 2011)[-1])
        fold = {
            "train_end_idx": cutoff,
            "validation_start_idx": cutoff + 1,
            "validation_end_idx": len(dates) - 1,
        }
        events = pd.DataFrame([
            {"type": "LOW", "pivot_date": dates.iloc[cutoff - 10],
             "confirmation_date": dates.iloc[cutoff - 5]},
            {"type": "LOW", "pivot_date": dates.iloc[cutoff - 4],
             "confirmation_date": dates.iloc[cutoff + 2]},
        ])
        train, _ = fold_events(events, dates, fold, 3)
        self.assertEqual(len(train), 1)


if __name__ == "__main__":
    unittest.main()
