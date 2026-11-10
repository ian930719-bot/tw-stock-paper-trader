import unittest
import pandas as pd

from tsmc_2330_selected_threshold_signal_diagnostics import period_events, build_target


class SelectedThresholdDiagnosticsTests(unittest.TestCase):
    def test_period_events_purge_labels_before_period_start(self):
        dates = pd.Series(pd.bdate_range("2020-01-01", periods=12))
        events = pd.DataFrame([
            {"type": "LOW", "pivot_date": dates.iloc[2], "confirmation_date": dates.iloc[3]},
            {"type": "LOW", "pivot_date": dates.iloc[7], "confirmation_date": dates.iloc[8]},
            {"type": "LOW", "pivot_date": dates.iloc[10], "confirmation_date": dates.iloc[11]},
        ])
        selected = period_events(events, dates, "2020-01-06", "2020-01-16", 2)
        self.assertEqual(pd.to_datetime(selected["pivot_date"]).tolist(), [dates.iloc[7], dates.iloc[10]])

    def test_target_marks_only_pre_pivot_sessions(self):
        dates = pd.Series(pd.bdate_range("2020-01-01", periods=10))
        events = pd.DataFrame([
            {"type": "HIGH", "pivot_date": dates.iloc[5]},
        ])
        target = build_target(dates, events, 3, "HIGH")
        self.assertEqual(target.nonzero()[0].tolist(), [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
