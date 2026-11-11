import tempfile
import unittest
from pathlib import Path
import pandas as pd

from tsmc_2330_false_signal_diagnostics import run

class FalseSignalDiagnosticsTests(unittest.TestCase):
    def test_outcome_classification_uses_future_pivots_only_for_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dates=pd.bdate_range("2024-01-01",periods=60)
            close=[100.0+i*.1 for i in range(60)]
            features=pd.DataFrame({"date":dates,"open":close,"high":[v*1.01 for v in close],
                "low":[v*.99 for v in close],"close":close,"rsi14":[50.0]*60,
                "corporate_action_window":[False]*60})
            events=pd.DataFrame([
                {"threshold_id":"p0.030_atr2.0","type":"HIGH","pivot_date":dates[13]},
                {"threshold_id":"p0.030_atr2.0","type":"LOW","pivot_date":dates[12]},
            ])
            predictions=pd.DataFrame([
                {"threshold_id":"p0.030_atr2.0","analysis_year":2022,"validation_year":2023,
                 "horizon_sessions":3,"pivot_type":"HIGH","date":dates[10],
                 "probability":.9,"threshold":.7,"signal":True,"event":True},
                {"threshold_id":"p0.030_atr2.0","analysis_year":2022,"validation_year":2023,
                 "horizon_sessions":1,"pivot_type":"LOW","date":dates[10],
                 "probability":.8,"threshold":.7,"signal":True,"event":False},
            ])
            features.to_csv(root/"features.csv",index=False)
            events.to_csv(root/"events.csv",index=False)
            predictions.to_csv(root/"predictions.csv",index=False)
            result=run(root/"features.csv",root/"events.csv",root/"predictions.csv",root/"out")
            self.assertEqual(result["signals_analysed"],2)
            detail=pd.read_csv(root/"out"/"tsmc_2330_oof_signal_outcomes.csv")
            high=detail.loc[detail.pivot_type=="HIGH","outcome_class"].iloc[0]
            low=detail.loc[detail.pivot_type=="LOW","outcome_class"].iloc[0]
            self.assertEqual(high,"exact_hit")
            self.assertEqual(low,"early_warning_1_to_5_sessions")
            self.assertTrue((root/"out"/"tsmc_2330_oof_signal_outcome_summary.csv").exists())
            self.assertTrue((root/"out"/"tsmc_2330_oof_signal_feature_medians.csv").exists())

if __name__=="__main__": unittest.main()
