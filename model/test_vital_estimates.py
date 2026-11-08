import unittest
from unittest.mock import patch

import numpy as np

from model import bpnet_inference as bp


class TemperatureTests(unittest.TestCase):
    def test_proposal_uses_percent_without_clipping(self):
        from model.temperature_estimation import estimate_core_temperature, perfusion_percent

        pi = perfusion_percent(20, 1000)
        self.assertEqual(pi, 2.0)
        self.assertAlmostEqual(estimate_core_temperature(34, 27, pi), 37.5897608621)
        self.assertGreater(estimate_core_temperature(40, 10, pi), 43)

    def test_invalid_inputs(self):
        from model.temperature_estimation import estimate_core_temperature, perfusion_percent

        self.assertIsNone(perfusion_percent(20, 0))
        for values in ((None, 27, 2), (34, np.nan, 2), (34, 27, 0)):
            self.assertIsNone(estimate_core_temperature(*values))


class BloodPressureTests(unittest.TestCase):
    def test_invalid_cuff_reference(self):
        self.assertIsNone(bp.parse_cuff_reference("120/130"))
        self.assertIsNone(bp.parse_cuff_reference("120/nan"))
        self.assertIsNone(bp.parse_cuff_reference("1e300/80"))
        self.assertEqual(bp.parse_cuff_reference("120 / 80"), (120, 80))

    def test_anchor_and_overlapping_windows_excluded_and_offsets_reset(self):
        predictor = bp.RevisedBPPredictor.__new__(bp.RevisedBPPredictor)
        signal = np.ones(3750)
        with patch.object(bp, "evaluate_revised_window", return_value=(True, "")):
            with patch.object(predictor, "predict_segment", side_effect=[
                (100, 60), *[(200, 100)] * 4,
                (110, 65), *[(200, 100)] * 4, (112, 67), (100, 60),
            ]):
                calibrated = predictor.predict_recording(signal, signal, cuff_reference=(120, 80), cuff_sample=100)
                uncalibrated = predictor.predict_recording(signal[:1250], signal[:1250])

        self.assertEqual(calibrated["bp_mode"], "calibrated")
        self.assertEqual((calibrated["sbp"], calibrated["dbp"]), (131, 86))
        self.assertEqual(uncalibrated["bp_mode"], "uncalibrated")
        self.assertEqual((uncalibrated["sbp"], uncalibrated["dbp"]), (100, 60))

    def test_no_valid_windows_is_unavailable(self):
        predictor = bp.RevisedBPPredictor.__new__(bp.RevisedBPPredictor)
        signal = np.zeros(1250)
        result = predictor.predict_recording(signal, signal)
        self.assertIsNone(result["sbp"])
        self.assertEqual(result["bp_mode"], "unavailable")

    def test_calibration_requires_a_later_non_overlapping_window(self):
        predictor = bp.RevisedBPPredictor.__new__(bp.RevisedBPPredictor)
        signal = np.ones(1500)
        with patch.object(bp, "evaluate_revised_window", return_value=(True, "")):
            with patch.object(predictor, "predict_segment", return_value=(120, 80)):
                result = predictor.predict_recording(signal, signal, cuff_reference=(120, 80), cuff_sample=100)
        self.assertEqual(result["bp_mode"], "unavailable")


if __name__ == "__main__":
    unittest.main()
