import json
import unittest
from pathlib import Path

import numpy as np

from ml_xgboost.triage_xgboost.preprocessing import (
    FEATURE_ORDER,
    TriagePreprocessingError,
    build_feature_map,
    prepare_onnx_input,
)


NORMAL_VITALS = {
    "temperature_c": 37.0,
    "spo2": 98.0,
    "respiratory_rate": 16.0,
    "heart_rate": 75.0,
    "systolic_bp": 120.0,
    "diastolic_bp": 80.0,
    "gcs_total": 15.0,
}


class TriagePreprocessingTests(unittest.TestCase):
    def test_builds_exact_17_feature_tensor(self):
        features = build_feature_map(NORMAL_VITALS)
        tensor = prepare_onnx_input(NORMAL_VITALS)

        self.assertEqual(tuple(features), FEATURE_ORDER)
        self.assertEqual(tensor.shape, (1, 17))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(tensor)))
        self.assertAlmostEqual(features["mean_arterial_pressure"], 93.333333, places=5)
        self.assertAlmostEqual(features["pulse_pressure"], 40.0)
        self.assertAlmostEqual(features["cardiopulmonary_stress"], 12.0)
        self.assertAlmostEqual(features["news_vital_score"], 0.0)

    def test_feature_order_matches_model_contract(self):
        contract_path = Path(__file__).resolve().parents[1] / "model_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))

        self.assertEqual(list(FEATURE_ORDER), contract["feature_order"])

    def test_clips_all_raw_vitals_to_training_ranges(self):
        features = build_feature_map(
            {
                "temperature_c": 50,
                "spo2": 110,
                "respiratory_rate": 1,
                "heart_rate": 300,
                "systolic_bp": 10,
                "diastolic_bp": 200,
                "gcs_total": 20,
            }
        )

        self.assertEqual(
            [features[name] for name in FEATURE_ORDER[:7]],
            [43.0, 100.0, 4.0, 230.0, 40.0, 160.0, 15.0],
        )

    def test_rejects_missing_values_without_training_medians(self):
        invalid = dict(NORMAL_VITALS)
        invalid["spo2"] = None

        with self.assertRaisesRegex(TriagePreprocessingError, "spo2"):
            prepare_onnx_input(invalid)

    def test_supports_explicit_imputation_values(self):
        invalid = dict(NORMAL_VITALS)
        invalid["spo2"] = np.nan

        tensor = prepare_onnx_input(invalid, imputation_values={"spo2": 97.5})

        self.assertAlmostEqual(float(tensor[0, 1]), 97.5)

    def test_news_score_preserves_training_boundary_behavior(self):
        vitals = dict(NORMAL_VITALS)
        vitals["respiratory_rate"] = 8.5

        features = build_feature_map(vitals)

        # Notebook memakai kondisi 9 <= RR <= 11; nilai desimal 8.5 jatuh
        # ke default 0. Perilaku ini dipertahankan sampai model dilatih ulang.
        self.assertEqual(features["news_vital_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
