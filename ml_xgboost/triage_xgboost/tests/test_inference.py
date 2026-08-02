import importlib.util
import unittest

import numpy as np

from ml_xgboost.triage_xgboost.inference import (
    REPRESENTATIVE_VITALS,
    TriageOnnxPredictor,
)


class _TensorMeta:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _FakeSession:
    def get_inputs(self):
        return [_TensorMeta("float_input", [None, 17])]

    def get_outputs(self):
        return [
            _TensorMeta("label", [None]),
            _TensorMeta("probabilities", [None, 3]),
        ]

    def run(self, output_names, feed):
        self.output_names = output_names
        self.tensor = feed["float_input"]
        return np.array([1], dtype=np.int64), np.array(
            [[0.1, 0.8, 0.1]], dtype=np.float32
        )


class TriageInferenceTests(unittest.TestCase):
    def test_adapter_maps_outputs_and_confidence(self):
        session = _FakeSession()
        predictor = TriageOnnxPredictor(session=session, warm_up=False)

        result = predictor.predict(REPRESENTATIVE_VITALS)

        self.assertEqual(result.class_id, 1)
        self.assertEqual(result.label, "DARURAT")
        self.assertAlmostEqual(result.confidence, 0.8, places=6)
        self.assertEqual(session.tensor.shape, (1, 17))
        self.assertEqual(session.tensor.dtype, np.float32)
        self.assertEqual(session.output_names, ["label", "probabilities"])

    @unittest.skipUnless(
        importlib.util.find_spec("onnxruntime"),
        "onnxruntime belum terpasang",
    )
    def test_real_model_smoke_inference(self):
        predictor = TriageOnnxPredictor(warm_up=True)

        result = predictor.predict(REPRESENTATIVE_VITALS)

        self.assertIn(result.class_id, (0, 1, 2))
        self.assertEqual(len(result.probabilities), 3)
        self.assertAlmostEqual(sum(result.probabilities), 1.0, places=4)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
