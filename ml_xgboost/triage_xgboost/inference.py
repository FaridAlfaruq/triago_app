"""Adapter ONNX Runtime untuk classifier triase terbaru."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .preprocessing import FEATURE_ORDER, build_feature_map, feature_map_to_onnx_input


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "triage_xgboost_model.onnx"
CLASS_LABELS = {
    0: "RESUSITASI",
    1: "DARURAT",
    2: "NON-DARURAT",
}
REPRESENTATIVE_VITALS = {
    "temperature_c": 37.0,
    "spo2": 98.0,
    "respiratory_rate": 16.0,
    "heart_rate": 75.0,
    "systolic_bp": 120.0,
    "diastolic_bp": 80.0,
    "gcs_total": 15.0,
}


class TriageModelError(RuntimeError):
    """Model ONNX tidak tersedia atau menghasilkan output tidak valid."""


@dataclass(frozen=True)
class TriagePrediction:
    class_id: int
    label: str
    confidence: float
    probabilities: tuple[float, float, float]
    inference_ms: float
    feature_values: dict[str, float]


class TriageOnnxPredictor:
    """Muat model sekali dan gunakan ulang sesi untuk seluruh pemeriksaan GUI."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        session: Any | None = None,
        warm_up: bool = True,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        started = perf_counter()

        if session is None:
            if not self.model_path.is_file():
                raise TriageModelError(f"Model ONNX tidak ditemukan: {self.model_path}")

            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise TriageModelError(
                    "onnxruntime belum terpasang. Jalankan instalasi dari requirements.txt."
                ) from exc

            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.log_severity_level = 3
            try:
                session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as exc:
                raise TriageModelError(f"Gagal membuka model ONNX: {exc}") from exc

        self.session = session
        self.input_name = self._validate_session_contract()
        self.load_ms = (perf_counter() - started) * 1000.0

        if warm_up:
            self.predict(REPRESENTATIVE_VITALS)

    def _validate_session_contract(self) -> str:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if len(inputs) != 1:
            raise TriageModelError(f"Model harus memiliki satu input, ditemukan {len(inputs)}.")

        input_meta = inputs[0]
        input_shape = list(input_meta.shape)
        if not input_shape or input_shape[-1] != len(FEATURE_ORDER):
            raise TriageModelError(
                f"Model mengharapkan bentuk {input_shape}, bukan 17 fitur."
            )

        output_names = {output.name for output in outputs}
        required_outputs = {"label", "probabilities"}
        if not required_outputs.issubset(output_names):
            raise TriageModelError(
                f"Output model {sorted(output_names)} tidak memuat label dan probabilities."
            )

        return input_meta.name

    def predict(self, vitals: dict[str, object]) -> TriagePrediction:
        feature_values = build_feature_map(vitals)
        tensor = feature_map_to_onnx_input(feature_values)

        started = perf_counter()
        try:
            labels, probability_rows = self.session.run(
                ["label", "probabilities"],
                {self.input_name: tensor},
            )
        except Exception as exc:
            raise TriageModelError(f"Inferensi ONNX gagal: {exc}") from exc
        inference_ms = (perf_counter() - started) * 1000.0

        class_id = int(np.asarray(labels).reshape(-1)[0])
        probabilities_array = np.asarray(probability_rows, dtype=np.float32).reshape(-1)

        if class_id not in CLASS_LABELS:
            raise TriageModelError(f"Kelas keluaran tidak dikenal: {class_id}")
        if probabilities_array.size != len(CLASS_LABELS):
            raise TriageModelError(
                f"Jumlah probabilitas {probabilities_array.size}, diharapkan 3."
            )
        if not np.all(np.isfinite(probabilities_array)):
            raise TriageModelError("Model menghasilkan probabilitas non-finite.")

        probabilities = tuple(float(value) for value in probabilities_array)
        return TriagePrediction(
            class_id=class_id,
            label=CLASS_LABELS[class_id],
            confidence=probabilities[class_id],
            probabilities=probabilities,
            inference_ms=inference_ms,
            feature_values={name: float(feature_values[name]) for name in FEATURE_ORDER},
        )
