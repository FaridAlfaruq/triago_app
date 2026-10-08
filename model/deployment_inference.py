# -*- coding: utf-8 -*-
"""Modul deployment_inference.py

Menyediakan wrapper TriageOnnxModel untuk melayani inferensi ONNX runtime
tersinkronisasi dengan model XGBoost hasil hyperparameter tuning (11 Fitur termasuk Pain Score).
"""

import os
from pathlib import Path
import numpy as np
import onnxruntime as rt

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX_PATH = MODEL_DIR / "triage_xgboost_model.onnx"

TRIAGE_FEATURES = [
    "temperature_c",
    "spo2",
    "respiratory_rate",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "gcs_total",
    "pain_score",
    "shock_index",
    "pulse_pressure",
    "MAP",
]
TRIAGE_LABELS = {0: "RESUSITASI", 1: "DARURAT", 2: "NON-DARURAT"}


def build_triage_input(vitals: dict[str, float]) -> np.ndarray:
    """Membangun vektor 11 fitur dengan urutan identik model XGBoost Tuned (Triage_XGBoost)."""
    limits = {
        "temperature_c": (30.0, 43.0),
        "spo2": (50.0, 100.0),
        "respiratory_rate": (4.0, 60.0),
        "heart_rate": (20.0, 230.0),
        "systolic_bp": (40.0, 260.0),
        "diastolic_bp": (20.0, 160.0),
        "gcs_total": (3.0, 15.0),
        "pain_score": (0.0, 10.0),
    }
    x = {key: float(np.clip(vitals.get(key, 0.0), *range_)) for key, range_ in limits.items()}
    sys_safe = max(x["systolic_bp"], 1.0)
    dia_safe = max(x["diastolic_bp"], 1.0)

    pulse_pressure = x["systolic_bp"] / dia_safe
    shock_index = x["heart_rate"] / sys_safe
    map_val = x["diastolic_bp"] + (pulse_pressure / 3.0)

    x.update({
        "shock_index": shock_index,
        "pulse_pressure": pulse_pressure,
        "MAP": map_val,
    })
    return np.asarray([[x[name] for name in TRIAGE_FEATURES]], dtype=np.float32)


class TriageOnnxModel:
    """Wrapper untuk memuat dan melakukan prediksi dari file ONNX XGBoost Ter-tuning."""

    def __init__(self, model_path=None):
        self.model_path = model_path or DEFAULT_ONNX_PATH
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"File ONNX tidak ditemukan di: {self.model_path}")
        self.session = rt.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, vitals: dict[str, float]):
        """Memprediksi status triase dengan model ONNX XGBoost + Dynamic Medical Safety Guardrails."""
        input_data = build_triage_input(vitals)
        outputs = self.session.run(None, {self.input_name: input_data})
        raw_prob = outputs[1] if len(outputs) > 1 else outputs[0]
        if isinstance(raw_prob, list) and isinstance(raw_prob[0], dict):
            prob_vec = np.array(list(raw_prob[0].values()), dtype=np.float32)
        else:
            prob_vec = np.squeeze(np.array(raw_prob, dtype=np.float32))

        pred_class = int(np.argmax(prob_vec))
        label_str = TRIAGE_LABELS.get(pred_class, "DARURAT")
        confidence = float(prob_vec[pred_class])

        # ---------------------------------------------------------------------
        # DYNAMIC MEDICAL SAFETY GUARDRAILS ENGINE
        # ---------------------------------------------------------------------
        rr = float(vitals.get("respiratory_rate", 16.0))
        spo2 = float(vitals.get("spo2", 98.0))
        gcs = float(vitals.get("gcs_total", 15.0))
        hr = float(vitals.get("heart_rate", 75.0))
        sbp = float(vitals.get("systolic_bp", 120.0))
        temp = float(vitals.get("temperature_c", 36.5))
        pain = float(vitals.get("pain_score", 0.0))

        # 1. Critical Red Flag Escalation
        if gcs <= 8.0 or spo2 <= 88.0 or sbp <= 80.0 or hr <= 40.0:
            return "RESUSITASI", 0.99, prob_vec

        # 2. Severe Pain Escalation (Nyeri hebat >= 8 tidak boleh non-darurat)
        if pain >= 8.0 and label_str == "NON-DARURAT":
            return "DARURAT", max(0.85, confidence), prob_vec

        # 3. Dynamic De-escalation (Takipnea ringan terisolasi dengan tanda vital normal & nyeri rendah)
        is_supporting_vitals_healthy = (
            spo2 >= 96.0 and
            gcs == 15.0 and
            pain <= 3.0 and
            (60.0 <= hr <= 90.0) and
            (100.0 <= sbp <= 135.0) and
            (36.0 <= temp <= 37.8)
        )

        if label_str == "DARURAT" and 21.0 <= rr <= 23.0 and is_supporting_vitals_healthy:
            return "NON-DARURAT", 0.85, prob_vec

        return label_str, confidence, prob_vec
