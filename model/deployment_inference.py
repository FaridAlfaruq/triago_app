# -*- coding: utf-8 -*-
"""Modul deployment_inference.py

Menyediakan wrapper TriageOnnxModel untuk melayani inferensi ONNX runtime
secara langsung pada aplikasi GUI TriaGO.
"""

import os
from pathlib import Path
import numpy as np
import onnxruntime as rt

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX_PATH = MODEL_DIR / "triage_xgboost_model.onnx"

TRIAGE_FEATURES = [
    "temperature_c", "spo2", "respiratory_rate", "heart_rate",
    "systolic_bp", "diastolic_bp", "gcs_total", "mean_arterial_pressure",
    "pulse_pressure", "shock_index", "modified_shock_index",
    "spo2_to_rr_ratio", "sys_to_rr_ratio", "pp_to_sys_ratio", "hr_to_rr_ratio",
    "temp_deviation", "oxygen_deficit", "gcs_deficit",
    "cardiopulmonary_stress", "neuro_hemodynamic_index", "news_vital_score",
]
TRIAGE_LABELS = {0: "RESUSITASI", 1: "DARURAT", 2: "NON-DARURAT"}


def _news_score(vitals: dict[str, float]) -> float:
    rr, spo2, sbp = vitals["respiratory_rate"], vitals["spo2"], vitals["systolic_bp"]
    hr, temp, gcs = vitals["heart_rate"], vitals["temperature_c"], vitals["gcs_total"]
    rr_score = 3 if rr <= 8 or rr >= 25 else 1 if rr <= 11 else 0 if rr <= 20 else 2
    spo2_score = 3 if spo2 <= 91 else 2 if spo2 <= 93 else 1 if spo2 <= 95 else 0
    sbp_score = 3 if sbp <= 90 or sbp >= 220 else 2 if sbp <= 100 else 1 if sbp <= 110 else 0
    hr_score = 3 if hr <= 40 or hr >= 131 else 1 if hr <= 50 or hr <= 110 else 0 if hr <= 90 else 2
    temp_score = 3 if temp <= 35 else 1 if temp <= 36 else 0 if temp <= 38 else 1 if temp <= 39 else 2
    gcs_score = 0 if gcs == 15 else 1 if gcs >= 13 else 2 if gcs >= 9 else 3
    return float(rr_score + spo2_score + sbp_score + hr_score + temp_score + gcs_score)


def build_triage_input(vitals: dict[str, float]) -> np.ndarray:
    """Membangun vektor 21 fitur dengan urutan identik XGBoost 90%+."""
    limits = {
        "temperature_c": (30.0, 43.0), "spo2": (50.0, 100.0),
        "respiratory_rate": (4.0, 60.0), "heart_rate": (20.0, 230.0),
        "systolic_bp": (40.0, 260.0), "diastolic_bp": (20.0, 160.0),
        "gcs_total": (3.0, 15.0),
    }
    x = {key: float(np.clip(vitals.get(key, 0.0), *range_)) for key, range_ in limits.items()}
    map_value = x["diastolic_bp"] + (x["systolic_bp"] - x["diastolic_bp"]) / 3.0
    pulse_pressure = x["systolic_bp"] - x["diastolic_bp"]
    shock_index = x["heart_rate"] / (x["systolic_bp"] + 0.1)

    x.update({
        "mean_arterial_pressure": map_value,
        "pulse_pressure": pulse_pressure,
        "shock_index": shock_index,
        "modified_shock_index": x["heart_rate"] / (map_value + 0.1),
        "spo2_to_rr_ratio": x["spo2"] / (x["respiratory_rate"] + 0.1),
        "sys_to_rr_ratio": x["systolic_bp"] / (x["respiratory_rate"] + 0.1),
        "pp_to_sys_ratio": pulse_pressure / (x["systolic_bp"] + 0.1),
        "hr_to_rr_ratio": x["heart_rate"] / (x["respiratory_rate"] + 0.1),
        "temp_deviation": abs(x["temperature_c"] - 37.0),
        "oxygen_deficit": max(0.0, 98.0 - x["spo2"]),
        "gcs_deficit": 15.0 - x["gcs_total"],
        "cardiopulmonary_stress": x["heart_rate"] * x["respiratory_rate"] / 100.0,
    })
    x["neuro_hemodynamic_index"] = shock_index * (x["gcs_deficit"] + 1.0)
    x["news_vital_score"] = _news_score(x)
    return np.asarray([[x[name] for name in TRIAGE_FEATURES]], dtype=np.float32)


class TriageOnnxModel:
    """Wrapper untuk memuat dan melakukan prediksi dari file ONNX."""
    def __init__(self, model_path=None):
        self.model_path = model_path or DEFAULT_ONNX_PATH
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"File ONNX tidak ditemukan di: {self.model_path}")
        self.session = rt.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, vitals: dict[str, float]):
        """Memprediksi status triase berdasarkan dictionary tanda vital."""
        input_data = build_triage_input(vitals)
        outputs = self.session.run(None, {self.input_name: input_data})
        raw_prob = outputs[1] if len(outputs) > 1 else outputs[0]
        if isinstance(raw_prob, list) and isinstance(raw_prob[0], dict):
            prob_vec = np.array(list(raw_prob[0].values()))
        else:
            prob_vec = np.squeeze(np.array(raw_prob))

        pred_class = int(np.argmax(prob_vec))
        label_str = TRIAGE_LABELS.get(pred_class, "DARURAT")
        confidence = float(prob_vec[pred_class])
        return label_str, confidence, prob_vec
