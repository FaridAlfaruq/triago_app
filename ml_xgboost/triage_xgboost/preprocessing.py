"""Preprocessing yang kompatibel dengan training model triase ONNX."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


RAW_FEATURES = (
    "temperature_c",
    "spo2",
    "respiratory_rate",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "gcs_total",
)

FEATURE_ORDER = RAW_FEATURES + (
    "shock_index",
    "mean_arterial_pressure",
    "pulse_pressure",
    "modified_shock_index",
    "temp_deviation",
    "oxygen_deficit",
    "gcs_deficit",
    "cardiopulmonary_stress",
    "neuro_hemodynamic_index",
    "news_vital_score",
)

CLIP_RANGES = {
    "temperature_c": (30.0, 43.0),
    "spo2": (50.0, 100.0),
    "respiratory_rate": (4.0, 60.0),
    "heart_rate": (20.0, 230.0),
    "systolic_bp": (40.0, 260.0),
    "diastolic_bp": (20.0, 160.0),
    "gcs_total": (3.0, 15.0),
}


class TriagePreprocessingError(ValueError):
    """Input vital tidak dapat diubah menjadi input model yang valid."""


def sanitize_vitals(
    vitals: Mapping[str, object],
    *,
    imputation_values: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Validasi, imputasi opsional, dan clipping tujuh tanda vital mentah."""

    clean: dict[str, float] = {}
    invalid_features: list[str] = []

    for name in RAW_FEATURES:
        raw_value = vitals.get(name)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = np.nan

        if not np.isfinite(value):
            replacement = None if imputation_values is None else imputation_values.get(name)
            try:
                value = float(replacement)
            except (TypeError, ValueError):
                value = np.nan

        if not np.isfinite(value):
            invalid_features.append(name)
            continue

        lower, upper = CLIP_RANGES[name]
        clean[name] = float(np.clip(value, lower, upper))

    if invalid_features:
        joined = ", ".join(invalid_features)
        raise TriagePreprocessingError(
            f"Tanda vital hilang atau tidak valid: {joined}. "
            "Median training belum tersedia, sehingga input tidak boleh diimputasi diam-diam."
        )

    return clean


def calculate_news_subscore(vitals: Mapping[str, float]) -> float:
    """Replikasi persis aturan NEWS-adapted yang dipakai saat training."""

    rr = vitals["respiratory_rate"]
    if rr <= 8:
        rr_score = 3
    elif 9 <= rr <= 11:
        rr_score = 1
    elif 12 <= rr <= 20:
        rr_score = 0
    elif 21 <= rr <= 24:
        rr_score = 2
    elif rr >= 25:
        rr_score = 3
    else:
        rr_score = 0

    spo2 = vitals["spo2"]
    if spo2 <= 91:
        spo2_score = 3
    elif 92 <= spo2 <= 93:
        spo2_score = 2
    elif 94 <= spo2 <= 95:
        spo2_score = 1
    elif spo2 >= 96:
        spo2_score = 0
    else:
        spo2_score = 0

    sbp = vitals["systolic_bp"]
    if sbp <= 90:
        sbp_score = 3
    elif 91 <= sbp <= 100:
        sbp_score = 2
    elif 101 <= sbp <= 110:
        sbp_score = 1
    elif 111 <= sbp <= 219:
        sbp_score = 0
    elif sbp >= 220:
        sbp_score = 3
    else:
        sbp_score = 0

    heart_rate = vitals["heart_rate"]
    if heart_rate <= 40:
        hr_score = 3
    elif 41 <= heart_rate <= 50:
        hr_score = 1
    elif 51 <= heart_rate <= 90:
        hr_score = 0
    elif 91 <= heart_rate <= 110:
        hr_score = 1
    elif 111 <= heart_rate <= 130:
        hr_score = 2
    elif heart_rate >= 131:
        hr_score = 3
    else:
        hr_score = 0

    temperature = vitals["temperature_c"]
    if temperature <= 35.0:
        temp_score = 3
    elif 35.1 <= temperature <= 36.0:
        temp_score = 1
    elif 36.1 <= temperature <= 38.0:
        temp_score = 0
    elif 38.1 <= temperature <= 39.0:
        temp_score = 1
    elif temperature >= 39.1:
        temp_score = 2
    else:
        temp_score = 0

    gcs = vitals["gcs_total"]
    if gcs == 15:
        gcs_score = 0
    elif 13 <= gcs <= 14:
        gcs_score = 1
    elif 9 <= gcs <= 12:
        gcs_score = 2
    elif gcs <= 8:
        gcs_score = 3
    else:
        gcs_score = 0

    return float(rr_score + spo2_score + sbp_score + hr_score + temp_score + gcs_score)


def build_feature_map(
    vitals: Mapping[str, object],
    *,
    imputation_values: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Bangun tujuh fitur mentah dan sepuluh fitur turunan sesuai notebook."""

    features = sanitize_vitals(vitals, imputation_values=imputation_values)

    heart_rate = features["heart_rate"]
    systolic_bp = features["systolic_bp"]
    diastolic_bp = features["diastolic_bp"]
    gcs_deficit = 15.0 - features["gcs_total"]

    shock_index = heart_rate / (systolic_bp + 0.1)
    mean_arterial_pressure = diastolic_bp + (
        (systolic_bp - diastolic_bp) / 3.0
    )

    features.update(
        {
            "shock_index": shock_index,
            "mean_arterial_pressure": mean_arterial_pressure,
            "pulse_pressure": systolic_bp - diastolic_bp,
            "modified_shock_index": heart_rate / (mean_arterial_pressure + 0.1),
            "temp_deviation": abs(features["temperature_c"] - 37.0),
            "oxygen_deficit": max(98.0 - features["spo2"], 0.0),
            "gcs_deficit": gcs_deficit,
            "cardiopulmonary_stress": (
                heart_rate * features["respiratory_rate"]
            )
            / 100.0,
            "neuro_hemodynamic_index": shock_index * (gcs_deficit + 1.0),
            "news_vital_score": calculate_news_subscore(features),
        }
    )

    return features


def prepare_onnx_input(
    vitals: Mapping[str, object],
    *,
    imputation_values: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Kembalikan tensor float32 dengan bentuk ``[1, 17]``."""

    feature_map = build_feature_map(vitals, imputation_values=imputation_values)
    return feature_map_to_onnx_input(feature_map)


def feature_map_to_onnx_input(feature_map: Mapping[str, float]) -> np.ndarray:
    """Susun feature map tervalidasi menjadi tensor sesuai urutan kontrak."""

    tensor = np.asarray(
        [[feature_map[name] for name in FEATURE_ORDER]],
        dtype=np.float32,
    )

    if tensor.shape != (1, len(FEATURE_ORDER)):
        raise TriagePreprocessingError(
            f"Bentuk input tidak valid: {tensor.shape}; diharapkan (1, 17)."
        )
    if not np.all(np.isfinite(tensor)):
        raise TriagePreprocessingError("Feature engineering menghasilkan nilai non-finite.")

    return tensor
