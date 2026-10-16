"""Transport-neutral wire contract. No signal processing or model inference."""
import math
from uuid import UUID


def is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_payload(data):
    if not isinstance(data, dict):
        raise ValueError("Payload triage harus berupa object JSON")
    data = dict(data)
    if not isinstance(data.get("bed_id"), str) or not data["bed_id"].strip():
        raise ValueError("bed_id harus berupa string yang tidak kosong")
    category = data.get("triage_category")
    if not isinstance(category, str) or category.lower() not in {"red", "yellow", "green"}:
        raise ValueError("triage_category harus red, yellow, atau green")
    data["triage_category"] = category.lower()
    vitals = data.get("vitals")
    if not isinstance(vitals, dict):
        raise ValueError("vitals harus berupa object JSON")
    for key in ("hr", "spo2", "rr", "temp_core", "sys", "dia"):
        value = vitals.get(key)
        if not is_finite_number(value):
            raise ValueError(f"vitals.{key} harus berupa angka finite")
    for key in ("gcs_score", "xgboost_score"):
        value = data.get(key)
        if not is_finite_number(value):
            raise ValueError(f"{key} harus berupa angka finite")
    for key in ("patient_name", "relative_name"):
        if key in data and data[key] is not None and not isinstance(data[key], str):
            raise ValueError(f"{key} harus berupa string")
    return data


def validate_envelope(envelope):
    if (not isinstance(envelope, dict) or type(envelope.get("schema_version")) is not int
            or envelope["schema_version"] != 1):
        raise ValueError("schema_version MQTT tidak didukung")
    event_id = envelope.get("event_id")
    if not isinstance(event_id, str) or str(UUID(event_id)) != event_id:
        raise ValueError("event_id harus UUID canonical")
    device_id = envelope.get("device_id")
    if not isinstance(device_id, str) or not device_id or any(c in device_id for c in "/+#\x00"):
        raise ValueError("device_id MQTT tidak valid")
    validate_payload(envelope.get("data"))
    return envelope
