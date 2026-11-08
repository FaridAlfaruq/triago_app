import os
import sqlite3

import requests

from service.config import load_config
from service.triage_payload import validate_payload


class TriageApiClient:
    def __init__(self, base_url=None):
        load_config()
        self.transport = os.environ.get("TRIAGO_TRANSPORT", "http").lower()
        if self.transport not in {"http", "mqtt"}:
            raise ValueError("TRIAGO_TRANSPORT harus http atau mqtt")
        self.mqtt_publisher = None
        self.last_delivery_status = "not_sent"
        # Bed hasil alokasi backend untuk pengiriman terakhir (lihat get_assignment).
        self.last_event_id = None
        self.last_assignment = None
        self.base_url = (
            base_url
            or os.environ.get("TRIAGO_API_URL")
            or "http://127.0.0.1:5000"
        ).rstrip("/")
        self.endpoint_update = f"{self.base_url}/api/triage/update"
        self.timeout = 3.0
        if self.transport == "mqtt":
            try:
                # Resume an existing outbox when the GUI opens, before the next measurement.
                self._get_mqtt_publisher()
            except (ValueError, OSError, sqlite3.Error, ImportError) as exc:
                print(f"[MQTT CONFIG] Publisher belum siap: {exc}")

    def _get_mqtt_publisher(self):
        if self.mqtt_publisher is None:
            from service.mqtt_client import MqttPublisher
            self.mqtt_publisher = MqttPublisher()
        return self.mqtt_publisher

    def send_triage_result(self, gcs_score: int, vitals: dict, classification: str, score: float,
                           patient_name=None, relative_name=None, pain_score=None) -> bool:
        """Mengirimkan payload triage; bed dialokasikan backend sesuai kategori."""
        payload = {
            "gcs_score": gcs_score,
            "pain_score": pain_score,
            "triage_category": classification.lower(),  # 'red', 'yellow', 'green'
            "xgboost_score": round(float(score), 2),
            "vitals": {
                "hr": round(vitals.get("hr", 0), 1),
                "spo2": round(vitals.get("spo2", 0), 1),
                "rr": round(vitals.get("rr", 0), 1),
                "temp_core": round(vitals.get("temp_core", 0.0), 1), # Suhu Inti Estimasi
                "sys": int(vitals.get("sys", 0)),
                "dia": int(vitals.get("dia", 0))
            }
        }
        if patient_name is not None:
            payload["patient_name"] = patient_name
        if relative_name is not None:
            payload["relative_name"] = relative_name
        self.last_delivery_status = "not_sent"
        self.last_event_id = None
        self.last_assignment = None
        try:
            payload = validate_payload(payload)
            if self.transport == "mqtt":
                event_id = self._get_mqtt_publisher().enqueue(payload)
                self.last_event_id = event_id
                self.last_delivery_status = "queued"
                print(f"[MQTT QUEUED] Event {event_id} disimpan di Raspi; pengiriman berjalan di background.")
                return True
        except (ValueError, OSError, sqlite3.Error, ImportError) as exc:
            print(f"[TRANSPORT ERROR] {exc}; hasil belum dikirim/disimpan di antrean.")
            return False

        try:
            response = requests.post(self.endpoint_update, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                resp_data = response.json() if response.content else {}
                saved_db = resp_data.get("saved_to_database", False)
                self.last_delivery_status = "saved" if saved_db else "dashboard_only"
                self.last_assignment = {key: resp_data.get(key) for key in ("status", "assigned_bed", "zone")}
                bed_id = resp_data.get("assigned_bed") or "WAITING"
                if saved_db:
                    print(f"[API SUCCESS] Data Bed {bed_id} BERHASIL dikirim ke Backend & BERHASIL disimpan ke Database (phpMyAdmin).")
                else:
                    print(f"[API WARNING] Data Bed {bed_id} berhasil dikirim ke Web Dashboard, tetapi GAGAL disimpan ke Database MySQL.")
                return True
            else:
                err_msg = response.text if response.text else f"Status {response.status_code}"
                print(f"[API ERROR] Server menolak data (Status {response.status_code}): {err_msg}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"[API WARNING] Gagal terhubung ke Flask Server ({self.base_url}): {e}")
            return False

    def get_assignment(self):
        """Bed hasil alokasi untuk pengiriman terakhir, atau None bila belum ada balasan.

        HTTP mendapatkannya langsung dari respons; MQTT menunggu balasan backend
        di topic assignment perangkat ini."""
        if self.last_assignment is not None:
            return self.last_assignment
        if self.last_event_id and self.mqtt_publisher is not None:
            self.last_assignment = self.mqtt_publisher.get_assignment(self.last_event_id)
        return self.last_assignment

    def close(self):
        if self.mqtt_publisher is not None:
            self.mqtt_publisher.close()
