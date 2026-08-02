import os

import requests


class TriageApiClient:
    def __init__(self, base_url=None):
        self.base_url = (
            base_url
            or os.environ.get("TRIAGO_API_URL")
            or "https://triago-bmeits-dsc7btake6fhhxde.indonesiacentral-01.azurewebsites.net"
        ).rstrip("/")
        self.endpoint_update = f"{self.base_url}/api/triage/update"
        self.timeout = 15.0

    def send_triage_result(self, bed_id: str, gcs_score: int, vitals: dict, classification: str, score: float) -> bool:
        """Mengirimkan payload triage dengan Suhu Inti dan Skor GCS."""
        payload = {
            "bed_id": bed_id,
            "gcs_score": gcs_score,
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

        try:
            response = requests.post(self.endpoint_update, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                resp_data = response.json() if response.content else {}
                saved_db = resp_data.get("saved_to_database", False)
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
