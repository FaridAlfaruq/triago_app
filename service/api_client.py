import requests

class TriageApiClient:
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url
        self.endpoint_update = f"{self.base_url}/api/triage/update"
        self.timeout = 3.0

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
                print(f"[API SUCCESS] Data Bed {bed_id} berhasil dikirim ke Dashboard Web.")
                return True
            else:
                print(f"[API ERROR] Server menolak data: Status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"[API WARNING] Gagal terhubung ke Flask Server ({self.base_url}): {e}")
            return False