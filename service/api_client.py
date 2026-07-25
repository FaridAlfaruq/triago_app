# services/api_client.py
import requests
import logging

class TriageApiClient:
    """Modul khusus pengirim payload data vital sign & hasil klasifikasi ke Flask API."""
    
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url
        self.endpoint_update = f"{self.base_url}/api/triage/update"
        self.timeout = 3.0  # Waktu tunggu maksimal 3 detik agar GUI tidak menggantung

    def send_triage_result(self, bed_id: str, patient_info: dict, vitals: dict, classification: str, score: float) -> bool:
        """
        Mengirimkan payload lengkap ke dashboard IGD.
        classification: 'red' (Resusitasi), 'yellow' (Darurat), atau 'green' (Non-Darurat)
        """
        payload = {
            "bed_id": bed_id,
            "patient_name": patient_info.get("nama", "Pasien Tanpa Nama"),
            "triage_category": classification.lower(),  # 'red', 'yellow', 'green'
            "xgboost_score": round(float(score), 2),
            "vitals": {
                "hr": vitals.get("hr", 0),
                "spo2": vitals.get("spo2", 0),
                "rr": vitals.get("rr", 0),
                "temp_skin": round(vitals.get("temp_skin", 0.0), 2),
                "temp_ambient": round(vitals.get("temp_ambient", 0.0), 2),
                "sys": vitals.get("sys", 0),
                "dia": vitals.get("dia", 0)
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