import time
import threading

# Sama dengan tombol "Tata Letak Kasur" di GUI Raspi (GUI/regist_page.py): bed 01-12.
BED_COUNT = 12

# Target waktu penanganan (menit) mengikuti kategori triase, bukan lokasi bed.
TARGET_MINUTES = {"red": 0, "yellow": 10, "green": 60}


def normalize_bed_id(bed_id):
    """'1', '01', 'Bed 01' -> '01'. Nilai lain dikembalikan apa adanya."""
    text = str(bed_id or "").strip()
    digits = text[3:].strip() if text.lower().startswith("bed") else text
    return f"{int(digits):02d}" if digits.isdigit() else text


class BedManager:
    """Mengelola status ketersediaan dan alokasi bed IGD."""

    def __init__(self):
        self.beds = {}
        self._lock = threading.Lock()  # aman diakses dari beberapa thread Flask/SocketIO sekaligus
        self._init_beds()

    def _init_beds(self):
        self.beds = {f"{i:02d}": self._create_empty_bed(f"{i:02d}") for i in range(1, BED_COUNT + 1)}

    def _create_empty_bed(self, bed_id):
        return {
            "bed_id": bed_id,
            "target_minutes": None,
            "status": "empty",
            "patient_name": None,
            "relative_name": None,
            "patient_details": {},
            "gcs_score": None,
            "pain_score": None,
            "xgboost_score": None,
            "vitals": {},
            "arrival_timestamp": None,
        }

    def assign_patient_to_bed(self, data: dict, persist=None):
        category = data.get("triage_category")

        with self._lock:
            # Bed yang dipilih perawat di GUI Raspi diutamakan; bila tidak ada atau
            # sudah terisi, pakai bed kosong dengan nomor terkecil.
            assigned_bed_id = normalize_bed_id(data.get("bed_id"))
            if assigned_bed_id not in self.beds or self.beds[assigned_bed_id]["status"] != "empty":
                assigned_bed_id = next(
                    (bed_id for bed_id, bed in self.beds.items() if bed["status"] == "empty"), None)

            if assigned_bed_id:
                arrival_timestamp = int(time.time() * 1000)
                if persist is not None:
                    # Persist before occupying a bed. A failed MQTT database write
                    # can then be retried without allocating an extra bed.
                    persist({**data, "bed_id": assigned_bed_id,
                             "arrival_timestamp": arrival_timestamp})
                self.beds[assigned_bed_id].update({
                    "status": category,
                    "target_minutes": TARGET_MINUTES.get(category),
                    "gcs_score": data.get("gcs_score", 15),
                    "pain_score": data.get("pain_score"),
                    "patient_name": data.get("patient_name"),
                    "relative_name": data.get("relative_name"),
                    "xgboost_score": data.get("xgboost_score"),
                    "vitals": data.get("vitals", {}),
                    "arrival_timestamp": arrival_timestamp,
                })
                return assigned_bed_id
        return None

    def update_patient_identity(self, bed_id: str, patient_name: str, relative_name: str,
                                patient_details=None):
        with self._lock:
            if bed_id not in self.beds:
                return False
            self.beds[bed_id].update({
                "patient_name": patient_name,
                "relative_name": relative_name,
                "patient_details": dict(patient_details or {}),
            })
            return True

    def discharge_bed(self, bed_id):
        """Kosongkan kembali sebuah bed (mis. setelah pasien dipindah/selesai ditangani)."""
        with self._lock:
            if bed_id not in self.beds:
                return False
            self.beds[bed_id] = self._create_empty_bed(bed_id)
        return True

    def reset_all_beds(self):
        """Kosongkan SEMUA bed sekaligus. Berguna untuk testing tanpa perlu restart Flask."""
        with self._lock:
            self._init_beds()
        return self.beds

    def get_all_beds_status(self):
        return self.beds
