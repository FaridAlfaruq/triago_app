import time
import threading


class BedManager:
    """Mengelola status ketersediaan dan alokasi bed berdasarkan zona klasifikasi."""

    def __init__(self):
        self.beds = {}
        self._lock = threading.Lock()  # aman diakses dari beberapa thread Flask/SocketIO sekaligus
        self._init_beds()

    def _init_beds(self):
        # Zona A (Resusitasi / Kritis: 5 Bed A1-A5)
        for i in range(1, 6):
            self.beds[f"A{i}"] = self._create_empty_bed(f"A{i}", "A", 0)

        # Zona B (Darurat: 4 Bed B1-B4)
        for i in range(1, 5):
            self.beds[f"B{i}"] = self._create_empty_bed(f"B{i}", "B", 10)

        # Zona C (Non-Darurat: 3 Bed C1-C3)
        for i in range(1, 4):
            self.beds[f"C{i}"] = self._create_empty_bed(f"C{i}", "C", 60)

    def _create_empty_bed(self, bed_id, zone, target_minutes):
        return {
            "bed_id": bed_id,
            "zone": zone,
            "target_minutes": target_minutes,
            "status": "empty",
            "patient_name": None,
            "relative_name": None,
            "gcs_score": None,
            "xgboost_score": None,
            "vitals": {},
            "arrival_timestamp": None,
        }

    # Urutan zona yang dicoba per kategori triase.
    # Prioritas utama tetap zona sesuai kategori, tapi kalau penuh,
    # cascading ke zona lain daripada menolak pasien sama sekali.
    _ZONE_FALLBACK = {
        "red":    ["A", "B", "C"],
        "yellow": ["B", "A", "C"],
        "green":  ["C", "B", "A"],
    }

    def assign_patient_to_bed(self, data: dict):
        category = data.get("triage_category")
        zone_order = self._ZONE_FALLBACK.get(category, ["C", "B", "A"])

        with self._lock:
            assigned_bed_id = data.get("bed_id")
            if (
                not assigned_bed_id
                or assigned_bed_id not in self.beds
                or self.beds[assigned_bed_id]["status"] != "empty"
            ):
                assigned_bed_id = None
                for zone in zone_order:
                    assigned_bed_id = self._find_available_bed(zone)
                    if assigned_bed_id:
                        break

            if assigned_bed_id:
                self.beds[assigned_bed_id].update({
                    "status": category,
                    "gcs_score": data.get("gcs_score", 15),
                    "patient_name": data.get("patient_name"),
                    "relative_name": data.get("relative_name"),
                    "xgboost_score": data.get("xgboost_score"),
                    "vitals": data.get("vitals", {}),
                    "arrival_timestamp": int(time.time() * 1000),
                })
                return assigned_bed_id
        return None

    def _find_available_bed(self, zone_prefix):
        for bed_id, bed_info in self.beds.items():
            if bed_info["zone"] == zone_prefix and bed_info["status"] == "empty":
                return bed_id
        return None

    def update_patient_identity(self, bed_id: str, patient_name: str, relative_name: str):
        with self._lock:
            if bed_id not in self.beds:
                return False
            self.beds[bed_id].update({
                "patient_name": patient_name,
                "relative_name": relative_name
            })
            return True

    def discharge_bed(self, bed_id):
        """Kosongkan kembali sebuah bed (mis. setelah pasien dipindah/selesai ditangani)."""
        with self._lock:
            if bed_id not in self.beds:
                return False
            zone = self.beds[bed_id]["zone"]
            target_minutes = self.beds[bed_id]["target_minutes"]
            self.beds[bed_id] = self._create_empty_bed(bed_id, zone, target_minutes)
        return True

    def reset_all_beds(self):
        """Kosongkan SEMUA bed sekaligus. Berguna untuk testing tanpa perlu restart Flask."""
        with self._lock:
            self._init_beds()
        return self.beds

    def get_all_beds_status(self):
        """Mengembalikan status seluruh bed, di-hydrate otomatis dari database jika ada data pasien aktif."""
        with self._lock:
            try:
                try:
                    from backend.database import get_patient_history
                except (ImportError, ModuleNotFoundError):
                    from database import get_patient_history

                history = get_patient_history(limit=50)
                if history:
                    # Proses riwayat pasien terbaru dari database (urutan waktu terlama ke terbaru)
                    for row in reversed(history):
                        bed_id = row.get("bed_id")
                        status = row.get("status")
                        if bed_id and bed_id in self.beds and status and status != "empty":
                            self.beds[bed_id].update({
                                "status": status,
                                "gcs_score": row.get("gcs_score", 15),
                                "patient_name": row.get("patient_name"),
                                "relative_name": row.get("relative_name"),
                                "xgboost_score": row.get("xgboost_score"),
                                "vitals": {
                                    "hr": row.get("heart_rate"),
                                    "spo2": row.get("spo2"),
                                    "temp_core": row.get("temperature"),
                                    "rr": row.get("respiration_rate"),
                                    "sys": row.get("systolic_bp"),
                                    "dia": row.get("diastolic_bp"),
                                },
                                "arrival_timestamp": row.get("arrival_timestamp"),
                            })
            except Exception as e:
                print(f"[BED_MANAGER][WARN] Hydration database error: {e}")

            return self.beds