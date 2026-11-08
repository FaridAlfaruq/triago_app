import os
import time
import threading
from copy import deepcopy

# Zona IGD memakai kode kategori triase: red = Resusitasi, yellow = Darurat,
# green = Non-Darurat. Urutan ini juga urutan nomor bed pada denah U
# (sisi kiri, sisi bawah, sisi kanan).
ZONES = ("red", "yellow", "green")
DEFAULT_ZONE_BEDS = {"red": 3, "yellow": 6, "green": 3}
ZONE_ENV = {"red": "TRIAGO_BEDS_RESUSITASI", "yellow": "TRIAGO_BEDS_DARURAT",
            "green": "TRIAGO_BEDS_NONDARURAT"}

# Target waktu penanganan (menit) mengikuti kategori triase, bukan lokasi bed.
TARGET_MINUTES = {"red": 0, "yellow": 10, "green": 60}

# Zona yang dicoba bila zona utama penuh. Pasien kuning didahulukan ke
# Non-Darurat agar bed Resusitasi tetap tersedia untuk pasien merah; pasien
# hijau menunggu bila zonanya penuh.
ZONE_FALLBACK = {
    "red": ["red", "yellow", "green"],
    "yellow": ["yellow", "green", "red"],
    "green": ["green"],
}


def zone_bed_counts():
    """Jumlah bed per zona dari .env, dengan default 3 / 6 / 3."""
    counts = {}
    for zone in ZONES:
        raw = os.environ.get(ZONE_ENV[zone], "").strip()
        try:
            counts[zone] = int(raw) if raw else DEFAULT_ZONE_BEDS[zone]
        except ValueError:
            raise ValueError(f"{ZONE_ENV[zone]} harus berupa angka") from None
        if not 1 <= counts[zone] <= 50:
            raise ValueError(f"{ZONE_ENV[zone]} harus antara 1 dan 50")
    return counts


class BedManager:
    """Mengelola status ketersediaan dan alokasi bed IGD per zona."""

    def __init__(self, zone_beds=None):
        self.zone_beds = dict(zone_beds or zone_bed_counts())
        self.beds = {}
        self._lock = threading.Lock()  # aman diakses dari beberapa thread Flask/SocketIO sekaligus
        self._init_beds()

    def _init_beds(self):
        # Nomor bed berurutan mengikuti denah U: Resusitasi, Darurat, Non-Darurat.
        self.beds = {}
        number = 1
        for zone in ZONES:
            for _ in range(self.zone_beds[zone]):
                bed_id = f"{number:02d}"
                self.beds[bed_id] = self._create_empty_bed(bed_id, zone)
                number += 1

    def _create_empty_bed(self, bed_id, zone):
        return {
            "bed_id": bed_id,
            "zone": zone,
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

    def _find_available_bed(self, category):
        if category not in ZONE_FALLBACK:
            raise ValueError("triage_category harus red, yellow, atau green")
        for zone in ZONE_FALLBACK[category]:
            for bed_id, bed in self.beds.items():
                if bed["zone"] == zone and bed["status"] == "empty":
                    return bed_id
        return None

    def assign_patient_to_bed(self, data: dict, persist=None):
        category = data.get("triage_category")

        with self._lock:
            # Bed ditentukan sistem dari kategori triase; bed_id dari perangkat lama diabaikan.
            assigned_bed_id = self._find_available_bed(category)

            if assigned_bed_id:
                arrival_timestamp = int(time.time() * 1000)
                if persist is not None:
                    # Persist before occupying a bed. A failed MQTT database write
                    # can then be retried without allocating an extra bed.
                    persist({**data, "bed_id": assigned_bed_id,
                             "zone": self.beds[assigned_bed_id]["zone"],
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
        """Kembalikan salinan bed yang diperbarui, atau None bila bed tidak ada."""
        with self._lock:
            if bed_id not in self.beds:
                return None
            self.beds[bed_id].update({
                "patient_name": patient_name,
                "relative_name": relative_name,
                "patient_details": dict(patient_details or {}),
            })
            return deepcopy(self.beds[bed_id])

    def discharge_bed(self, bed_id):
        """Kosongkan kembali sebuah bed (mis. setelah pasien dipindah/selesai ditangani)."""
        with self._lock:
            if bed_id not in self.beds:
                return False
            self.beds[bed_id] = self._create_empty_bed(bed_id, self.beds[bed_id]["zone"])
        return True

    def reset_all_beds(self):
        """Kosongkan SEMUA bed sekaligus. Berguna untuk testing tanpa perlu restart Flask."""
        with self._lock:
            self._init_beds()
            return deepcopy(self.beds)

    def get_all_beds_status(self):
        # Salinan diambil di dalam lock agar jsonify tidak membaca dict yang
        # sedang diubah worker MQTT.
        with self._lock:
            return deepcopy(self.beds)

    def get_bed(self, bed_id):
        with self._lock:
            bed = self.beds.get(bed_id)
            return deepcopy(bed) if bed is not None else None
