# backend/bed_manager.py
import time

class BedManager:
    """Mengelola status ketersediaan dan alokasi bed berdasarkan zona klasifikasi."""
    
    def __init__ (self):
        # Struktur awal Bed sesuai zona di HTML (Zone A: A1-A6, Zone B: B1-B12, Zone C: C1-C6)[cite: 2]
        self.beds = {}
        self._init_beds()

    def _init_beds(self):
        # Inisialisasi Bed Zona A (Resusitasi - Target 0 mnt)[cite: 2]
        for i in range(1, 7):
            self.beds[f"A{i}"] = self._create_empty_bed(f"A{i}", "A", 0)

        # Inisialisasi Bed Zona B (Darurat - Target <= 10 mnt)[cite: 2]
        for i in range(1, 13):
            self.beds[f"B{i}"] = self._create_empty_bed(f"B{i}", "B", 10)

        # Inisialisasi Bed Zona C (Non-Darurat - Target <= 60 mnt)[cite: 2]
        for i in range(1, 7):
            self.beds[f"C{i}"] = self._create_empty_bed(f"C{i}", "C", 60)

    def _create_empty_bed(self, bed_id, zone, target_minutes):
        return {
            "bed_id": bed_id,
            "zone": zone,
            "target_minutes": target_minutes,
            "status": "empty",       # 'empty' atau occupied category ('red', 'yellow', 'green')
            "patient_name": None,
            "xgboost_score": None,
            "vitals": {},
            "arrival_timestamp": None
        }

    def assign_patient_to_bed(self, data: dict):
        category = data.get("triage_category")  # 'red', 'yellow', 'green'
        
        # Penentuan zona otomatis berdasarkan klasifikasi
        target_zone = "A" if category == "red" else ("B" if category == "yellow" else "C")
        
        # Cari bed kosong pertama di zona tujuan, jika penuh pakai bed_id acak/spesifik
        assigned_bed_id = data.get("bed_id")
        if not assigned_bed_id or assigned_bed_id not in self.beds:
            assigned_bed_id = self._find_available_bed(target_zone)

        if assigned_bed_id:
            self.beds[assigned_bed_id].update({
                "status": category,
                "patient_name": data.get("patient_name"),
                "xgboost_score": data.get("xgboost_score"),
                "vitals": data.get("vitals", {}),
                "arrival_timestamp": int(time.time() * 1000)  # Epoch ms untuk timer JS[cite: 2]
            })
            return assigned_bed_id
        return None

    def _find_available_bed(self, zone_prefix):
        for bed_id, bed_info in self.beds.items():
            if bed_info["zone"] == zone_prefix and bed_info["status"] == "empty":
                return bed_id
        return None

    def get_all_beds_status(self):
        return self.beds