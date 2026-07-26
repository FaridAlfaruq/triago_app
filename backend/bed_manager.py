import time

class BedManager:
    """Mengelola status ketersediaan dan alokasi bed berdasarkan zona klasifikasi."""
    
    def __init__(self):
        self.beds = {}
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
            "xgboost_score": None,
            "vitals": {},
            "arrival_timestamp": None
        }

    def assign_patient_to_bed(self, data: dict):
        category = data.get("triage_category")
        target_zone = "A" if category == "red" else ("B" if category == "yellow" else "C")
        
        assigned_bed_id = data.get("bed_id")
        if not assigned_bed_id or assigned_bed_id not in self.beds or self.beds[assigned_bed_id]["status"] != "empty":
            assigned_bed_id = self._find_available_bed(target_zone)

        if assigned_bed_id:
            self.beds[assigned_bed_id].update({
                "status": category,
                "gcs_score": data.get("gcs_score", 15),
                "patient_name": None,
                "relative_name": None,
                "xgboost_score": data.get("xgboost_score"),
                "vitals": data.get("vitals", {}),
                "arrival_timestamp": int(time.time() * 1000)
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