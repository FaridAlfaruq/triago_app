import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.bed_manager import BedManager, normalize_bed_id
from backend.patient_details import validate_patient_details
from backend.sqlite_store import SQLiteStore
from backend.triage_service import TriageService


def measurement(bed_id="05", category="yellow"):
    return {"bed_id": bed_id, "triage_category": category, "gcs_score": 14, "xgboost_score": 0.5,
            "vitals": {"hr": 100, "spo2": 95, "rr": 20, "temp_core": 37.5, "sys": 130, "dia": 85}}


class BedNumberingTests(unittest.TestCase):
    def test_beds_match_raspi_gui_numbering(self):
        self.assertEqual(list(BedManager().beds), [f"{i:02d}" for i in range(1, 13)])

    def test_bed_id_normalization(self):
        self.assertEqual(normalize_bed_id("1"), "01")
        self.assertEqual(normalize_bed_id("01"), "01")
        self.assertEqual(normalize_bed_id("Bed 7"), "07")
        self.assertEqual(normalize_bed_id("A1"), "A1")

    def test_bed_chosen_on_raspi_is_used(self):
        beds = BedManager()
        self.assertEqual(beds.assign_patient_to_bed(measurement("7")), "07")
        self.assertEqual(beds.beds["07"]["target_minutes"], 10)

    def test_occupied_or_unknown_bed_falls_back_to_lowest_empty(self):
        beds = BedManager()
        beds.assign_patient_to_bed(measurement("01"))
        self.assertEqual(beds.assign_patient_to_bed(measurement("01")), "02")
        self.assertEqual(beds.assign_patient_to_bed(measurement("A1")), "03")

    def test_full_igd_returns_none(self):
        beds = BedManager()
        for _ in range(12):
            self.assertIsNotNone(beds.assign_patient_to_bed(measurement("00")))
        self.assertIsNone(beds.assign_patient_to_bed(measurement("00")))

    def test_target_follows_category_and_discharge_clears_details(self):
        beds = BedManager()
        bed_id = beds.assign_patient_to_bed(measurement("03", "green"))
        self.assertEqual(beds.beds[bed_id]["target_minutes"], 60)
        beds.update_patient_identity(bed_id, "Budi", "Ani", {"nik": "3578010101900001"})
        self.assertTrue(beds.discharge_bed(bed_id))
        self.assertEqual(beds.beds[bed_id]["status"], "empty")
        self.assertEqual(beds.beds[bed_id]["patient_details"], {})


class PatientDetailsTests(unittest.TestCase):
    def test_valid_details_are_trimmed_and_empty_fields_dropped(self):
        clean = validate_patient_details({
            "nik": "3578010101900001", "birth_date": "1990-01-01", "sex": "L",
            "payer": "bpjs", "phone": "0812-3456-7890", "chief_complaint": "  Sesak napas  ",
            "allergies": "", "address": None,
        })
        self.assertEqual(clean["chief_complaint"], "Sesak napas")
        self.assertNotIn("allergies", clean)
        self.assertNotIn("address", clean)

    def test_invalid_details_are_rejected(self):
        for details in ({"nik": "123"}, {"sex": "X"}, {"birth_date": "2999-01-01"},
                        {"birth_date": "01-01-1990"}, {"phone": "abc"}, {"unknown": "x"},
                        {"chief_complaint": "x" * 501}, {"nik": 3578010101900001}, []):
            with self.assertRaises(ValueError, msg=details):
                validate_patient_details(details)

    def test_sqlite_stores_details_on_latest_row_and_migrates_old_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "patients.sqlite3")
            store.save_patient({**measurement("04"), "arrival_timestamp": 1})
            store.save_patient({**measurement("04"), "arrival_timestamp": 2})
            self.assertTrue(store.update_patient_info("04", "Budi", "Ani", {"payer": "bpjs"}))
            rows = store.get_patient_history()
            self.assertEqual(rows[0]["patient_details"], '{"payer": "bpjs"}')
            self.assertIsNone(rows[1]["patient_details"])


class PatientEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"TRIAGO_MQTT_ENABLED": "0", "TRIAGO_DB_BACKEND": "sqlite",
                                     "TRIAGO_SQLITE_PATH": os.path.join(tempfile.mkdtemp(), "db.sqlite3"),
                                     "TRIAGO_REQUIRE_AUTH": "0", "TRIAGO_DASHBOARD_USER": "",
                                     "TRIAGO_DASHBOARD_PASSWORD": ""}):
            from backend import app as module
        cls.backend = module

    def setUp(self):
        self.beds = BedManager()
        self.update = Mock(return_value=True)
        for name, value in (("bed_manager", self.beds), ("update_patient_info", self.update),
                            ("triage_service", TriageService(self.beds, Mock(return_value=True), Mock()))):
            patcher = patch.object(self.backend, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = self.backend.app.test_client()
        self.client.post("/api/triage/update", json=measurement("06"))

    def test_details_reach_bed_state_and_database(self):
        response = self.client.post("/api/beds/06/patient", json={
            "patient_name": "Budi", "relative_name": "Ani",
            "details": {"sex": "L", "payer": "umum", "allergies": ""}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["data"]["patient_details"], {"sex": "L", "payer": "umum"})
        self.update.assert_called_once_with("06", "Budi", "Ani", {"sex": "L", "payer": "umum"})

    def test_invalid_details_return_400_without_changes(self):
        response = self.client.post("/api/beds/06/patient", json={"details": {"nik": "12"}})
        self.assertEqual(response.status_code, 400)
        self.assertIn("NIK", response.json["message"])
        self.update.assert_not_called()
        self.assertEqual(self.beds.beds["06"]["patient_details"], {})


if __name__ == "__main__":
    unittest.main()
