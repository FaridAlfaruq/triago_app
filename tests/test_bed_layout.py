import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.bed_manager import BedManager, zone_bed_counts
from backend.patient_details import validate_patient_details
from backend.sqlite_store import SQLiteStore
from backend.triage_service import TriageService


def measurement(category="yellow"):
    return {"triage_category": category, "gcs_score": 14, "xgboost_score": 0.5,
            "vitals": {"hr": 100, "spo2": 95, "rr": 20, "temp_core": 37.5, "sys": 130, "dia": 85}}


def fill(beds, category, count):
    return [beds.assign_patient_to_bed(measurement(category)) for _ in range(count)]


class BedZoneTests(unittest.TestCase):
    def test_default_beds_are_numbered_along_the_u_by_zone(self):
        beds = BedManager()
        self.assertEqual(list(beds.beds), [f"{i:02d}" for i in range(1, 13)])
        zones = [bed["zone"] for bed in beds.beds.values()]
        self.assertEqual(zones, ["red"] * 3 + ["yellow"] * 6 + ["green"] * 3)

    def test_bed_counts_come_from_env(self):
        with patch.dict(os.environ, {"TRIAGO_BEDS_RESUSITASI": "2", "TRIAGO_BEDS_DARURAT": "8",
                                     "TRIAGO_BEDS_NONDARURAT": "4"}):
            beds = BedManager()
        self.assertEqual(len(beds.beds), 14)
        self.assertEqual(beds.beds["03"]["zone"], "yellow")
        self.assertEqual(beds.beds["11"]["zone"], "green")
        for invalid in ("0", "abc", "51"):
            with patch.dict(os.environ, {"TRIAGO_BEDS_DARURAT": invalid}), self.assertRaises(ValueError):
                zone_bed_counts()

    def test_category_goes_to_its_own_zone(self):
        beds = BedManager()
        self.assertEqual(beds.assign_patient_to_bed(measurement("red")), "01")
        self.assertEqual(beds.assign_patient_to_bed(measurement("yellow")), "04")
        self.assertEqual(beds.assign_patient_to_bed(measurement("green")), "10")
        self.assertEqual(beds.beds["04"]["target_minutes"], 10)

    def test_full_red_zone_overflows_to_darurat_then_non_darurat(self):
        beds = BedManager()
        fill(beds, "red", 3)
        self.assertEqual(beds.assign_patient_to_bed(measurement("red")), "04")
        fill(beds, "yellow", 5)
        self.assertEqual(beds.assign_patient_to_bed(measurement("red")), "10")
        self.assertEqual(beds.beds["10"]["status"], "red")
        self.assertEqual(beds.beds["10"]["zone"], "green")

    def test_full_yellow_zone_uses_non_darurat_before_resusitasi(self):
        beds = BedManager()
        fill(beds, "yellow", 6)
        self.assertEqual(beds.assign_patient_to_bed(measurement("yellow")), "10")
        self.assertEqual(beds.beds["10"]["status"], "yellow")
        fill(beds, "yellow", 2)
        self.assertEqual(beds.beds["01"]["status"], "empty")  # Resusitasi disisakan
        self.assertEqual(beds.assign_patient_to_bed(measurement("yellow")), "01")
        fill(beds, "yellow", 2)
        self.assertIsNone(beds.assign_patient_to_bed(measurement("yellow")))

    def test_unknown_category_is_rejected(self):
        with self.assertRaises(ValueError):
            BedManager().assign_patient_to_bed(measurement("RED"))

    def test_status_snapshot_is_a_copy(self):
        beds = BedManager()
        snapshot = beds.get_all_beds_status()
        beds.assign_patient_to_bed(measurement("red"))
        self.assertEqual(snapshot["01"]["status"], "empty")
        snapshot["02"]["status"] = "red"
        self.assertEqual(beds.beds["02"]["status"], "empty")

    def test_full_green_zone_waits_instead_of_taking_other_zones(self):
        beds = BedManager()
        fill(beds, "green", 3)
        self.assertIsNone(beds.assign_patient_to_bed(measurement("green")))
        self.assertEqual(beds.beds["01"]["status"], "empty")

    def test_discharge_keeps_zone_and_clears_details(self):
        beds = BedManager()
        bed_id = beds.assign_patient_to_bed(measurement("green"))
        self.assertEqual(beds.beds[bed_id]["target_minutes"], 60)
        beds.update_patient_identity(bed_id, "Budi", "Ani", {"nik": "3578010101900001"})
        self.assertTrue(beds.discharge_bed(bed_id))
        self.assertEqual(beds.beds[bed_id]["status"], "empty")
        self.assertEqual(beds.beds[bed_id]["zone"], "green")
        self.assertEqual(beds.beds[bed_id]["patient_details"], {})

    def test_persisted_row_records_zone(self):
        save = Mock(return_value=True)
        result = TriageService(BedManager(), save, Mock()).process(measurement("red"))
        self.assertEqual((result["assigned_bed"], result["zone"]), ("01", "red"))
        self.assertEqual(save.call_args.args[0]["zone"], "red")


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

    def test_sqlite_stores_details_on_current_patient_row(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "patients.sqlite3")
            store.save_patient({**measurement(), "bed_id": "04", "arrival_timestamp": 1})
            store.save_patient({**measurement(), "bed_id": "04", "arrival_timestamp": 2})
            self.assertTrue(store.update_patient_info("04", 2, "Budi", "Ani", {"payer": "bpjs"}))
            rows = store.get_patient_history()
            self.assertEqual(rows[0]["patient_details"], '{"payer": "bpjs"}')
            self.assertIsNone(rows[1]["patient_details"])

    def test_identity_never_overwrites_previous_patient_row(self):
        # Pasien sebelumnya tersimpan; pasien sekarang (arrival 2) gagal tersimpan.
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "patients.sqlite3")
            store.save_patient({**measurement(), "bed_id": "04", "arrival_timestamp": 1})
            self.assertFalse(store.update_patient_info("04", 2, "Budi", "Ani", {"payer": "bpjs"}))
            self.assertNotEqual(store.get_patient_history()[0]["patient_name"], "Budi")


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
        self.client.post("/api/triage/update", json=measurement())  # bed 04, zona Darurat

    def test_details_reach_bed_state_and_database(self):
        response = self.client.post("/api/beds/04/patient", json={
            "patient_name": "Budi", "relative_name": "Ani",
            "details": {"sex": "L", "payer": "umum", "allergies": ""}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["data"]["patient_details"], {"sex": "L", "payer": "umum"})
        self.update.assert_called_once_with("04", self.beds.beds["04"]["arrival_timestamp"],
                                            "Budi", "Ani", {"sex": "L", "payer": "umum"})

    def test_non_text_names_return_400(self):
        for body in ({"patient_name": 123}, {"relative_name": ["x"]}, ["x"]):
            response = self.client.post("/api/beds/04/patient", json=body)
            self.assertEqual(response.status_code, 400, body)
        self.update.assert_not_called()

    def test_invalid_details_return_400_without_changes(self):
        response = self.client.post("/api/beds/04/patient", json={"details": {"nik": "12"}})
        self.assertEqual(response.status_code, 400)
        self.assertIn("NIK", response.json["message"])
        self.update.assert_not_called()
        self.assertEqual(self.beds.beds["04"]["patient_details"], {})


if __name__ == "__main__":
    unittest.main()
