import importlib.util
import json
from contextlib import closing
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from backend.bed_manager import BedManager
from backend.mqtt_receiver import MqttReceiver
from backend.sqlite_store import SQLiteStore
from backend.triage_service import TriageService
from service.triage_payload import validate_payload
from test_mqtt import config, envelope, payload


class PainPayloadTests(unittest.TestCase):
    def test_zero_and_legacy_unknown_are_distinct(self):
        self.assertEqual(validate_payload({**payload(), "pain_score": 0})["pain_score"], 0)
        self.assertIsNone(validate_payload({**payload(), "pain_score": None})["pain_score"])
        self.assertNotIn("pain_score", validate_payload(payload()))

    def test_invalid_scores_are_rejected(self):
        for value in (-1, 11, 1.5, True, "8", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_payload({**payload(), "pain_score": value})


class PainReceiverTests(unittest.TestCase):
    def test_pain_and_model_score_reach_bed_and_persistent_history(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "patients.sqlite3")
            beds, emit, client = BedManager(), Mock(), Mock()
            service = TriageService(beds, store.save_patient, emit)
            with patch("backend.mqtt_receiver.create_client", return_value=client):
                receiver = MqttReceiver(service.process, config(), Path(directory) / "inbox.sqlite3")
            event = envelope()
            event["data"].update(pain_score=8, xgboost_score=0.91, triage_category="yellow")
            message = SimpleNamespace(payload=json.dumps(event).encode(), mid=1, qos=1,
                                      retain=False, topic="triago/test/devices/raspi-01/triage")
            receiver._on_message(client, None, message)
            self.assertTrue(receiver.process_once())
            receiver._on_message(client, None, message)
            self.assertFalse(receiver.process_once())
            rows = store.get_patient_history()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["pain_score"], 8)
            self.assertAlmostEqual(rows[0]["xgboost_score"], 0.91)
            self.assertEqual(beds.beds["A1"]["pain_score"], 8)
            self.assertEqual(emit.call_args.args[1]["pain_score"], 8)


class PainStorageTests(unittest.TestCase):
    def test_existing_sqlite_schema_is_migrated_without_losing_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            with closing(sqlite3.connect(path)) as conn:
                with conn:
                    conn.execute("""CREATE TABLE patient_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, bed_id TEXT, zone TEXT,
                        status TEXT, patient_name TEXT, relative_name TEXT,
                        gcs_score INTEGER, xgboost_score REAL, heart_rate REAL,
                        spo2 REAL, temperature REAL, respiration_rate REAL,
                        systolic_bp INTEGER, diastolic_bp INTEGER, arrival_timestamp INTEGER)""")
                    conn.execute("INSERT INTO patient_data(bed_id, patient_name) VALUES ('A1', 'OLD DUMMY')")
            reopened = SQLiteStore(path)
            old = reopened.get_patient_history()[0]
            self.assertEqual(old["patient_name"], "OLD DUMMY")
            self.assertIsNone(old["pain_score"])
            reopened.save_patient({**payload(), "pain_score": 0, "arrival_timestamp": 100})
            self.assertEqual(reopened.get_patient_history()[0]["pain_score"], 0)
            self.assertEqual(len(reopened.get_patient_history()), 2)
            SQLiteStore(path)  # Repeated startup must be harmless.

    def test_mysql_insert_includes_pain_score_and_legacy_none(self):
        from backend import database
        connection = Mock()
        with patch.object(database, "_sqlite_store", None), patch.object(database, "get_connection", return_value=connection):
            self.assertTrue(database.save_patient({**payload(), "pain_score": 7}))
            sql, values = connection.cursor.return_value.execute.call_args.args
            columns = [name.strip() for name in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
            self.assertEqual(values[columns.index("pain_score")], 7)
            self.assertTrue(database.save_patient(payload()))
            self.assertIsNone(connection.cursor.return_value.execute.call_args.args[1][columns.index("pain_score")])


@unittest.skipUnless(importlib.util.find_spec("onnxruntime"), "Model runtime is optional on the backend")
class PainModelTests(unittest.TestCase):
    def test_real_model_uses_matching_eleven_feature_contract(self):
        from model.deployment_inference import TriageOnnxModel, TRIAGE_FEATURES, build_triage_input
        model = TriageOnnxModel()
        vitals = dict(temperature_c=36.5, spo2=98, respiratory_rate=16, heart_rate=75,
                      systolic_bp=120, diastolic_bp=80, gcs_total=15, pain_score=8)
        features = build_triage_input(vitals)
        self.assertEqual(features.shape, (1, 11))
        self.assertEqual(model.session.get_inputs()[0].shape[-1], len(TRIAGE_FEATURES))
        self.assertEqual(features[0, TRIAGE_FEATURES.index("pain_score")], 8)
        label, score, probabilities = model.predict(vitals)
        self.assertIn(label, {"RESUSITASI", "DARURAT", "NON-DARURAT"})
        self.assertTrue(0 <= score <= 1)
        self.assertEqual(len(probabilities), 3)

    def test_model_with_old_feature_width_fails_before_inference(self):
        from model.deployment_inference import TriageOnnxModel
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name="input", shape=[None, 21])]
        with patch("model.deployment_inference.rt.InferenceSession", return_value=session):
            with self.assertRaisesRegex(ValueError, "21 fitur"):
                TriageOnnxModel()
