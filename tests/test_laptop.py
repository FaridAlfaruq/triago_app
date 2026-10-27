import base64
from contextlib import closing
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from flask import Flask
from flask_socketio import SocketIO
from werkzeug.test import Client
from werkzeug.wrappers import Response

from backend.access_control import DashboardLogin, dashboard_credentials
from backend.bed_manager import BedManager
from backend.sqlite_store import SQLiteStore
from backend.triage_service import TriageService
from service import laptop
from service.mqtt_client import MqttConfig


AUTH = "Basic " + base64.b64encode(b"tester:a-long-test-password").decode()
TEST_ENV = {"TRIAGO_MQTT_HOST": "cluster.example", "TRIAGO_MQTT_USERNAME": "demo",
            "TRIAGO_MQTT_PASSWORD": "test-password", "TRIAGO_REQUIRE_AUTH": "1",
            "TRIAGO_DASHBOARD_USER": "tester", "TRIAGO_DASHBOARD_PASSWORD": "a-long-test-password",
            "TRIAGO_MQTT_ENABLED": "1", "TRIAGO_DB_BACKEND": "sqlite"}


def measurement():
    return {"triage_category": "green", "gcs_score": 15, "xgboost_score": 0.2,
            "patient_name": "SIMULASI", "vitals": {"hr": 75, "spo2": 98, "rr": 16,
                                                    "temp_core": 36.5, "sys": 120, "dia": 80}}


class DashboardAccessTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        self.socketio = SocketIO(app, async_mode="threading")
        self.addCleanup(self.socketio.server.eio.shutdown)
        app.add_url_rule("/", endpoint="index", view_func=lambda: "dashboard")
        app.add_url_rule("/api/history", endpoint="history", view_func=lambda: "[]")
        app.add_url_rule("/api/beds/reset", endpoint="reset", view_func=lambda: "reset", methods=["POST"])
        app.wsgi_app = DashboardLogin(app.wsgi_app, "tester", "a-long-test-password")
        self.client = Client(app, Response)

    def test_dashboard_api_and_socket_handshake_require_login(self):
        for path in ("/", "/api/history", "/socket.io/?EIO=4&transport=polling"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 401)
                self.assertIn("Basic", response.headers["WWW-Authenticate"])
        self.assertEqual(self.client.post("/api/beds/reset").status_code, 401)

    def test_correct_login_works_and_private_pages_are_not_cached(self):
        response = self.client.get("/", headers={"Authorization": AUTH})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(self.client.get("/api/history", headers={"Authorization": AUTH}).status_code, 200)

    def test_real_engine_io_handshake_requires_and_accepts_credentials(self):
        response = self.client.get("/socket.io/?EIO=4&transport=polling",
                                   headers={"Authorization": AUTH, "Origin": "http://localhost"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data.startswith(b"0{"))
        sid = json.loads(response.data[1:])["sid"]
        self.client.post(f"/socket.io/?EIO=4&transport=polling&sid={sid}", data="1",
                         headers={"Authorization": AUTH})

    def test_wrong_password_and_cross_origin_mutation_are_rejected(self):
        self.assertEqual(self.client.get("/", headers={"Authorization": "Basic d3Jvbmc6d3Jvbmc="}).status_code, 401)
        self.assertEqual(self.client.post("/api/beds/reset", headers={
            "Authorization": AUTH, "Origin": "https://other.example"}).status_code, 403)

    def test_incomplete_login_configuration_fails_closed(self):
        with patch.dict(os.environ, {"TRIAGO_REQUIRE_AUTH": "1"}, clear=True):
            with self.assertRaises(ValueError):
                dashboard_credentials()


class LaptopDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "patients.sqlite3"
        self.store = SQLiteStore(self.path)

    def test_triage_persists_real_measurement_and_emits_dashboard_event(self):
        emit = Mock()
        service = TriageService(BedManager(), self.store.save_patient, emit)
        result = service.process(measurement(), event_id="test-event", require_database=True)
        self.assertTrue(result["saved_to_database"])
        emit.assert_called_once()
        rows = SQLiteStore(self.path).get_patient_history()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["heart_rate"], 75)
        self.assertEqual(rows[0]["bed_id"], "10")  # bed pertama zona Non-Darurat
        self.assertEqual(rows[0]["zone"], "green")
        self.assertEqual(rows[0]["status"], "green")
        service.process(measurement(), event_id="test-event", require_database=True)
        self.assertEqual(len(self.store.get_patient_history()), 1)

    def test_identity_edit_updates_only_latest_measurement_for_bed(self):
        self.store.save_patient({**measurement(), "bed_id": "01", "arrival_timestamp": 1})
        self.store.save_patient({**measurement(), "bed_id": "01", "arrival_timestamp": 2})
        self.assertTrue(self.store.update_patient_info("01", "Nama Baru", "Wali"))
        rows = self.store.get_patient_history()
        self.assertEqual(rows[0]["patient_name"], "Nama Baru")
        self.assertEqual(rows[1]["patient_name"], "SIMULASI")
        self.assertFalse(self.store.update_patient_info("MISSING", "Nama", "Wali"))


class LaptopWorkflowTests(unittest.TestCase):
    def test_roles_have_different_ids_even_with_one_shared_env(self):
        with (patch.dict(os.environ, {**TEST_ENV,
                                     "TRIAGO_MQTT_PUBLISHER_CLIENT_ID": "laptop-pub",
                                     "TRIAGO_MQTT_BACKEND_CLIENT_ID": "laptop-sub"}, clear=True),
              patch("service.mqtt_client.load_config"), patch("service.laptop.load_config")):
            laptop.check_configuration()
            self.assertNotEqual(MqttConfig.from_env("publisher").client_id, MqttConfig.from_env("backend").client_id)

    def test_duplicate_client_ids_and_placeholder_credentials_are_rejected(self):
        with (patch.dict(os.environ, {**TEST_ENV, "TRIAGO_MQTT_CLIENT_ID": "same"}, clear=True),
              patch("service.mqtt_client.load_config"), patch("service.laptop.load_config")):
            with self.assertRaisesRegex(ValueError, "Client ID"):
                laptop.check_configuration()
        with (patch.dict(os.environ, {**TEST_ENV, "TRIAGO_MQTT_HOST": "your-cluster.example"}, clear=True),
              patch("service.mqtt_client.load_config"), patch("service.laptop.load_config")):
            with self.assertRaisesRegex(ValueError, "TRIAGO_MQTT_HOST"):
                laptop.check_configuration()

    def test_init_generates_secret_and_preserves_existing_env(self):
        template = (laptop.PROJECT_ROOT / "config/laptop.env.example").read_text()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config/laptop.env.example").write_text(template)
            with patch.object(laptop, "PROJECT_ROOT", root):
                self.assertEqual(laptop.initialize(), 0)
                first = (root / ".env").read_text()
                self.assertNotIn("replace-with-generated-dashboard-password", first)
                self.assertEqual(laptop.initialize(), 1)
                self.assertEqual((root / ".env").read_text(), first)

    def test_share_refuses_backend_without_authentication_before_starting_tunnel(self):
        with (patch("service.laptop.check_configuration", return_value=("tester", "password")),
              patch("service.laptop.urlopen", return_value=closing(io.BytesIO(b"{}"))),
              patch("service.laptop.subprocess.call") as run):
            with self.assertRaisesRegex(ValueError, "tanpa login"):
                laptop.share()
            run.assert_not_called()

    def test_share_checks_actual_backend_login_then_starts_only_loopback_tunnel(self):
        health = {"access": {"login_required": True}, "mqtt": {"subscribed": True}}
        unauthorized = HTTPError("http://127.0.0.1:5000/api/health", 401, "Unauthorized", {}, None)
        with (patch("service.laptop.check_configuration", return_value=("tester", "password")),
              patch("service.laptop.urlopen", side_effect=[unauthorized, closing(io.BytesIO(json.dumps(health).encode()))]),
              patch("service.laptop.tunnel_binary", return_value="cloudflared"),
              patch("service.laptop.subprocess.call", return_value=0) as run):
            self.assertEqual(laptop.share(), 0)
            command = run.call_args.args[0]
            self.assertEqual(command[:3], ["cloudflared", "tunnel", "--url"])
            self.assertTrue(command[3].startswith("http://127.0.0.1:"))
            self.assertNotIn("password", str(command))


if __name__ == "__main__":
    unittest.main()
