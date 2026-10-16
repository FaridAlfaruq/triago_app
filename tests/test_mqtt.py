import json
import os
from pathlib import Path
import sqlite3
import ssl
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import paho.mqtt.client as mqtt

from backend.bed_manager import BedManager
from backend.mqtt_receiver import MqttReceiver
from backend.triage_service import DatabaseUnavailable, TriageService
from service.api_client import TriageApiClient
from service.mqtt_client import MqttConfig, MqttPublisher, create_client
from service.mqtt_queue import MessageQueue
from service.triage_payload import validate_envelope, validate_payload


def payload():
    return {"bed_id": "A1", "gcs_score": 15, "triage_category": "green",
            "xgboost_score": 0.2,
            "vitals": {"hr": 75, "spo2": 98, "rr": 16, "temp_core": 36.5, "sys": 120, "dia": 80}}


def envelope():
    return {"schema_version": 1, "event_id": str(uuid4()),
            "device_id": "raspi-01", "data": payload()}


def config():
    return MqttConfig("example.hivemq.cloud", 8883, "user", "password",
                      "triago/test", "raspi-01", "triago-test")


class MqttFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "queue.sqlite3"
        self.save = Mock(return_value=True)
        self.emit = Mock()
        self.beds = BedManager()
        self.service = TriageService(self.beds, self.save, self.emit)
        self.client = Mock()

    def receiver(self):
        with patch("backend.mqtt_receiver.create_client", return_value=self.client):
            return MqttReceiver(self.service.process, config(), self.path)

    def deliver(self, receiver, event, *, topic=None, retained=False):
        message = SimpleNamespace(payload=json.dumps(event).encode(), mid=42, qos=1,
                                  topic=topic or "triago/test/devices/raspi-01/triage", retain=retained)
        receiver._on_message(self.client, None, message)

    def ready_for_retry(self):
        with MessageQueue(self.path).connection() as conn:
            conn.execute("UPDATE messages SET next_attempt = 0")

    def occupied_count(self):
        return sum(bed["status"] != "empty" for bed in self.beds.get_all_beds_status().values())

    def test_mqtt_message_updates_database_and_dashboard_once(self):
        receiver = self.receiver()
        event = envelope()
        self.deliver(receiver, event)
        self.assertEqual(receiver.queue.pending_count(), 1)
        self.save.assert_not_called()  # MQTT callback only persists the inbox.
        self.client.ack.assert_called_once_with(42, 1)
        self.assertTrue(receiver.process_once())
        self.deliver(receiver, event)  # QoS 1 redelivery, even after completion.
        self.assertFalse(receiver.process_once())
        self.save.assert_called_once()
        self.emit.assert_called_once()
        self.assertEqual(self.occupied_count(), 1)
        self.assertEqual(receiver.queue.pending_count(), 0)

    def test_mysql_failure_leaves_message_pending_and_bed_empty(self):
        receiver = self.receiver()
        self.deliver(receiver, envelope())
        self.save.return_value = False
        self.assertFalse(receiver.process_once())
        self.assertEqual(self.occupied_count(), 0)
        self.emit.assert_not_called()
        self.assertEqual(receiver.queue.pending_count(), 1)
        self.save.return_value = True
        self.ready_for_retry()
        self.assertTrue(receiver.process_once())
        self.assertEqual(self.occupied_count(), 1)
        self.emit.assert_called_once()

    def test_inbox_survives_restart_and_deduplicates_completed_events(self):
        receiver = self.receiver()
        event = envelope()
        self.deliver(receiver, event)
        restarted = self.receiver()
        self.assertTrue(restarted.process_once())
        restarted_again = self.receiver()
        self.deliver(restarted_again, event)
        self.assertFalse(restarted_again.process_once())
        self.save.assert_called_once()

    def test_storage_failure_is_not_acked(self):
        receiver = self.receiver()
        with patch.object(receiver.queue, "put", side_effect=sqlite3.OperationalError("disk full")):
            self.deliver(receiver, envelope())
        self.client.ack.assert_not_called()
        self.assertTrue(receiver.reconnect_requested.is_set())

    def test_invalid_and_retained_messages_do_not_touch_beds_or_database(self):
        receiver = self.receiver()
        self.deliver(receiver, envelope(), topic="triago/test/devices/wrong/triage")
        self.deliver(receiver, envelope(), retained=True)
        bad = envelope()
        bad["data"]["vitals"]["hr"] = float("nan")
        self.deliver(receiver, bad)
        self.assertEqual(receiver.queue.pending_count(), 0)
        self.assertEqual(self.occupied_count(), 0)
        self.save.assert_not_called()
        self.assertEqual(self.client.ack.call_count, 3)

    def test_full_beds_preserve_waiting_patient(self):
        for _ in range(12):
            self.service.process(payload())
        self.save.reset_mock()
        receiver = self.receiver()
        self.deliver(receiver, envelope())
        self.assertTrue(receiver.process_once())
        self.assertEqual(self.save.call_args.args[0]["bed_id"], "WAITING")
        self.assertEqual(self.occupied_count(), 12)

    def test_retry_after_inbox_completion_failure_does_not_insert_twice(self):
        receiver = self.receiver()
        self.deliver(receiver, envelope())
        with patch.object(receiver.queue, "complete", side_effect=sqlite3.OperationalError("busy")):
            self.assertFalse(receiver.process_once())
        self.ready_for_retry()
        self.assertTrue(receiver.process_once())
        self.save.assert_called_once()
        self.assertEqual(self.occupied_count(), 1)

    def test_websocket_failure_does_not_retry_database_insert(self):
        receiver = self.receiver()
        self.emit.side_effect = RuntimeError("socket unavailable")
        self.deliver(receiver, envelope())
        self.assertTrue(receiver.process_once())
        self.save.assert_called_once()
        self.assertEqual(receiver.queue.pending_count(), 0)

    def test_legacy_http_keeps_dashboard_only_behavior_on_db_failure(self):
        self.save.return_value = False
        result = self.service.process(payload())
        self.assertFalse(result["saved_to_database"])
        self.assertEqual(self.occupied_count(), 1)
        self.emit.assert_called_once()

    def test_event_id_with_changed_contents_is_rejected(self):
        queue = MessageQueue(self.path)
        event = envelope()
        queue.put(event)
        event["data"]["vitals"]["hr"] = 80
        with self.assertRaises(ValueError):
            queue.put(event)
        self.assertEqual(queue.pending_count(), 1)

    def test_publisher_preserves_outbox_when_offline_or_puback_missing(self):
        with (patch("service.mqtt_client.create_client", return_value=self.client),
              patch("service.mqtt_client.threading.Thread")):
            publisher = MqttPublisher(config(), self.path)
            self.addCleanup(publisher.close)
        publisher.enqueue(payload())
        self.client.is_connected.return_value = False
        self.assertFalse(publisher.publish_once())
        self.client.publish.assert_not_called()
        self.assertEqual(MessageQueue(self.path).pending_count(), 1)
        self.client.is_connected.return_value = True
        info = Mock(rc=mqtt.MQTT_ERR_SUCCESS)
        info.is_published.return_value = False
        self.client.publish.return_value = info
        self.assertFalse(publisher.publish_once())
        self.assertEqual(MessageQueue(self.path).pending_count(), 1)
        self.ready_for_retry()
        info.is_published.return_value = True
        self.assertTrue(publisher.publish_once())
        sent = self.client.publish.call_args
        self.assertEqual(sent.kwargs, {"qos": 1, "retain": False})
        self.assertEqual(sent.args[0], "triago/test/devices/raspi-01/triage")
        self.assertEqual(MessageQueue(self.path).pending_count(), 0)

    def test_api_client_mqtt_enqueues_without_http_or_network_wait(self):
        with (patch.dict(os.environ, {"TRIAGO_TRANSPORT": "mqtt"}),
              patch("service.api_client.load_config"),
              patch("service.mqtt_client.MqttPublisher") as publisher,
              patch("service.api_client.requests.post") as post):
            api = TriageApiClient()
            self.assertTrue(api.send_triage_result("A1", 15, payload()["vitals"], "Green", 0.2,
                                                   patient_name="Pasien Uji"))
            self.assertEqual(api.last_delivery_status, "queued")
            post.assert_not_called()
            sent = publisher.return_value.enqueue.call_args.args[0]
            self.assertEqual(sent["patient_name"], "Pasien Uji")
            self.assertEqual(sent["triage_category"], "green")

    def test_api_client_starts_outbox_on_startup_without_new_measurement(self):
        with (patch.dict(os.environ, {"TRIAGO_TRANSPORT": "mqtt"}),
              patch("service.api_client.load_config"),
              patch("service.mqtt_client.MqttPublisher") as publisher):
            api = TriageApiClient()
            publisher.assert_called_once_with()
            publisher.return_value.enqueue.assert_not_called()
            api.close()
            publisher.return_value.close.assert_called_once()

    def test_tls_verifies_server_certificate_and_hostname(self):
        with patch("service.mqtt_client.mqtt.Client") as constructor:
            client = create_client(config(), subscriber=True)
            context = client.tls_set_context.call_args.args[0]
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)
            self.assertFalse(constructor.call_args.kwargs["clean_session"])
            self.assertTrue(constructor.call_args.kwargs["manual_ack"])

    def test_subscription_status_tracks_broker_permissions(self):
        receiver = self.receiver()
        receiver._on_subscribe(self.client, None, 1, [SimpleNamespace(is_failure=False)], None)
        self.assertTrue(receiver.status()["subscribed"])
        receiver._on_subscribe(self.client, None, 2, [SimpleNamespace(is_failure=True)], None)
        self.assertFalse(receiver.status()["subscribed"])

    def test_api_client_returns_failure_when_outbox_cannot_be_written(self):
        with (patch.dict(os.environ, {"TRIAGO_TRANSPORT": "mqtt"}),
              patch("service.api_client.load_config"),
              patch("service.mqtt_client.MqttPublisher") as publisher):
            publisher.return_value.enqueue.side_effect = sqlite3.OperationalError("read only")
            api = TriageApiClient()
            self.assertFalse(api.send_triage_result("A1", 15, payload()["vitals"], "green", 0.2))
            self.assertEqual(api.last_delivery_status, "not_sent")

    def test_mqtt_requires_credentials_and_valid_topic(self):
        with patch.dict(os.environ, {}, clear=True), patch("service.mqtt_client.load_config"):
            with self.assertRaisesRegex(ValueError, "TRIAGO_MQTT_HOST"):
                MqttConfig.from_env("publisher")
        with (patch.dict(os.environ, {"TRIAGO_MQTT_HOST": "host", "TRIAGO_MQTT_USERNAME": "u",
                                    "TRIAGO_MQTT_PASSWORD": "p", "TRIAGO_DEVICE_ID": "bad/+"}),
              patch("service.mqtt_client.load_config")):
            with self.assertRaisesRegex(ValueError, "DEVICE_ID"):
                MqttConfig.from_env("publisher")

    def test_malformed_payload_is_rejected(self):
        for invalid in (None, [], "text", {"bed_id": "A1"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_payload(invalid)
        bad = envelope()
        bad["event_id"] = "not-a-uuid"
        with self.assertRaises(ValueError):
            validate_envelope(bad)
        bad = payload()
        bad["vitals"]["hr"] = 10 ** 1000
        with self.assertRaises(ValueError):
            validate_payload(bad)


class HttpRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"TRIAGO_MQTT_ENABLED": "0", "TRIAGO_DB_BACKEND": "mysql",
                                    "TRIAGO_REQUIRE_AUTH": "0", "TRIAGO_DASHBOARD_USER": "",
                                    "TRIAGO_DASHBOARD_PASSWORD": ""}):
            from backend import app as module
        cls.backend = module

    def setUp(self):
        self.client = self.backend.app.test_client()
        self.service = TriageService(BedManager(), Mock(return_value=True), Mock())
        patcher = patch.object(self.backend, "triage_service", self.service)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_http_triage_contract_still_works(self):
        response = self.client.post("/api/triage/update", json=payload())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["assigned_bed"], "A1")
        self.assertTrue(response.json["saved_to_database"])
        self.service.emit.assert_called_once()

    def test_http_rejects_invalid_json_without_saving(self):
        for data in ([], {"vitals": []}):
            response = self.client.post("/api/triage/update", json=data)
            self.assertEqual(response.status_code, 400)
        self.service.save_patient.assert_not_called()

    def test_health_exposes_disabled_mqtt(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["mqtt"], {"enabled": False})


if __name__ == "__main__":
    unittest.main()
