"""HiveMQ TLS connection and durable Raspberry Pi publisher."""
import atexit
from dataclasses import dataclass
import json
import logging
import os
import ssl
import threading
from uuid import uuid4

import paho.mqtt.client as mqtt

from service.config import PROJECT_ROOT, load_config
from service.mqtt_queue import MessageQueue
from service.triage_payload import validate_envelope, validate_payload

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MqttConfig:
    host: str
    port: int
    username: str
    password: str
    topic_prefix: str
    device_id: str
    client_id: str
    ca_file: str | None = None

    @classmethod
    def from_env(cls, role):
        load_config()
        required = ["TRIAGO_MQTT_HOST", "TRIAGO_MQTT_USERNAME", "TRIAGO_MQTT_PASSWORD"]
        missing = [key for key in required if not os.environ.get(key)]
        if missing:
            raise ValueError("Konfigurasi MQTT belum lengkap: " + ", ".join(missing))
        device = os.environ.get("TRIAGO_DEVICE_ID", "raspi-01")
        prefix = os.environ.get("TRIAGO_MQTT_TOPIC_PREFIX", "triago/biomed").strip("/")
        if not prefix or any(c in prefix for c in "+#\x00") or "//" in prefix:
            raise ValueError("TRIAGO_MQTT_TOPIC_PREFIX tidak valid")
        if not device or any(c in device for c in "/+#\x00"):
            raise ValueError("TRIAGO_DEVICE_ID tidak valid")
        # Separate IDs permit publisher + backend in two terminals on one laptop.
        client_id = os.environ.get(f"TRIAGO_MQTT_{role.upper()}_CLIENT_ID",
                                   os.environ.get("TRIAGO_MQTT_CLIENT_ID", f"triago-{device}-{role}"))
        if not client_id:
            raise ValueError("TRIAGO_MQTT_CLIENT_ID tidak boleh kosong")
        port = int(os.environ.get("TRIAGO_MQTT_PORT", "8883"))
        if not 1 <= port <= 65535:
            raise ValueError("Port MQTT tidak valid")
        return cls(os.environ[required[0]], port, os.environ[required[1]],
                   os.environ[required[2]], prefix, device, client_id,
                   os.environ.get("TRIAGO_MQTT_CA_FILE") or None)

    @property
    def publish_topic(self):
        return f"{self.topic_prefix}/devices/{self.device_id}/triage"

    @property
    def subscribe_topic(self):
        return f"{self.topic_prefix}/devices/+/triage"


def create_client(config, *, subscriber=False):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=config.client_id, protocol=mqtt.MQTTv311,
                         clean_session=not subscriber, manual_ack=subscriber)
    client.username_pw_set(config.username, config.password)
    # Certificate and hostname verification remain enabled, including custom CAs.
    client.tls_set_context(ssl.create_default_context(cafile=config.ca_file))
    client.reconnect_delay_set(min_delay=1, max_delay=60)
    client.max_queued_messages_set(1)
    return client


class MqttPublisher:
    def __init__(self, config=None, queue_path=None):
        self.config = config or MqttConfig.from_env("publisher")
        self.queue = MessageQueue(queue_path or os.environ.get("TRIAGO_MQTT_OUTBOX")
                                  or PROJECT_ROOT / "runtime" / "mqtt-outbox.sqlite3")
        self.client = create_client(self.config)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._run, name="mqtt-outbox", daemon=True)
        self.client.connect_async(self.config.host, self.config.port, keepalive=60)
        self.client.loop_start()
        self.worker.start()
        atexit.register(self.close)

    def enqueue(self, data):
        envelope = {"schema_version": 1, "event_id": str(uuid4()),
                    "device_id": self.config.device_id, "data": validate_payload(data)}
        self.queue.put(validate_envelope(envelope))
        return envelope["event_id"]

    def publish_once(self):
        if not self.client.is_connected():
            return False
        row = self.queue.next_message()
        if not row:
            return False
        try:
            # Use the queued identity, including messages from before a config change.
            envelope = validate_envelope(json.loads(row["payload"]))
            topic = f"{self.config.topic_prefix}/devices/{envelope['device_id']}/triage"
            info = self.client.publish(topic, row["payload"], qos=1, retain=False)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                raise RuntimeError("Broker belum menerima pesan")
            info.wait_for_publish(timeout=5)
            if not info.is_published():
                raise TimeoutError("PUBACK belum diterima")
            self.queue.complete(row["event_id"])
            log.info("MQTT event %s diterima broker; pemrosesan backend terpisah", row["event_id"])
            return True
        except (RuntimeError, ValueError, OSError) as exc:
            self.queue.retry(row["event_id"])
            log.warning("MQTT publish belum selesai (%s); event tetap di outbox", type(exc).__name__)
            return False

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.publish_once()
            except Exception:
                log.exception("MQTT outbox gagal diproses; data tetap di SQLite")
            self.stop_event.wait(1)

    def close(self):
        self.stop_event.set()
        self.worker.join(timeout=6)
        self.client.disconnect()
        self.client.loop_stop()
