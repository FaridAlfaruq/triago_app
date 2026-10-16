"""Durable HiveMQ subscriber; processing shares the HTTP triage service."""
import atexit
import json
import logging
import os
import sqlite3
import threading

from service.config import PROJECT_ROOT
from service.mqtt_client import MqttConfig, create_client
from service.mqtt_queue import MessageQueue
from service.triage_payload import validate_envelope

log = logging.getLogger(__name__)


class MqttReceiver:
    def __init__(self, process, config=None, queue_path=None):
        self.process = process
        self.config = config or MqttConfig.from_env("backend")
        self.queue = MessageQueue(queue_path or os.environ.get("TRIAGO_MQTT_INBOX")
                                  or PROJECT_ROOT / "runtime" / "mqtt-inbox.sqlite3")
        self.client = create_client(self.config, subscriber=True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_subscribe = self._on_subscribe
        self.client.on_message = self._on_message
        self.subscribed = False
        self.stop_event = threading.Event()
        self.reconnect_requested = threading.Event()
        self.worker = threading.Thread(target=self._run, name="mqtt-inbox", daemon=True)

    def start(self):
        self.client.connect_async(self.config.host, self.config.port, keepalive=60)
        self.client.loop_start()
        self.worker.start()
        atexit.register(self.close)

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.error("MQTT backend ditolak broker: %s", reason_code)
            return
        client.subscribe(self.config.subscribe_topic, qos=1)

    def _on_subscribe(self, client, userdata, mid, reason_codes, properties):
        self.subscribed = bool(reason_codes) and all(not code.is_failure for code in reason_codes)
        if not self.subscribed:
            log.error("Subscription MQTT ditolak; periksa izin topic di HiveMQ")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.subscribed = False

    def _on_message(self, client, userdata, message):
        try:
            if message.retain or len(message.payload) > 65536:
                raise ValueError("Pesan retained/terlalu besar ditolak")
            envelope = validate_envelope(json.loads(message.payload))
            expected = f"{self.config.topic_prefix}/devices/{envelope['device_id']}/triage"
            if message.topic != expected:
                raise ValueError("device_id tidak sesuai topic")
            self.queue.put(envelope)
        except (ValueError, TypeError, UnicodeError, RecursionError):
            # Invalid messages are poison messages, not transient failures.
            log.warning("Pesan MQTT ditolak: format/topic tidak valid")
        except (sqlite3.Error, OSError):
            # No PUBACK: reconnect and let the persistent broker session redeliver.
            log.exception("Inbox MQTT tidak dapat ditulis; pesan belum di-ACK")
            self.reconnect_requested.set()
            return
        client.ack(message.mid, message.qos)

    def process_once(self):
        row = self.queue.next_message()
        if not row:
            return False
        envelope = json.loads(row["payload"])
        try:
            self.process(envelope["data"], event_id=envelope["event_id"], require_database=True)
            self.queue.complete(envelope["event_id"])
            log.info("MQTT event %s selesai disimpan dan diproses", envelope["event_id"])
            return True
        except Exception as exc:
            self.queue.retry(envelope["event_id"])
            log.warning("MQTT event %s tertunda (%s); akan dicoba ulang",
                        envelope["event_id"], type(exc).__name__)
            return False

    def _run(self):
        while not self.stop_event.is_set():
            try:
                if self.reconnect_requested.is_set():
                    self.client.disconnect()
                    self.client.loop_stop()
                    self.client.connect_async(self.config.host, self.config.port, keepalive=60)
                    self.client.loop_start()
                    self.reconnect_requested.clear()
                if self.process_once():
                    continue
            except Exception:
                log.exception("MQTT inbox gagal diproses; data tetap di SQLite")
            self.stop_event.wait(1)

    def status(self):
        return {"enabled": True, "connected": self.client.is_connected(),
                "subscribed": self.subscribed, "pending": self.queue.pending_count()}

    def close(self):
        self.stop_event.set()
        self.worker.join(timeout=6)
        self.client.disconnect()
        self.client.loop_stop()
