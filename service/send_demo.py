"""Send one synthetic measurement through the configured HiveMQ connection.

Run: python -m service.send_demo
No sensor, model, or public dashboard is needed.
"""
import logging
import time

from service.config import load_config
from service.mqtt_client import MqttPublisher


def main():
    load_config()
    logging.basicConfig(level=logging.INFO)
    try:
        publisher = MqttPublisher()
    except (ValueError, OSError) as exc:
        print(f"Konfigurasi demo belum siap: {exc}")
        return 1
    try:
        event_id = publisher.enqueue({
            "bed_id": "A1", "gcs_score": 15, "triage_category": "green",
            "pain_score": 3,
            "xgboost_score": 0.2, "patient_name": "SIMULASI MQTT",
            "relative_name": "DATA DUMMY",
            "vitals": {"hr": 75, "spo2": 98, "rr": 16, "temp_core": 36.5,
                       "sys": 120, "dia": 80},
        })
        print(f"Event simulasi {event_id} masuk outbox. Menunggu broker hingga 20 detik...")
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if publisher.queue.pending_count() == 0:
                print("Outbox kosong: broker telah mengakui pengiriman.")
                print("Cek dashboard dan /api/history untuk memastikan backend menyimpannya.")
                return 0
            time.sleep(0.25)
        print("Belum selesai. Pesan tetap tersimpan di outbox; GUI/demo berikutnya akan mencoba ulang.")
        return 2
    finally:
        publisher.close()


if __name__ == "__main__":
    raise SystemExit(main())
