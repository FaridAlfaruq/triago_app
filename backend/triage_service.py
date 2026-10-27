"""One triage-processing path for HTTP and MQTT."""
from collections import OrderedDict
from copy import deepcopy
import logging
import threading
import time

from service.triage_payload import validate_payload

log = logging.getLogger(__name__)


class DatabaseUnavailable(RuntimeError):
    pass


class TriageService:
    def __init__(self, bed_manager, save_patient, emit):
        self.bed_manager = bed_manager
        self.save_patient = save_patient
        self.emit = emit
        self.lock = threading.Lock()
        self.receipts = OrderedDict()

    def process(self, data, *, event_id=None, require_database=False):
        data = validate_payload(data)
        with self.lock:
            if event_id in self.receipts:
                return deepcopy(self.receipts[event_id])
            saved_to_db = False

            def persist(db_payload):
                nonlocal saved_to_db
                saved_to_db = self.save_patient(db_payload)
                if require_database and not saved_to_db:
                    raise DatabaseUnavailable("Database belum menyimpan event MQTT")

            assigned = self.bed_manager.assign_patient_to_bed(data, persist=persist)
            if assigned is None:
                persist({**data, "bed_id": "WAITING", "zone": "-",
                         "arrival_timestamp": int(time.time() * 1000)})
                result = {"status": "warning", "assigned_bed": None, "zone": None,
                          "message": "Bed zona ini penuh; pasien masuk daftar tunggu.",
                          "saved_to_database": saved_to_db}
            else:
                bed = deepcopy(self.bed_manager.get_all_beds_status()[assigned])
                result = {"status": "success", "assigned_bed": assigned, "zone": bed["zone"],
                          "message": f"Pasien diarahkan ke bed {assigned}",
                          "saved_to_database": saved_to_db, "data": bed}
            if event_id is not None:
                self.receipts[event_id] = deepcopy(result)
                if len(self.receipts) > 2048:
                    self.receipts.popitem(last=False)
            if assigned is not None:
                try:
                    self.emit("bed_updated", bed)
                except Exception:
                    # A dashboard failure must not repeat a committed DB insert.
                    log.exception("Broadcast gagal; refresh dashboard untuk status terbaru")
            return result
