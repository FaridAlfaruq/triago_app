"""Explicit SQLite storage for laptop trials without a MySQL server."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time


class SQLiteStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS patient_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bed_id TEXT, zone TEXT, status TEXT,
                patient_name TEXT, relative_name TEXT,
                gcs_score INTEGER, xgboost_score REAL,
                pain_score INTEGER,
                heart_rate REAL, spo2 REAL, temperature REAL,
                respiration_rate REAL, systolic_bp INTEGER,
                diastolic_bp INTEGER, arrival_timestamp INTEGER,
                patient_details TEXT
            )""")
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(patient_data)")}
            if "pain_score" not in columns:
                conn.execute("ALTER TABLE patient_data ADD COLUMN pain_score INTEGER")
            if "patient_details" not in columns:
                conn.execute("ALTER TABLE patient_data ADD COLUMN patient_details TEXT")

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def save_patient(self, data):
        vitals = data.get("vitals") or {}

        def vital(*keys):
            return next((vitals[key] for key in keys if vitals.get(key) is not None), None)

        values = (
            data.get("bed_id"), data.get("zone"), data.get("triage_category"),
            data.get("patient_name"), data.get("relative_name"),
            data.get("gcs_score"), data.get("xgboost_score"),
            data.get("pain_score"),
            vital("hr", "heart_rate"), vital("spo2"),
            vital("temp_core", "temperature", "temp", "temp_skin"),
            vital("rr", "respiration_rate"), vital("sys", "systolic_bp", "systolic"),
            vital("dia", "diastolic_bp", "diastolic"),
            data.get("arrival_timestamp", int(time.time() * 1000)),
        )
        with self.connection() as conn:
            conn.execute("""INSERT INTO patient_data (
                bed_id, zone, status, patient_name, relative_name,
                gcs_score, xgboost_score, pain_score, heart_rate, spo2, temperature,
                respiration_rate, systolic_bp, diastolic_bp, arrival_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", values)
        return True

    def get_patient_history(self, limit=50):
        with self.connection() as conn:
            rows = conn.execute("""SELECT * FROM patient_data
                ORDER BY arrival_timestamp DESC, id DESC LIMIT ?""", (max(0, limit),)).fetchall()
        return [dict(row) for row in rows]

    def update_patient_info(self, bed_id, patient_name, relative_name, patient_details=None):
        details = json.dumps(patient_details, ensure_ascii=False) if patient_details else None
        with self.connection() as conn:
            cursor = conn.execute("""UPDATE patient_data
                SET patient_name = ?, relative_name = ?, patient_details = ?
                WHERE id = (SELECT id FROM patient_data WHERE bed_id = ?
                            ORDER BY arrival_timestamp DESC, id DESC LIMIT 1)""",
                                  (patient_name, relative_name, details, bed_id))
            return cursor.rowcount > 0
