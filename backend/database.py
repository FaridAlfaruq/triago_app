import json
import os
import time
import threading
from pathlib import Path
import mysql.connector
from mysql.connector import Error

# =========================================================================
# KONFIGURASI (ambil dari environment variable, jangan hardcode password)
# =========================================================================
DB_HOST = os.environ.get("TRIAGO_DB_HOST", "localhost")
DB_USER = os.environ.get("TRIAGO_DB_USER", "root")
DB_PASSWORD = os.environ.get("TRIAGO_DB_PASSWORD", "")
DB_NAME = os.environ.get("TRIAGO_DB_NAME", "triago")
DB_BACKEND = os.environ.get("TRIAGO_DB_BACKEND", "mysql").lower()
if DB_BACKEND not in {"mysql", "sqlite"}:
    raise ValueError("TRIAGO_DB_BACKEND harus mysql atau sqlite")

_sqlite_store = None
if DB_BACKEND == "sqlite":
    try:
        from .sqlite_store import SQLiteStore
    except ImportError:
        from sqlite_store import SQLiteStore
    _sqlite_store = SQLiteStore(os.environ.get("TRIAGO_SQLITE_PATH")
                                or Path(__file__).resolve().parent.parent / "runtime" / "triago-laptop.sqlite3")

_connections = threading.local()


def _connect():
    """Membuat koneksi baru ke MySQL."""
    _connections.db = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        connection_timeout=5,
    )
    print(f"[DATABASE] Berhasil terkoneksi ke MySQL ({DB_HOST}/{DB_NAME})")


def get_connection():
    """Mengembalikan koneksi aktif. Reconnect otomatis bila koneksi terputus
    (mengatasi error 'MySQL server has gone away' pada server yang jalan lama)."""
    try:
        db = getattr(_connections, "db", None)
        if db is None or not db.is_connected():
            _connect()
    except Error as err:
        print(f"[DATABASE][ERROR] Gagal konek ke MySQL: {err}")
        raise
    return _connections.db


# Koneksi dibuat lazily per thread: HTTP dan MQTT tidak berbagi koneksi MySQL.


def save_patient(data: dict) -> bool:
    """Simpan satu baris hasil pengukuran/klasifikasi pasien ke tabel patient_data.
    Mengembalikan True bila berhasil, False bila gagal (server tidak akan ikut crash)."""
    if _sqlite_store is not None:
        return _sqlite_store.save_patient(data)
    try:
        conn = get_connection()
    except Error:
        print("[DATABASE][ERROR] Data TIDAK disimpan karena koneksi MySQL gagal.")
        return False

    cursor = conn.cursor()
    sql = """
    INSERT INTO patient_data(
        bed_id,
        zone,
        status,
        patient_name,
        relative_name,
        gcs_score,
        pain_score,
        xgboost_score,
        heart_rate,
        spo2,
        temperature,
        respiration_rate,
        systolic_bp,
        diastolic_bp,
        arrival_timestamp
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    vitals = data.get("vitals", {})
    if not isinstance(vitals, dict):
        vitals = {}

    def _get_vital(keys, default=None):
        for k in keys:
            val = vitals.get(k)
            if val is not None:
                return val
        return default

    values = (
        data.get("bed_id"),
        data.get("zone"),
        data.get("triage_category"),
        data.get("patient_name"),
        data.get("relative_name"),
        data.get("gcs_score"),
        data.get("pain_score"),
        data.get("xgboost_score"),
        _get_vital(["hr", "heart_rate"]),
        _get_vital(["spo2"]),
        _get_vital(["temp_core", "temperature", "temp", "temp_skin"]),
        _get_vital(["rr", "respiration_rate"]),
        _get_vital(["sys", "systolic_bp", "systolic"]),
        _get_vital(["dia", "diastolic_bp", "diastolic"]),
        data.get("arrival_timestamp", int(time.time() * 1000)),
    )

    try:
        cursor.execute(sql, values)
        conn.commit()
        print("[DATABASE] Data pasien berhasil disimpan ke MySQL.", flush=True)
        return True
    except Error as err:
        print(f"[DATABASE][ERROR] Gagal insert data pasien: {err}", flush=True)
        if getattr(err, "errno", None) == 1054:
            print("[DATABASE] Kolom tabel belum lengkap; jalankan migrasi di backend/migrations/", flush=True)
        conn.rollback()
        return False
    finally:
        cursor.close()


def get_patient_history(limit: int = 50):
    """Ambil riwayat pengukuran terbaru dari MySQL (opsional, untuk laporan/riwayat)."""
    if _sqlite_store is not None:
        return _sqlite_store.get_patient_history(limit)
    try:
        conn = get_connection()
    except Error:
        return []

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT * FROM patient_data ORDER BY arrival_timestamp DESC LIMIT %s",
            (limit,),
        )
        return cursor.fetchall()
    except Error as err:
        print(f"[DATABASE][ERROR] Gagal mengambil riwayat pasien: {err}")
        return []
    finally:
        cursor.close()


def update_patient_info(bed_id: str, patient_name: str, relative_name: str,
                        patient_details: dict = None) -> bool:
    """Memperbarui identitas pasien pada baris terbaru di MySQL untuk bed tertentu."""
    if _sqlite_store is not None:
        return _sqlite_store.update_patient_info(bed_id, patient_name, relative_name, patient_details)
    try:
        conn = get_connection()
    except Error:
        print("[DATABASE][ERROR] Gagal update identitas pasien karena koneksi MySQL gagal.", flush=True)
        return False

    cursor = conn.cursor()
    sql = """
    UPDATE patient_data 
    SET patient_name = %s, relative_name = %s, patient_details = %s
    WHERE bed_id = %s 
    ORDER BY arrival_timestamp DESC 
    LIMIT 1
    """
    details = json.dumps(patient_details, ensure_ascii=False) if patient_details else None
    try:
        cursor.execute(sql, (patient_name, relative_name, details, bed_id))
        conn.commit()
        print(f"[DATABASE] Identitas pasien Bed {bed_id} berhasil diperbarui di MySQL.", flush=True)
        return True
    except Error as err:
        print(f"[DATABASE][ERROR] Gagal update identitas pasien: {err}", flush=True)
        conn.rollback()
        return False
    finally:
        cursor.close()
