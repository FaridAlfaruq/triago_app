import os
import time
import mysql.connector
from mysql.connector import Error

# =========================================================================
# KONFIGURASI (ambil dari environment variable, jangan hardcode password)
# =========================================================================
DB_HOST = os.environ.get("TRIAGO_DB_HOST", "localhost")
DB_USER = os.environ.get("TRIAGO_DB_USER", "root")
DB_PASSWORD = os.environ.get("TRIAGO_DB_PASSWORD", "parapencariberkah")
DB_NAME = os.environ.get("TRIAGO_DB_NAME", "triago")

_db = None


def _connect():
    """Membuat koneksi baru ke MySQL."""
    global _db
    _db = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )
    print(f"[DATABASE] Berhasil terkoneksi ke MySQL ({DB_HOST}/{DB_NAME})")


def get_connection():
    """Mengembalikan koneksi aktif. Reconnect otomatis bila koneksi terputus
    (mengatasi error 'MySQL server has gone away' pada server yang jalan lama)."""
    global _db
    try:
        if _db is None or not _db.is_connected():
            _connect()
    except Error as err:
        print(f"[DATABASE][ERROR] Gagal konek ke MySQL: {err}")
        raise
    return _db


# Percobaan koneksi pertama saat modul di-import.
# Dibuat try/except supaya Flask TIDAK crash total kalau MySQL belum
# menyala saat backend pertama kali dijalankan (misal urutan boot Raspberry Pi).
try:
    _connect()
except Error as err:
    print(f"[DATABASE][WARN] MySQL belum siap saat startup: {err}")
    print("[DATABASE][WARN] Server tetap berjalan, akan mencoba reconnect otomatis saat ada data masuk.")


def save_patient(data: dict) -> bool:
    """Simpan satu baris hasil pengukuran/klasifikasi pasien ke tabel patient_data.
    Mengembalikan True bila berhasil, False bila gagal (server tidak akan ikut crash)."""
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
        xgboost_score,
        heart_rate,
        spo2,
        temperature,
        respiration_rate,
        systolic_bp,
        diastolic_bp,
        arrival_timestamp
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
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
        conn.rollback()
        return False
    finally:
        cursor.close()


def get_patient_history(limit: int = 50):
    """Ambil riwayat pengukuran terbaru dari MySQL (opsional, untuk laporan/riwayat)."""
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