import os
import time
import sqlite3

try:
    import mysql.connector
    from mysql.connector import Error
    HAS_MYSQL = True
except ImportError:
    HAS_MYSQL = False
    Error = Exception

# =========================================================================
# KONFIGURASI (ambil dari environment variable, jangan hardcode password)
# =========================================================================
DB_HOST = os.environ.get("TRIAGO_DB_HOST", "localhost")
DB_USER = os.environ.get("TRIAGO_DB_USER", "root")
DB_PASSWORD = os.environ.get("TRIAGO_DB_PASSWORD", "parapencariberkah")
DB_NAME = os.environ.get("TRIAGO_DB_NAME", "triago")

_db = None
_is_sqlite = False


def _init_sqlite_schema(conn):
    """Membuat tabel patient_data di SQLite jika belum ada."""
    schema = """
    CREATE TABLE IF NOT EXISTS patient_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bed_id TEXT,
        zone TEXT,
        status TEXT,
        patient_name TEXT,
        relative_name TEXT,
        gcs_score INTEGER,
        xgboost_score REAL,
        heart_rate REAL,
        spo2 REAL,
        temperature REAL,
        respiration_rate REAL,
        systolic_bp INTEGER,
        diastolic_bp INTEGER,
        arrival_timestamp INTEGER
    );
    """
    cursor = conn.cursor()
    cursor.execute(schema)
    conn.commit()
    cursor.close()


def _connect():
    """Membuat koneksi ke MySQL atau SQLite fallback."""
    global _db, _is_sqlite
    if HAS_MYSQL:
        try:
            _db = mysql.connector.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
            )
            _is_sqlite = False
            print(f"[DATABASE] Berhasil terkoneksi ke MySQL ({DB_HOST}/{DB_NAME})")
            return
        except Exception:
            pass

    # Fallback to local SQLite database in backend directory
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "triage.db")
    _db = sqlite3.connect(db_path, check_same_thread=False)
    _db.row_factory = sqlite3.Row
    _is_sqlite = True
    _init_sqlite_schema(_db)
    print(f"[DATABASE] Berhasil menggunakan database SQLite3 ({db_path})")


def get_connection():
    """Mengembalikan koneksi aktif."""
    global _db
    try:
        if _db is None:
            _connect()
        elif not _is_sqlite and HAS_MYSQL:
            if not getattr(_db, "is_connected", lambda: True)():
                _connect()
    except Exception as err:
        print(f"[DATABASE][WARN] Reconnecting database due to error: {err}")
        _connect()
    return _db


# Percobaan koneksi pertama saat modul di-import
try:
    _connect()
except Exception as err:
    print(f"[DATABASE][WARN] Database initialization notice: {err}")


def save_patient(data: dict) -> bool:
    """Simpan satu baris hasil pengukuran/klasifikasi pasien ke tabel patient_data."""
    try:
        conn = get_connection()
    except Exception as e:
        print(f"[DATABASE][ERROR] Data TIDAK disimpan karena koneksi database gagal: {e}")
        return False

    cursor = conn.cursor()
    
    placeholder = "?" if _is_sqlite else "%s"
    sql = f"""
    INSERT INTO patient_data(
        bed_id, zone, status, patient_name, relative_name,
        gcs_score, xgboost_score, heart_rate, spo2, temperature,
        respiration_rate, systolic_bp, diastolic_bp, arrival_timestamp
    )
    VALUES ({placeholder},{placeholder},{placeholder},{placeholder},{placeholder},{placeholder},{placeholder},{placeholder},{placeholder},{placeholder},{placeholder},{placeholder},{placeholder},{placeholder})
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
        db_type = "SQLite3" if _is_sqlite else "MySQL"
        print(f"[DATABASE] Data pasien {data.get('bed_id')} berhasil disimpan ke {db_type}.", flush=True)
        return True
    except Exception as err:
        print(f"[DATABASE][ERROR] Gagal insert data pasien: {err}", flush=True)
        conn.rollback()
        return False
    finally:
        cursor.close()


def get_patient_history(limit: int = 50):
    """Ambil riwayat pengukuran terbaru dari database."""
    try:
        conn = get_connection()
    except Exception:
        return []

    cursor = conn.cursor()
    try:
        placeholder = "?" if _is_sqlite else "%s"
        query = f"SELECT * FROM patient_data ORDER BY arrival_timestamp DESC LIMIT {placeholder}"
        cursor.execute(query, (limit,))
        
        if _is_sqlite:
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        else:
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    except Exception as err:
        print(f"[DATABASE][ERROR] Gagal mengambil riwayat pasien: {err}")
        return []
    finally:
        cursor.close()


def update_patient_info(bed_id: str, patient_name: str, relative_name: str) -> bool:
    """Memperbarui nama pasien dan penanggung jawab pada baris terbaru."""
    try:
        conn = get_connection()
    except Exception:
        print("[DATABASE][ERROR] Gagal update identitas pasien karena koneksi database gagal.", flush=True)
        return False

    cursor = conn.cursor()
    placeholder = "?" if _is_sqlite else "%s"
    
    if _is_sqlite:
        sql = f"UPDATE patient_data SET patient_name = {placeholder}, relative_name = {placeholder} WHERE id = (SELECT id FROM patient_data WHERE bed_id = {placeholder} ORDER BY arrival_timestamp DESC LIMIT 1)"
        params = (patient_name, relative_name, bed_id)
    else:
        sql = f"UPDATE patient_data SET patient_name = {placeholder}, relative_name = {placeholder} WHERE bed_id = {placeholder} ORDER BY arrival_timestamp DESC LIMIT 1"
        params = (patient_name, relative_name, bed_id)

    try:
        cursor.execute(sql, params)
        conn.commit()
        print(f"[DATABASE] Identitas pasien Bed {bed_id} berhasil diperbarui.", flush=True)
        return True
    except Exception as err:
        print(f"[DATABASE][ERROR] Gagal update identitas pasien: {err}", flush=True)
        conn.rollback()
        return False
    finally:
        cursor.close()