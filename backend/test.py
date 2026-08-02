"""
Script tes cepat untuk memastikan database.py bisa insert data.
Jalankan langsung: python3 test_db_insert.py
Taruh file ini SEJAJAR dengan database.py (folder backend).

Kalau ada error, akan tercetak jelas di terminal -- baca pesan errornya
untuk tahu penyebab pastinya (kolom tidak ada, password salah, dll).
"""
import database

# Contoh payload lengkap, meniru apa yang dikirim GUI ke /api/triage/update
dummy_payload = {
    "bed_id": "A1",
    "zone": "A",
    "triage_category": "red",
    "patient_name": "Test Pasien",
    "relative_name": "Test Keluarga",
    "gcs_score": 15,
    "xgboost_score": 0.88,
    "vitals": {
        "hr": 110.5,
        "spo2": 98.2,
        "rr": 16,
        "sys": 120,
        "dia": 80,
        "temp_core": 36.5,
    },
}

print("--- Mulai tes insert ke MySQL ---")
print(f"Host   : {database.DB_HOST}")
print(f"User   : {database.DB_USER}")
print(f"DB name: {database.DB_NAME}")
print()

# Coba lihat dulu struktur tabel yang sebenarnya ada di MySQL,
# supaya ketahuan kalau nama kolomnya beda dari yang diharapkan.
try:
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute("DESCRIBE patient_data")
    print("Struktur tabel patient_data saat ini:")
    for row in cursor.fetchall():
        print(" -", row[0], "|", row[1])
    cursor.close()
    print()
except Exception as e:
    print(f"[GAGAL] Tidak bisa membaca struktur tabel: {e}")
    print("Kemungkinan: MySQL belum jalan, atau tabel patient_data belum dibuat.")
    exit(1)

# Coba insert data dummy
result = database.save_patient(dummy_payload)
print()
if result:
    print("BERHASIL: data dummy berhasil masuk ke MySQL.")
else:
    print("GAGAL: cek pesan [DATABASE][ERROR] di atas untuk tahu penyebabnya.")