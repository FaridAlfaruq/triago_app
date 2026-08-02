import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

try:
    from .bed_manager import BedManager
    from .database import save_patient, get_patient_history
except ImportError:
    # Tetap mendukung eksekusi langsung: python backend/app.py
    from bed_manager import BedManager
    from database import save_patient, get_patient_history

# Inisialisasi Flask App & Arahkan static folder ke folder 'website'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEBSITE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "website"))
SERVER_HOST = os.environ.get("TRIAGO_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("TRIAGO_PORT", "5000"))
DEBUG_MODE = os.environ.get("TRIAGO_DEBUG", "0").lower() in {"1", "true", "yes"}
CORS_ORIGINS = os.environ.get("TRIAGO_CORS_ORIGINS", "*")

app = Flask(__name__, static_folder=WEBSITE_DIR, static_url_path="")
CORS(
    app,
    resources={r"/api/*": {"origins": CORS_ORIGINS}},
)

# Inisialisasi SocketIO untuk komunikasi dua arah (Real-time)
socketio = SocketIO(
    app,
    cors_allowed_origins=CORS_ORIGINS,
    async_mode="threading",
)

# Inisialisasi BedManager untuk mengelola status bed IGD
bed_manager = BedManager()


# =========================================================================
# 1. ROUTE WEB DASHBOARD
# =========================================================================
@app.route("/")
def index():
    """Melayani halaman utama index.html dari folder website."""
    return send_from_directory(WEBSITE_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health-check sederhana untuk pengujian koneksi dari laptop."""
    return jsonify({
        "status": "ok",
        "service": "triago-backend",
        "port": SERVER_PORT,
    })


# =========================================================================
# 2. ENDPOINT API UNTUK PENGIRIMAN DATA DARI GUI / HARDWARE
# =========================================================================
@app.route("/api/triage/update", methods=["POST"])
def update_triage():
    """
    Endpoint yang dipanggil oleh TriageApiClient di output_page.py.
    Menerima JSON payload parameter vital sign & hasil klasifikasi,
    lalu (1) mengalokasikan bed, (2) menyimpan ke MySQL, (3) broadcast ke dashboard.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Payload JSON tidak ditemukan"}), 400

    print("\n[SERVER LOG] Menerima data baru dari hardware/GUI:", flush=True)
    print(f" -> Bed ID Target : {data.get('bed_id')}", flush=True)
    print(f" -> Kategori Triase: {data.get('triage_category')}", flush=True)
    print(f" -> Pasien         : {data.get('patient_name')}", flush=True)

    # 1. Masukkan/Update status pasien pada BedManager (in-memory, untuk dashboard real-time)
    assigned_bed_id = bed_manager.assign_patient_to_bed(data)

    if not assigned_bed_id:
        # Semua bed di semua zona penuh. JANGAN buang data pasien --
        # tetap simpan ke MySQL (ditandai bed_id "WAITING") supaya
        # data pengukuran tidak hilang, lalu beri tahu operator.
        print(f"[SERVER LOG][WARN] Semua bed penuh untuk kategori '{data.get('triage_category')}'. "
              f"Data disimpan sebagai antrian, tidak ditempatkan di bed manapun.")

        db_payload = {**data, "bed_id": "WAITING", "zone": "-"}
        saved_to_db = save_patient(db_payload)

        return jsonify({
            "status": "warning",
            "message": "Semua bed penuh — pasien masuk daftar tunggu, data tetap dicatat.",
            "assigned_bed": None,
            "saved_to_database": saved_to_db,
        }), 200

    updated_bed_info = bed_manager.get_all_beds_status()[assigned_bed_id]

    # 2. SIMPAN KE MYSQL
    # Zona diambil dari bed yang BENAR-BENAR dialokasikan (bisa berbeda dari
    # bed_id yang diminta kalau bed itu ternyata sudah terisi).
    db_payload = {**data, "bed_id": assigned_bed_id, "zone": updated_bed_info["zone"]}
    saved_to_db = save_patient(db_payload)
    if not saved_to_db:
        # Kegagalan simpan ke DB tidak boleh menghentikan update dashboard real-time,
        # tapi harus tercatat di log server.
        print(f"[SERVER LOG][WARN] Data bed {assigned_bed_id} GAGAL disimpan ke MySQL.", flush=True)

    # 3. PANCARKAN EVENT WEBSOCKET KE FRONTEND WEB SECARA INSTAN
    socketio.emit("bed_updated", updated_bed_info)

    print(f"[SERVER LOG] Berhasil memperbarui {assigned_bed_id} dan memancarkan data via WebSocket.\n", flush=True)

    return jsonify({
        "status": "success",
        "message": f"Data berhasil diupdate pada bed {assigned_bed_id}",
        "assigned_bed": assigned_bed_id,
        "saved_to_database": saved_to_db,
        "data": updated_bed_info,
    }), 200


# =========================================================================
# 3. ENDPOINT API STATUS SELURUH BED (INITIAL LOAD)
# =========================================================================
@app.route("/api/beds", methods=["GET"])
def get_all_beds():
    """Mengembalikan status seluruh bed (Zona A, B, C) untuk inisialisasi web."""
    return jsonify(bed_manager.get_all_beds_status())


@app.route("/api/beds/reset", methods=["POST"])
def reset_all_beds():
    """Kosongkan SEMUA bed sekaligus. Endpoint khusus untuk testing/development
    supaya tidak perlu discharge satu-satu atau restart Flask berulang kali."""
    updated_beds = bed_manager.reset_all_beds()
    socketio.emit("beds_reset", updated_beds)
    print("[SERVER LOG] Semua bed telah direset ke status kosong.")
    return jsonify({"status": "success", "data": updated_beds}), 200


@app.route("/api/beds/<bed_id>/discharge", methods=["POST"])
def discharge_bed(bed_id):
    """Mengosongkan kembali sebuah bed, mis. setelah pasien dipindahkan/selesai ditangani."""
    success = bed_manager.discharge_bed(bed_id)
    if not success:
        return jsonify({"status": "error", "message": f"Bed {bed_id} tidak ditemukan"}), 404

    updated_bed_info = bed_manager.get_all_beds_status()[bed_id]
    socketio.emit("bed_updated", updated_bed_info)
    print(f"[SERVER LOG] Bed {bed_id} telah dikosongkan kembali.")

    return jsonify({"status": "success", "bed_id": bed_id, "data": updated_bed_info}), 200


# =========================================================================
# 4. ENDPOINT RIWAYAT PASIEN (OPSIONAL, dari MySQL)
# =========================================================================
@app.route("/api/history", methods=["GET"])
def get_history():
    limit = request.args.get("limit", default=50, type=int)
    return jsonify(get_patient_history(limit=limit))


# =========================================================================
# 5. WEBSOCKET EVENTS
# =========================================================================
@socketio.on("connect")
def handle_connect():
    print("[SOCKET LOG] Client Web Dashboard terhubung.")


@socketio.on("disconnect")
def handle_disconnect():
    print("[SOCKET LOG] Client Web Dashboard terputus.")


# =========================================================================
# MAIN EXECUTION
# =========================================================================
if __name__ == "__main__":
    print("=== Menjalankan Server IGD Command Center ===")
    print(f"Static directory: {WEBSITE_DIR}")
    print(f"Listen address : http://{SERVER_HOST}:{SERVER_PORT}")
    print(
        "Akses laptop  : http://<IP-RASPBERRY-PI>:"
        f"{SERVER_PORT}"
    )
    print("Cari IP Raspi  : hostname -I\n")

    socketio.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        debug=DEBUG_MODE,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )