import os
import sys
import logging

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.config import load_config
from backend.access_control import DashboardLogin, dashboard_credentials

load_config()

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

try:
    from .bed_manager import BedManager
    from .patient_details import validate_patient_details
    from .database import save_patient, get_patient_history, update_patient_info, DB_BACKEND
    from .triage_service import TriageService
except ImportError:
    # Tetap mendukung eksekusi langsung: python backend/app.py
    from bed_manager import BedManager
    from patient_details import validate_patient_details
    from database import save_patient, get_patient_history, update_patient_info, DB_BACKEND
    from triage_service import TriageService

# Inisialisasi Flask App & Arahkan static folder ke folder 'website'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEBSITE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "website"))
SERVER_HOST = os.environ.get("TRIAGO_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("PORT", os.environ.get("TRIAGO_PORT", "5000")))
DEBUG_MODE = os.environ.get("TRIAGO_DEBUG", "0").lower() in {"1", "true", "yes"}
CORS_ORIGINS = os.environ.get("TRIAGO_CORS_ORIGINS") or None
LOGIN_CREDENTIALS = dashboard_credentials()

app = Flask(__name__, static_folder=WEBSITE_DIR, static_url_path="")
if CORS_ORIGINS:
    CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})

# Inisialisasi SocketIO untuk komunikasi dua arah (Real-time)
socketio = SocketIO(
    app,
    cors_allowed_origins=CORS_ORIGINS,
    async_mode="threading",
)
if LOGIN_CREDENTIALS:
    # Wrap outside SocketIO so /socket.io polling and upgrades require login too.
    app.wsgi_app = DashboardLogin(app.wsgi_app, *LOGIN_CREDENTIALS)

# Inisialisasi BedManager untuk mengelola status bed IGD
bed_manager = BedManager()
triage_service = TriageService(bed_manager, save_patient, socketio.emit)
mqtt_receiver = None
if os.environ.get("TRIAGO_MQTT_ENABLED", "0").lower() in {"1", "true", "yes"}:
    from backend.mqtt_receiver import MqttReceiver
    logging.basicConfig(level=logging.INFO)
    mqtt_receiver = MqttReceiver(triage_service.process)
    mqtt_receiver.start()


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
        "mqtt": mqtt_receiver.status() if mqtt_receiver else {"enabled": False},
        "database": {"engine": DB_BACKEND},
        "access": {"login_required": LOGIN_CREDENTIALS is not None},
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
    try:
        data = request.get_json(silent=True)
        return jsonify(triage_service.process(data)), 200
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as e:
        import traceback
        print(f"[SERVER ERROR] Terjadi kesalahan saat memproses data triage: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# =========================================================================
# 3. ENDPOINT API STATUS SELURUH BED (INITIAL LOAD)
# =========================================================================
@app.route("/api/beds", methods=["GET"])
def get_all_beds():
    """Mengembalikan status seluruh bed (01-12) untuk inisialisasi web."""
    return jsonify(bed_manager.get_all_beds_status())


@app.route("/api/beds/reset", methods=["POST"])
def reset_all_beds():
    """Kosongkan SEMUA bed sekaligus. Endpoint khusus untuk testing/development
    supaya tidak perlu discharge satu-satu atau restart Flask berulang kali."""
    updated_beds = bed_manager.reset_all_beds()
    socketio.emit("beds_reset", updated_beds)
    print("[SERVER LOG] Semua bed telah direset ke status kosong.")
    return jsonify({"status": "success", "data": updated_beds}), 200


@app.route("/api/beds/<bed_id>/patient", methods=["POST"])
def update_bed_patient(bed_id):
    """Memperbarui identitas dan data registrasi pasien pada bed tertentu."""
    data = request.get_json(silent=True) or {}
    patient_name = str(data.get("patient_name") or "").strip()[:100] or "Pasien Tanpa Nama"
    relative_name = str(data.get("relative_name") or "").strip()[:100] or "—"
    try:
        details = validate_patient_details(data.get("details"))
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    success = bed_manager.update_patient_identity(bed_id, patient_name, relative_name, details)
    if not success:
        return jsonify({"status": "error", "message": f"Bed {bed_id} tidak ditemukan"}), 404

    # Simpan/update ke MySQL database
    saved_to_db = update_patient_info(bed_id, patient_name, relative_name, details)

    updated_bed_info = bed_manager.get_all_beds_status()[bed_id]
    socketio.emit("bed_updated", updated_bed_info)
    print(f"[SERVER LOG] Identitas pasien Bed {bed_id} diperbarui: {patient_name} / {relative_name}", flush=True)

    return jsonify({
        "status": "success",
        "message": f"Data pasien Bed {bed_id} berhasil diperbarui",
        "saved_to_database": saved_to_db,
        "data": updated_bed_info
    }), 200


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
