import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from bed_manager import BedManager

# Inisialisasi Flask App & Arahkan static folder ke folder 'website'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEBSITE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "website"))

app = Flask(__name__, static_folder=WEBSITE_DIR, static_url_path="")
CORS(app)  # Izinkan CORS agar Live Server atau PyQt6 bisa terhubung tanpa hambatan

# Inisialisasi SocketIO untuk komunikasi dua arah (Real-time)
socketio = SocketIO(app, cors_allowed_origins="*")

# Inisialisasi BedManager untuk mengelola status bed IGD
bed_manager = BedManager()


# =========================================================================
# 1. ROUTE WEB DASHBOARD
# =========================================================================
@app.route("/")
def index():
    """Melayani halaman utama index.html dari folder website."""
    return send_from_directory(WEBSITE_DIR, "index.html")


# =========================================================================
# 2. ENDPOINT API UNTUK PENGIRIMAN DATA DARI GUI / HARDWARE
# =========================================================================
@app.route("/api/triage/update", methods=["POST"])
def update_triage():
    """
    Endpoint yang dipanggil oleh TriageApiClient di output_page.py.
    Menerima JSON payload 6 parameter vital sign & hasil klasifikasi.
    """
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "Payload JSON tidak ditemukan"}), 400

    print(f"\n[SERVER LOG] Menerima data baru dari hardware/GUI:")
    print(f" -> Bed ID Target : {data.get('bed_id')}")
    print(f" -> Kategori Triase: {data.get('triage_category')}")
    print(f" -> Pasien         : {data.get('patient_name')}")

    # 1. Masukkan/Update status pasien pada BedManager
    assigned_bed_id = bed_manager.assign_patient_to_bed(data)
    
    if assigned_bed_id:
        updated_bed_info = bed_manager.get_all_beds_status()[assigned_bed_id]
        
        # 2. PANCARKAN EVENT WEBSOCKET KE FRONTEND WEB SECARA INSTAN
        socketio.emit("bed_updated", updated_bed_info)
        
        print(f"[SERVER LOG] Berhasil memperbarui {assigned_bed_id} dan memancarkan data via WebSocket.\n")
        
        return jsonify({
            "status": "success", 
            "message": f"Data berhasil diupdate pada bed {assigned_bed_id}",
            "assigned_bed": assigned_bed_id,
            "data": updated_bed_info
        }), 200
    
    return jsonify({"status": "error", "message": "Gagal mengalokasikan bed"}), 500


# =========================================================================
# 3. ENDPOINT API STATUS SELURUH BED (INITIAL LOAD)
# =========================================================================
@app.route("/api/beds", methods=["GET"])
def get_all_beds():
    """Mengembalikan status seluruh bed (Zona A, B, C) untuk inisialisasi web."""
    return jsonify(bed_manager.get_all_beds_status())


# =========================================================================
# 4. WEBSOCKET EVENTS
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
    print(f"=== Menjalankan Server IGD Command Center ===")
    print(f"Static directory: {WEBSITE_DIR}")
    print(f"Server berjalan di http://127.0.0.1:5000\n")
    
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)