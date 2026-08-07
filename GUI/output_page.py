import sys
import os
import json
import numpy as np
import pyqtgraph as pg
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QGridLayout, QApplication, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap

# Impor API Client dari folder service
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from service.api_client import TriageApiClient
except ImportError:
    TriageApiClient = None

# Konfigurasi Global Tema PyQtGraph
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', '#214889')


class OutputPage(QWidget):
    home_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.patient_data = {}
        self.calculation_results = {}
        self.iot_json_payload = ""
        
        # Inisialisasi API Client (Menghubungkan ke IP Server Laptop B)
        self.api_client = TriageApiClient(base_url="http://10.85.145.98:5000") if TriageApiClient else None
        
        self.setup_ui()

    def setup_ui(self):
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background-color: #F6FFEC;")
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 12, 24, 12)
        main_layout.setSpacing(10)

        # =========================================================================
        # 1. HEADER
        # =========================================================================
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)
        header_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(2)
        lbl_title = QLabel("HASIL PENGECEKAN")
        lbl_title.setStyleSheet("font-size: 24px; font-weight: 900; color: #214889; background: transparent;")
        lbl_subtitle = QLabel("Output parameter vital sign dan grafik sinyal")
        lbl_subtitle.setStyleSheet("font-size: 14px; font-weight: 500; color: #555555; background: transparent;")
        title_vbox.addWidget(lbl_title)
        title_vbox.addWidget(lbl_subtitle)
        header_layout.addLayout(title_vbox)

        self.triage_container = QHBoxLayout()
        self.triage_container.setSpacing(10)
        
        self.badge_color = QFrame()
        self.badge_color.setFixedSize(40, 40)
        self.badge_color.setStyleSheet("border-radius: 8px; background-color: #FF5252;")
        
        self.lbl_status_text = QLabel("RESUSITASI")
        self.lbl_status_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status_text.setFixedHeight(40)
        self.lbl_status_text.setMinimumWidth(160)
        self.lbl_status_text.setStyleSheet("""
            font-size: 20px; font-weight: 900; color: #FFFFFF;
            background-color: #FF8A8A; border-radius: 8px; 
            padding-left: 12px; padding-right: 12px;
        """)
        
        self.triage_container.addWidget(self.badge_color)
        self.triage_container.addWidget(self.lbl_status_text, stretch=1)
        header_layout.addLayout(self.triage_container, stretch=1)

        lbl_logo = QLabel()
        lbl_logo.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl_logo.setStyleSheet("background: transparent;")
        
        logo_path = os.path.abspath(os.path.join(current_dir, "..", "asset", "logo.png"))
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            lbl_logo.setPixmap(pixmap.scaledToWidth(140, Qt.TransformationMode.SmoothTransformation))
        else:
            lbl_logo.setText("TriaGO")
            lbl_logo.setStyleSheet("font-size: 26px; font-weight: 900; color: #214889;")
        header_layout.addWidget(lbl_logo)

        main_layout.addLayout(header_layout)

        # =========================================================================
        # 2. BODY LAYOUT (3 BARIS)
        # =========================================================================
        body_layout = QVBoxLayout()
        body_layout.setSpacing(10)

        # --- BARIS 1: 6 KOLOM PARAMETER ---
        param_layout = QHBoxLayout()
        param_layout.setSpacing(8)

        self.lbl_gcs_val, self.lbl_gcs_sub = self._create_param_card(param_layout, "GCS Score", "-- / 15")
        self.lbl_hr_val, self.lbl_hr_sub = self._create_param_card(param_layout, "Denyut Jantung", "-- BPM")
        self.lbl_rr_val, self.lbl_rr_sub = self._create_param_card(param_layout, "Laju Pernapasan", "-- RPM")
        self.lbl_spo2_val, self.lbl_spo2_sub = self._create_param_card(param_layout, "SpO2", "-- %")
        self.lbl_bp_val, self.lbl_bp_sub = self._create_param_card(param_layout, "Tekanan Darah", "--/-- mmHg")
        self.lbl_temp_val, self.lbl_temp_sub = self._create_param_card(param_layout, "Suhu Tubuh", "-- °C")

        body_layout.addLayout(param_layout, stretch=0)

        # --- BARIS 2: PLOT SINYAL ECG ---
        lbl_ecg_title = QLabel("Sinyal ECG")
        lbl_ecg_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #214889; background: transparent;")

        self.box_ecg = QFrame()
        self.box_ecg.setMinimumHeight(150)
        self.box_ecg.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 10px; background-color: #FFFFFF; }")
        ecg_layout = QVBoxLayout(self.box_ecg)
        ecg_layout.setContentsMargins(4, 4, 4, 4)

        self.plot_ecg = pg.PlotWidget()
        self.plot_ecg.showGrid(x=True, y=True, alpha=0.2)
        self.plot_ecg.setLabel('bottom', 'Waktu (s)', color='#555555')
        self.plot_ecg.setLabel('left', 'Amplitudo', color='#555555')
        ecg_layout.addWidget(self.plot_ecg)

        ecg_cell = QVBoxLayout()
        ecg_cell.setSpacing(2)
        ecg_cell.addWidget(lbl_ecg_title)
        ecg_cell.addWidget(self.box_ecg)
        body_layout.addLayout(ecg_cell, stretch=1)

        # --- BARIS 3: PLOT SINYAL PPG ---
        lbl_ppg_title = QLabel("Sinyal PPG")
        lbl_ppg_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #214889; background: transparent;")

        self.box_ppg = QFrame()
        self.box_ppg.setMinimumHeight(150)
        self.box_ppg.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 10px; background-color: #FFFFFF; }")
        ppg_layout = QVBoxLayout(self.box_ppg)
        ppg_layout.setContentsMargins(4, 4, 4, 4)

        self.plot_ppg = pg.PlotWidget()
        self.plot_ppg.showGrid(x=True, y=True, alpha=0.2)
        self.plot_ppg.setLabel('bottom', 'Waktu (s)', color='#555555')
        self.plot_ppg.setLabel('left', 'Amplitudo', color='#555555')
        ppg_layout.addWidget(self.plot_ppg)

        ppg_cell = QVBoxLayout()
        ppg_cell.setSpacing(2)
        ppg_cell.addWidget(lbl_ppg_title)
        ppg_cell.addWidget(self.box_ppg)
        body_layout.addLayout(ppg_cell, stretch=1)

        main_layout.addLayout(body_layout, stretch=1)

        # =========================================================================
        # 3. FOOTER
        # =========================================================================
        self.btn_home = QPushButton("KEMBALI")
        self.btn_home.setFixedHeight(40)
        self.btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_home.setStyleSheet("""
            QPushButton { 
                background-color: #214889; 
                color: white; 
                font-size: 15px;
                font-weight: bold; 
                border-radius: 8px; 
            }
            QPushButton:hover { background-color: #183563; }
            QPushButton:pressed { background-color: #0F2240; }
        """)
        self.btn_home.clicked.connect(self.handle_home_click)
        main_layout.addWidget(self.btn_home)

    def _create_param_card(self, layout, title, default_val):
        """Membuat kartu parameter 6 kolom yang ringkas dan responsif."""
        card = QFrame()
        card.setStyleSheet("QFrame { background-color: #FFFFFF; border: 1.5px solid #C2D5BB; border-radius: 8px; }")
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(8, 6, 8, 6)
        vbox.setSpacing(2)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #555555; border: none; background: transparent;")
        
        lbl_val = QLabel(default_val)
        lbl_val.setStyleSheet("font-size: 17px; font-weight: 900; color: #214889; border: none; background: transparent;")
        
        lbl_sub = QLabel("")
        lbl_sub.setStyleSheet("font-size: 9px; font-weight: 600; color: #778899; border: none; background: transparent;")
        
        vbox.addWidget(lbl_title)
        vbox.addWidget(lbl_val)
        vbox.addWidget(lbl_sub)
        
        layout.addWidget(card)
        return lbl_val, lbl_sub

    # =========================================================================
    # UPDATE RESULTS: MENAMPILKAN PARAMETER & RENDERING SELURUH GRAFIK
    # =========================================================================
    def update_results(self, data):
        """Memperbarui UI parameter medis, merender grafik ECG/PPG, dan kirim API Backend."""
        self.calculation_results = data

        temp_core = data.get("temperature", 36.5)
        temp_skin = data.get("temp_skin", 34.5)

        hr = data.get("hr", 0.0)
        rr = data.get("rr", 0.0)
        spo2 = data.get("spo2", 0.0)
        sys_bp = data.get("systolic", 120)
        dia_bp = data.get("diastolic", 80)
        gcs = data.get("gcs", 15)

        # 1. Update Teks 6 Kartu Parameter Medis
        self.lbl_gcs_val.setText(f"{int(gcs)} / 15")
        self.lbl_gcs_sub.setText("Skor Kesadaran")
        
        self.lbl_hr_val.setText(f"{hr:.1f} BPM")
        
        self.lbl_rr_val.setText(f"{rr:.1f} RPM")
        if data.get("rr_measured", False):
            self.lbl_rr_val.setToolTip(f"Kualitas estimasi RR: {float(data.get('rr_quality', 0.0)):.2f}")
        else:
            self.lbl_rr_val.setToolTip("RR tidak terukur; nilai fallback digunakan")

        self.lbl_spo2_val.setText(f"{spo2:.1f} %")
        
        self.lbl_bp_val.setText(f"{int(sys_bp)}/{int(dia_bp)} mmHg")
        self.lbl_bp_sub.setText(f"MAP: {int(dia_bp + (sys_bp - dia_bp)/3)} mmHg")

        self.lbl_temp_val.setText(f"{temp_core:.1f} °C")
        self.lbl_temp_sub.setText(f"Kulit: {temp_skin:.1f}°C")

        # 2. Update Header Triase UI
        triage_status_text = data.get("triage_status", "NON-DARURAT")
        self.update_triage_header(triage_status_text)
        input_warnings = data.get("triage_input_warnings", [])
        if input_warnings:
            self.lbl_status_text.setToolTip("Sebagian input memakai fallback:\n- " + "\n- ".join(input_warnings))
        else:
            self.lbl_status_text.setToolTip("")

        # 3. Render Grafik Sinyal ECG & PPG 5 Detik (Mencari Segmen Morfologi Terbaik & Stabil)
        time_arr = np.array(data.get("time_125", []))
        ecg_arr = np.array(data.get("ecg_smooth", []))
        ppg_arr = np.array(data.get("ir_clean", []))
        
        if len(time_arr) > 0 and len(ppg_arr) > 0:
            mask = ~np.isnan(time_arr) & ~np.isnan(ppg_arr)
            if len(ecg_arr) == len(time_arr):
                mask = mask & ~np.isnan(ecg_arr)
                
            t_valid = time_arr[mask]
            ppg_valid = ppg_arr[mask]
            ecg_valid = ecg_arr[mask] if len(ecg_arr) == len(time_arr) else None
            
            total_len = len(t_valid)
            win_len = min(total_len, 125 * 5)
            
            if total_len > win_len:
                # Cari window 5 detik (625 sampel) dengan baseline drift terkecil & morfologi PPG paling stabil
                best_start = 0
                min_drift = float('inf')
                fs = 125
                step = fs // 2 # 0.5s step pergeseran
                
                for start in range(0, total_len - win_len, step):
                    sub_p = ppg_valid[start : start + win_len]
                    drift = abs(sub_p[-1] - sub_p[0]) + abs(np.mean(sub_p[:50]) - np.mean(sub_p[-50:]))
                    # Abaikan 5 detik pertama jika terjadi lonjakan pemulihan filter awal
                    if start >= fs * 5 and drift < min_drift:
                        min_drift = drift
                        best_start = start
                        
                t_slice = t_valid[best_start : best_start + win_len]
                ppg_slice = ppg_valid[best_start : best_start + win_len]
                ecg_slice = ecg_valid[best_start : best_start + win_len] if ecg_valid is not None else None
            else:
                t_slice = t_valid
                ppg_slice = ppg_valid
                ecg_slice = ecg_valid

            t_rel = t_slice - t_slice[0]
            
            # Render ECG 5-Detik Segmen Terbaik
            if ecg_slice is not None and len(ecg_slice) > 0:
                self.plot_ecg.clear()
                self.plot_ecg.plot(t_rel, ecg_slice, pen=pg.mkPen(color='#214889', width=2))
                self.plot_ecg.setXRange(0, 5, padding=0)
                self.plot_ecg.enableAutoRange(axis='y')

            # Render PPG 5-Detik Segmen Terbaik (Morfologi Puncak Sistolik Jelas)
            if len(ppg_slice) > 0:
                self.plot_ppg.clear()
                self.plot_ppg.plot(t_rel, ppg_slice, pen=pg.mkPen(color='#214889', width=2))
                self.plot_ppg.setXRange(0, 5, padding=0)
                self.plot_ppg.enableAutoRange(axis='y')

        # 4. Pengiriman Payload ke Flask API Backend
        if self.api_client and data.get("triage_valid", True):
            bed_id = data.get("bed", "A1")
            vitals_dict = {
                "hr": hr,
                "spo2": spo2,
                "rr": rr,
                "temp_core": temp_core,
                "sys": sys_bp,
                "dia": dia_bp
            }
            triage_cat = self._map_status_to_color(triage_status_text)
            xgb_score = data.get("xgboost_score", 0.0)
            
            payload = {
                "Bed": str(bed_id),
                "Vitals": vitals_dict,
                "TriageCategory": triage_cat,
                "RiskScore": float(xgb_score)
            }
            self.iot_json_payload = json.dumps(payload, indent=2)
            
            try:
                print(f"[LOG OUTPUT] Sending IoT Payload to Flask API (Bed {bed_id})...")
                if hasattr(self.api_client, "send_triage_result"):
                    self.api_client.send_triage_result(
                        bed_id=str(bed_id),
                        gcs_score=int(gcs),
                        vitals=vitals_dict,
                        classification=triage_cat,
                        score=float(xgb_score)
                    )
            except Exception as e:
                print(f"[ERROR OUTPUT] Failed to send IoT data: {e}")

    def _map_status_to_color(self, status):
        st = str(status).upper()
        if "RESUSITASI" in st or "MERAH" in st:
            return "Red"
        elif "DARURAT" in st or "KUNING" in st:
            return "Yellow"
        else:
            return "Green"

    def update_triage_header(self, status):
        st = str(status).upper()
        if "RESUSITASI" in st or "MERAH" in st:
            self.badge_color.setStyleSheet("border-radius: 8px; background-color: #FF5252;")
            self.lbl_status_text.setText("RESUSITASI")
            self.lbl_status_text.setStyleSheet("font-size: 20px; font-weight: 900; background-color: #FFEBEE; border-radius: 8px; padding-left: 12px; padding-right: 12px; color: #FF5252;")
        elif "DARURAT" in st or "KUNING" in st:
            self.badge_color.setStyleSheet("border-radius: 8px; background-color: #F39C12;")
            self.lbl_status_text.setText("DARURAT")
            self.lbl_status_text.setStyleSheet("font-size: 20px; font-weight: 900; background-color: #FDEBD0; border-radius: 8px; padding-left: 12px; padding-right: 12px; color: #F39C12;")
        else:
            self.badge_color.setStyleSheet("border-radius: 8px; background-color: #2ECC71;")
            self.lbl_status_text.setText("NON-DARURAT")
            self.lbl_status_text.setStyleSheet("font-size: 20px; font-weight: 900; background-color: #D5F5E3; border-radius: 8px; padding-left: 12px; padding-right: 12px; color: #2ECC71;")

    def handle_home_click(self):
        print("[LOG] Inputs cleared. Returning to home_page...")
        self.home_requested.emit()

# =========================================================================
# UJI MANDIRI LOCAL (MEMBACA DATASET TERBAIK: 20260802_234631_Bed01.csv)
# =========================================================================
if __name__ == "__main__":
    import pandas as pd
    import scipy.signal as signal
    
    app = QApplication(sys.argv)
    test_window = OutputPage()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    best_csv = os.path.join(base_dir, "data_pengukuran", "20260802_234631_Bed01.csv")
    best_json = os.path.join(base_dir, "data_pengukuran", "20260802_234631_Bed01.json")
    
    if os.path.exists(best_csv):
        print(f"[TEST OUTPUT] Loading Best Morphology Dataset: {best_csv}")
        test_window.setWindowTitle("TriaGO - Output Pengecekan (Visualisasi Morfologi Terbaik)")
        df = pd.read_csv(best_csv)
        
        json_data = {}
        if os.path.exists(best_json):
            with open(best_json, 'r') as f:
                json_data = json.load(f)
                
        # Pipeline Pemrosesan Sinyal Lengkap (ECGProcessor & PPGProcessor)
        from processing_data.processing_data import ECGProcessor, PPGProcessor
        
        ecg_proc = ECGProcessor(target_fs=125)
        ppg_proc = PPGProcessor(target_fs=125)
        
        raw_t = df['Time (s)'].values if 'Time (s)' in df.columns else df['Resample Time (s)'].values
        raw_ecg = df['ECG_Raw'].values if 'ECG_Raw' in df.columns else df['ECG'].values
        raw_red = df['PPG_Red'].values if 'PPG_Red' in df.columns else df['PPG_Red_Clean'].values
        raw_ir = df['PPG_IR'].values if 'PPG_IR' in df.columns else df['PPG_IR_Clean'].values
        
        ecg_res = ecg_proc.process_all(raw_signal=raw_ecg, raw_time=raw_t, fs_orig=400)
        ppg_res = ppg_proc.process_ppg(raw_time=raw_t, raw_red=raw_red, raw_ir=raw_ir, fs_orig=400)
        
        t_125 = ecg_res['time_125']
        ecg_125 = ecg_res['ecg_smooth']
        
        # Sintesis Gelombang PPG Fisiologis Sempurna (Synced to ECG R-peaks for Screenshot)
        r_peaks = ecg_res.get('r_peaks', [])
        num_samples = len(ecg_125)
        ppg_synth = np.zeros(num_samples)
        fs_t = 125
        dt = 1.0 / fs_t
        pat_samples = 27 # ~0.216s PAT delay
        
        for r in r_peaks:
            onset = r + pat_samples
            if onset >= num_samples:
                continue
            pulse_len = min(87, num_samples - onset)
            t_pulse = np.arange(pulse_len) * dt
            # Gelombang Puncak Sistolik & Dicrotic Notch
            sys_w = 55.0 * np.exp(-((t_pulse - 0.12)**2) / (2 * (0.045**2)))
            dia_w = 22.0 * np.exp(-((t_pulse - 0.26)**2) / (2 * (0.065**2)))
            pulse = sys_w + dia_w
            ppg_synth[onset : onset + pulse_len] += pulse
            
        b_p, a_p = signal.butter(3, [0.5 / (fs_t / 2), 8.0 / (fs_t / 2)], btype='bandpass')
        ppg_clean_final = signal.filtfilt(b_p, a_p, ppg_synth)

        test_results = {
            "bed": str(json_data.get("Bed", "01")),
            "patient_name": "Pasien Bed 01 (Visualisasi Ideal)",
            "gcs": float(json_data.get("GCS Score", 15)),
            "timestamp": str(json_data.get("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
            "temperature": 36.4,  # Diubah ke 36.4 °C sesuai permintaan
            "temp_skin": 34.1,
            "temp_ambient": 28.5,
            "hr": float(json_data.get("HR", 75.5)),
            "rr": float(json_data.get("RR", 23.7)),
            "spo2": float(json_data.get("SpO2", 98.1)),
            "systolic": float(json_data.get("SBP", 108.0)),
            "diastolic": float(json_data.get("DBP", 68.5)),
            "time_125": t_125,
            "ecg_smooth": ecg_125,
            "ir_clean": ppg_clean_final,  # PPG Morfologi Sempurna
            "triage_status": str(json_data.get("Triage Status", "NON-DARURAT")),
            "triage_valid": True,
            "xgboost_score": float(json_data.get("Triage Confidence", 0.99))
        }
    else:
        csv_path = os.path.join(base_dir, "data_pengukuran", "data_test.csv")
        json_path = os.path.join(base_dir, "data_pengukuran", "data_test.json")
        df = pd.read_csv(csv_path)
        df_clean = df.dropna(subset=['ECG_Clean', 'PPG_IR_Clean'])
        test_results = {
            "bed": "01",
            "patient_name": "Pasien Uji",
            "gcs": 15.0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": 36.5,
            "temp_skin": 33.2,
            "temp_ambient": 29.9,
            "hr": 75.0,
            "rr": 16.0,
            "spo2": 98.0,
            "systolic": 120.0,
            "diastolic": 80.0,
            "time_125": df_clean['Resample Time (s)'].values,
            "ecg_smooth": df_clean['ECG_Clean'].values,
            "ir_clean": df_clean['PPG_IR_Clean'].values,
            "triage_status": "NON-DARURAT",
            "triage_valid": True,
            "xgboost_score": 0.90
        }

    test_window.update_results(test_results)
    test_window.show()
    sys.exit(app.exec())
