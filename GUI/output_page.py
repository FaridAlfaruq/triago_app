import sys
import os
import json
import numpy as np
import pyqtgraph as pg
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QGridLayout, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap

# Konfigurasi Global Tema PyQtGraph (Background Putih & Teks Gelap)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', '#214889')


class OutputPage(QWidget):
    # Sinyal untuk kembali ke halaman utama / registrasi
    home_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.patient_data = {}
        self.calculation_results = {}
        self.iot_json_payload = ""
        self.setup_ui()

    def setup_ui(self):
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background-color: #F6FFEC;")
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 20, 40, 20)
        main_layout.setSpacing(15)

        # =========================================================================
        # 1. HEADER: JUDUL, INDIKATOR TRIASE, & LOGO
        # =========================================================================
        header_layout = QHBoxLayout()
        header_layout.setSpacing(20)
        header_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Kiri: Judul Halaman
        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(4)
        lbl_title = QLabel("HASIL PENGECEKAN")
        lbl_title.setStyleSheet("font-size: 32px; font-weight: 900; color: #214889; background: transparent;")
        lbl_subtitle = QLabel("Output parameter dan hasil klasifikasi ML")
        lbl_subtitle.setStyleSheet("font-size: 22px; font-weight: 500; color: #555555; background: transparent;")
        title_vbox.addWidget(lbl_title)
        title_vbox.addWidget(lbl_subtitle)
        header_layout.addLayout(title_vbox)

        # Tengah: Indikator Triase
        self.triage_container = QHBoxLayout()
        self.triage_container.setSpacing(10)
        
        self.badge_color = QFrame()
        self.badge_color.setFixedSize(70, 70)
        self.badge_color.setStyleSheet("border-radius: 8px; background-color: #FF5252;")
        
        self.lbl_status_text = QLabel("RESUSITASI")
        self.lbl_status_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status_text.setFixedHeight(65)
        self.lbl_status_text.setMinimumWidth(120)
        self.lbl_status_text.setStyleSheet("""
            font-size: 32px; font-weight: 900; color: #FFFFFF; 
            background-color: #FF8A8A; border-radius: 8px; 
            padding-left: 20px; padding-right: 20px;
        """)
        
        self.triage_container.addWidget(self.badge_color)
        self.triage_container.addWidget(self.lbl_status_text, stretch=1)
        header_layout.addLayout(self.triage_container, stretch=1)

        # Kanan: Logo TriaGO
        lbl_logo = QLabel()
        lbl_logo.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl_logo.setStyleSheet("background: transparent;")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.abspath(os.path.join(current_dir, "..", "asset", "logo.png"))
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            lbl_logo.setPixmap(pixmap.scaledToWidth(220, Qt.TransformationMode.SmoothTransformation))
        else:
            lbl_logo.setText("TriaGO")
            lbl_logo.setStyleSheet("font-size: 32px; font-weight: 900; color: #214889;")
        header_layout.addWidget(lbl_logo)

        main_layout.addLayout(header_layout)

        # =========================================================================
        # 2. BODY LAYOUT: 4 BOX UTAMA (SHAP, ECG, PARAMETER, PPG IR)
        # =========================================================================
        lbl_shap_title = QLabel("SHAP Analysis")
        lbl_shap_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #214889; background: transparent;")
        
        lbl_ecg_title = QLabel("Sinyal ECG (5 Detik)")
        lbl_ecg_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #214889; background: transparent;")
        
        lbl_param_title = QLabel("HASIL PARAMETER")
        lbl_param_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #214889; background: transparent;")
        
        lbl_ppg_title = QLabel("Sinyal PPG IR (5 Detik)")
        lbl_ppg_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #214889; background: transparent;")

        # --- A. KOTAK SHAP ---
        self.box_shap = QFrame()
        self.box_shap.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        shap_layout = QVBoxLayout(self.box_shap)
        shap_layout.setContentsMargins(10, 10, 10, 10)
        
        self.plot_shap = pg.PlotWidget()
        self.plot_shap.showGrid(x=True, y=False, alpha=0.2)
        self.plot_shap.setLabel('bottom', 'SHAP Value (Dampak Fitur)', color='#555555')
        shap_layout.addWidget(self.plot_shap)

        # --- B. KOTAK ECG ---
        self.box_ecg = QFrame()
        self.box_ecg.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        ecg_layout = QVBoxLayout(self.box_ecg)
        ecg_layout.setContentsMargins(10, 10, 10, 10)

        self.plot_ecg = pg.PlotWidget()
        self.plot_ecg.showGrid(x=True, y=True, alpha=0.2)
        self.plot_ecg.setLabel('bottom', 'Waktu (detik)', color='#555555')
        self.plot_ecg.setLabel('left', 'Amplitudo (mV)', color='#555555')
        ecg_layout.addWidget(self.plot_ecg)

        # --- C. KOTAK PARAMETER MEDIS (5 PARAMETER) ---
        self.box_parameter = QFrame()
        self.box_parameter.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        param_layout = QGridLayout(self.box_parameter)
        param_layout.setContentsMargins(15, 10, 15, 10)
        param_layout.setSpacing(10)

        # 5 Kartu Parameter
        self.lbl_temp_val, self.lbl_temp_sub = self._create_param_card(param_layout, "Suhu Tubuh", "-- °C", 0, 0)
        self.lbl_hr_val, _ = self._create_param_card(param_layout, "Denyut Jantung", "-- BPM", 0, 1)
        self.lbl_rr_val, _ = self._create_param_card(param_layout, "Laju Pernapasan", "-- RPM", 1, 0)
        self.lbl_spo2_val, _ = self._create_param_card(param_layout, "Saturasi Oksigen", "-- %", 1, 1)
        self.lbl_bp_val, _ = self._create_param_card(param_layout, "Tekanan Darah", "--/-- mmHg", 2, 0, colspan=2)

        # --- D. KOTAK PPG IR (TANPA LEGEND) ---
        self.box_ppg = QFrame()
        self.box_ppg.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        ppg_layout = QVBoxLayout(self.box_ppg)
        ppg_layout.setContentsMargins(10, 10, 10, 10)

        self.plot_ppg = pg.PlotWidget()
        self.plot_ppg.showGrid(x=True, y=True, alpha=0.2)
        self.plot_ppg.setLabel('bottom', 'Waktu (detik)', color='#555555')
        self.plot_ppg.setLabel('left', 'Amplitudo (a.u.)', color='#555555')
        ppg_layout.addWidget(self.plot_ppg)

        # --- BARIS 1 (SHAP + ECG) ---
        top_row_layout = QHBoxLayout()
        top_row_layout.setSpacing(20)
        
        shap_cell = QVBoxLayout()
        shap_cell.setSpacing(8)
        shap_cell.addWidget(lbl_shap_title)
        shap_cell.addWidget(self.box_shap, stretch=1)
        
        ecg_cell = QVBoxLayout()
        ecg_cell.setSpacing(8)
        ecg_cell.addWidget(lbl_ecg_title)
        ecg_cell.addWidget(self.box_ecg, stretch=1)
        
        top_row_layout.addLayout(shap_cell, stretch=1)
        top_row_layout.addLayout(ecg_cell, stretch=1)
        main_layout.addLayout(top_row_layout, stretch=1)

        # --- BARIS 2 (PARAMETER + PPG) ---
        bottom_row_layout = QHBoxLayout()
        bottom_row_layout.setSpacing(20)
        
        param_cell = QVBoxLayout()
        param_cell.setSpacing(8)
        param_cell.addWidget(lbl_param_title)
        param_cell.addWidget(self.box_parameter, stretch=1)
        
        ppg_cell = QVBoxLayout()
        ppg_cell.setSpacing(8)
        ppg_cell.addWidget(lbl_ppg_title)
        ppg_cell.addWidget(self.box_ppg, stretch=1)
        
        bottom_row_layout.addLayout(param_cell, stretch=1)
        bottom_row_layout.addLayout(ppg_cell, stretch=1)
        main_layout.addLayout(bottom_row_layout, stretch=1)

        # =========================================================================
        # 3. FOOTER: BUTTON KEMBALI
        # =========================================================================
        self.btn_home = QPushButton("KEMBALI")
        self.btn_home.setFixedHeight(50)
        self.btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_home.setStyleSheet("""
            QPushButton { 
                background-color: #214889; 
                color: white; 
                font-size: 18px; 
                font-weight: bold; 
                border-radius: 8px; 
            }
            QPushButton:hover { background-color: #183563; }
            QPushButton:pressed { background-color: #0F2240; }
        """)
        self.btn_home.clicked.connect(self.handle_home_click)
        main_layout.addWidget(self.btn_home)

    def _create_param_card(self, grid_layout, title, default_val, row, col, colspan=1):
        """Helper untuk membuat tampilan kartu parameter medis."""
        card = QFrame()
        card.setStyleSheet("QFrame { background-color: #F8FAF6; border: 1px solid #D5E5D0; border-radius: 8px; }")
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(10, 6, 10, 6)
        vbox.setSpacing(1)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #555555; border: none; background: transparent;")
        
        lbl_val = QLabel(default_val)
        lbl_val.setStyleSheet("font-size: 20px; font-weight: 900; color: #214889; border: none; background: transparent;")
        
        lbl_sub = QLabel("")
        lbl_sub.setStyleSheet("font-size: 11px; font-weight: 600; color: #778899; border: none; background: transparent;")
        
        vbox.addWidget(lbl_title)
        vbox.addWidget(lbl_val)
        vbox.addWidget(lbl_sub)
        
        grid_layout.addWidget(card, row, col, 1, colspan)
        return lbl_val, lbl_sub

    # =========================================================================
    # FUNGSI UTAMA: MENERIMA DATA, PLOTTING, & IOT JSON PREPARATION
    # =========================================================================
    def update_results(self, data):
        """Dipanggil dari MainGUI untuk mengisi parameter medis, plot sinyal 5s, SHAP, dan payload IoT."""
        self.calculation_results = data

        # 1. Update Parameter Suhu (Core, Skin, Burton, Ambient)
        temp_core = data.get("temperature", 36.5)
        temp_skin = data.get("temp_skin", 34.5)
        temp_burton = data.get("temp_burton", 35.8)
        temp_amb = data.get("temp_ambient", 28.0)

        hr = data.get("hr", 0.0)
        rr = data.get("rr", 0.0)
        spo2 = data.get("spo2", 0.0)
        sys_bp = data.get("systolic", 120)
        dia_bp = data.get("diastolic", 80)

        # Menampilkan Suhu Inti sebagai Angka Utama & Rincian Burton/Kulit di Bawahnya
        self.lbl_temp_val.setText(f"{temp_core:.1f} °C")
        self.lbl_temp_sub.setText(f"Kulit: {temp_skin:.1f}°C | Tb (Burton): {temp_burton:.1f}°C")

        self.lbl_hr_val.setText(f"{hr:.1f} BPM")
        self.lbl_rr_val.setText(f"{rr:.1f} RPM")
        self.lbl_spo2_val.setText(f"{spo2:.1f} %")
        self.lbl_bp_val.setText(f"{int(sys_bp)}/{int(dia_bp)} mmHg")

        # 2. Potong Sinyal Menjadi 5 Detik Pertama (fs = 125 Hz -> 625 samples)
        fs = 125
        max_samples = 5 * fs

        time_x = data.get("time_125", np.array([]))
        ecg_y = data.get("ecg_smooth", np.array([]))
        ir_y = data.get("ir_clean", np.array([]))

        # Slicing array 5 detik
        time_5s = time_x[:max_samples] if len(time_x) >= max_samples else time_x
        ecg_5s = ecg_y[:max_samples] if len(ecg_y) >= max_samples else ecg_y
        ir_5s = ir_y[:max_samples] if len(ir_y) >= max_samples else ir_y

        if len(time_5s) == 0 and len(ecg_5s) > 0:
            time_5s = np.linspace(0, 5, len(ecg_5s))

        # 3. Plot Sinyal ECG (5 Detik)
        self.plot_ecg.clear()
        if len(ecg_5s) > 0:
            self.plot_ecg.plot(time_5s, ecg_5s, pen=pg.mkPen('#214889', width=2))
            self.plot_ecg.setXRange(0, 5)

        # 4. Plot Sinyal PPG IR Saja (5 Detik, Tanpa Legend)
        self.plot_ppg.clear()
        if len(ir_5s) > 0:
            self.plot_ppg.plot(time_5s, ir_5s, pen=pg.mkPen('#2980B9', width=2))
            self.plot_ppg.setXRange(0, 5)

        # 5. Render Grafik SHAP Analysis Nyata
        shap_features = data.get("shap_features", ["GCS", "SpO2", "HR", "RR", "Suhu"])
        shap_values = data.get("shap_values", [0.0, 0.0, 0.0, 0.0, 0.0])
        self._render_real_shap(shap_features, shap_values)

        # 6. Persiapkan JSON Payload untuk IoT (6 Parameter Utama)
        self.iot_json_payload = self.prepare_iot_payload(data)

    def _render_real_shap(self, features, shap_values):
        """Membuat grafik horizontal bar SHAP yang teratur dan rapi."""
        self.plot_shap.clear()

        # Matikan SI Prefix otomatis (x0.001) agar angka sumbu X murni
        self.plot_shap.getAxis('bottom').enableAutoSIPrefix(False)

        # Cek jika data kosong atau bernilai 0 semua
        shap_array = np.array(shap_values)
        if len(features) == 0 or len(shap_array) == 0 or np.all(shap_array == 0):
            return

        # Mapping nama teknis ke nama klinis sederhana
        name_mapping = {
            'temperature_c': 'Suhu',
            'spo2': 'SpO2',
            'respiratory_rate': 'Laju Nafas',
            'heart_rate': 'Heart Rate',
            'systolic_bp': 'Sistolik',
            'diastolic_bp': 'Diastolik',
            'gcs_total': 'Skor GCS',
            'shock_index': 'Shock Index',
            'map': 'MAP',
            'pulse_pressure': 'Pulse Press.',
            'hypoxia': 'Hipoksia',
            'tachypnea': 'Takipnea',
            'abnormal_temp': 'Suhu Abn.',
            'abnormal_hr': 'HR Abn.',
            'gcs_squared': 'GCS²',
            'gcs_map_index': 'GCS-MAP',
            'gcs_shock_index': 'GCS-SI',
            'total_abnormal': 'Tot. Abnormal'
        }

        # Ambil Top 7 Fitur dengan Dampak SHAP Terbesar (Mencegah teks bertumpuk)
        abs_vals = np.abs(shap_array)
        top_indices = np.argsort(abs_vals)[-7:] 
        
        top_features = [features[i] for i in top_indices]
        top_values = [shap_array[i] for i in top_indices]

        labels = [name_mapping.get(f, f) for f in top_features]
        y_pos = np.arange(len(top_features))

        # Render Horizontal Bar (Hijau = Dampak Positif, Merah = Dampak Negatif)
        for y, val in zip(y_pos, top_values):
            color = '#E74C3C' if val < 0 else '#2ECC71'
            x0 = min(0.0, float(val))
            x1 = max(0.0, float(val))
            
            bar = pg.BarGraphItem(
                x0=x0, x1=x1,
                y=float(y),
                height=0.45,
                brush=pg.mkBrush(color),
                pen=pg.mkPen(color)
            )
            self.plot_shap.addItem(bar)

        # Atur Ticks Sumbu Y & Kunci Rentang Tampilan
        axis_y = self.plot_shap.getAxis('left')
        ticks = [list(zip(y_pos, labels))]
        axis_y.setTicks(ticks)
        
        self.plot_shap.setYRange(-0.8, len(top_features) - 0.2)
        self.plot_shap.enableAutoRange(axis='x')

    def prepare_iot_payload(self, data):
        """Menyiapkan struktur JSON parameter medis utama untuk IoT."""
        payload = {
            "device_id": f"TRIAGO_BED_{data.get('bed', '01')}",
            "timestamp": data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "telemetry": {
                "suhu_inti_c": data.get("temperature", 36.5),
                "suhu_kulit_c": data.get("temp_skin", 34.5),
                "suhu_burton_c": data.get("temp_burton", 35.8),
                "suhu_lingkungan_c": data.get("temp_ambient", 28.0),
                "saturasi_oksigen_pct": data.get("spo2", 98.0),
                "laju_pernapasan_rpm": data.get("rr", 16.0),
                "denyut_jantung_bpm": data.get("hr", 75.0),
                "gcs_score": data.get("gcs", 15),
                "tekanan_darah": {
                    "sistol_mmhg": data.get("systolic", 120),
                    "diastol_mmhg": data.get("diastolic", 80)
                }
            },
            "triage_status": data.get("triage_status", "NON-DARURAT")
        }
        
        json_str = json.dumps(payload, indent=4)
        return json_str

    def update_triage_header(self, status):
        """Fungsi dinamis untuk mengubah warna header sesuai hasil klasifikasi."""
        status = status.upper()
        if status == "RESUSITASI":
            self.badge_color.setStyleSheet("border-radius: 8px; background-color: #E74C3C;")
            self.lbl_status_text.setText("RESUSITASI")
            self.lbl_status_text.setStyleSheet("font-size: 32px; font-weight: 900; background-color: #FADBD8; border-radius: 8px; padding-left: 20px; padding-right: 20px; color: #E74C3C;")
        elif status == "DARURAT":
            self.badge_color.setStyleSheet("border-radius: 8px; background-color: #F39C12;")
            self.lbl_status_text.setText("DARURAT")
            self.lbl_status_text.setStyleSheet("font-size: 32px; font-weight: 900; background-color: #FDEBD0; border-radius: 8px; padding-left: 20px; padding-right: 20px; color: #F39C12;")
        elif "NON" in status or status == "HIJAU":
            self.badge_color.setStyleSheet("border-radius: 8px; background-color: #2ECC71;")
            self.lbl_status_text.setText("NON-DARURAT")
            self.lbl_status_text.setStyleSheet("font-size: 32px; font-weight: 900; background-color: #D5F5E3; border-radius: 8px; padding-left: 20px; padding-right: 20px; color: #2ECC71;")

    def handle_home_click(self):
        print("[LOG] Inputs cleared. Returning to home_page...")
        self.home_requested.emit()


# =========================================================================
# BLOK PENGETESAN MANDIRI (LOCAL TESTING BLOCK)
# =========================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    test_window = OutputPage()
    test_window.setWindowTitle("TriaGO - Test Output Pengecekan")
    test_window.showMaximized()
    
    # Dummy Data 10 Detik
    fs = 125
    t_dummy = np.linspace(0, 10, 10 * fs)
    ecg_dummy = np.sin(2 * np.pi * 1.5 * t_dummy) + 0.2 * np.random.normal(size=len(t_dummy))
    ir_dummy = 1.2 + 0.4 * np.sin(2 * np.pi * 1.5 * t_dummy)

    dummy_results = {
        "bed": "02",
        "gcs": 15,
        "timestamp": "2026-07-24 17:27:44",
        "temperature": 36.5,
        "hr": 110.5,
        "rr": 16.0,
        "spo2": 98.2,
        "systolic": 120,
        "diastolic": 80,
        "time_125": t_dummy,
        "ecg_smooth": ecg_dummy,
        "ir_clean": ir_dummy,
        "triage_status": "DARURAT",
        "shap_features": ["gcs_total", "systolic_bp", "spo2", "heart_rate", "temperature_c"],
        "shap_values": [0.35, -0.22, 0.18, -0.12, 0.05]
    }

    test_window.update_results(dummy_results)
    test_window.update_triage_header(dummy_results["triage_status"]) 
    
    sys.exit(app.exec())