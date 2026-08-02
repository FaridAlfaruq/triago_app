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
        main_layout.setSpacing(8)

        # =========================================================================
        # 1. HEADER
        # =========================================================================
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)
        header_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(2)
        lbl_title = QLabel("HASIL PENGECEKAN")
        lbl_title.setStyleSheet("font-size: 26px; font-weight: 900; color: #214889; background: transparent;")
        lbl_subtitle = QLabel("Output parameter dan hasil klasifikasi ML")
        lbl_subtitle.setStyleSheet("font-size: 15px; font-weight: 500; color: #555555; background: transparent;")
        title_vbox.addWidget(lbl_title)
        title_vbox.addWidget(lbl_subtitle)
        header_layout.addLayout(title_vbox)

        self.triage_container = QHBoxLayout()
        self.triage_container.setSpacing(10)
        
        self.badge_color = QFrame()
        self.badge_color.setFixedSize(48, 48)
        self.badge_color.setStyleSheet("border-radius: 8px; background-color: #FF5252;")
        
        self.lbl_status_text = QLabel("RESUSITASI")
        self.lbl_status_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status_text.setFixedHeight(48)
        self.lbl_status_text.setMinimumWidth(180)
        self.lbl_status_text.setStyleSheet("""
            font-size: 22px; font-weight: 900; color: #FFFFFF;
            background-color: #FF8A8A; border-radius: 8px; 
            padding-left: 12px; padding-right: 12px;
        """)
        
        self.triage_container.addWidget(self.badge_color)
        self.triage_container.addWidget(self.lbl_status_text, stretch=1)
        header_layout.addLayout(self.triage_container, stretch=1)

        lbl_logo = QLabel()
        lbl_logo.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl_logo.setStyleSheet("background: transparent;")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.abspath(os.path.join(current_dir, "..", "asset", "logo.png"))
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            lbl_logo.setPixmap(pixmap.scaledToWidth(160, Qt.TransformationMode.SmoothTransformation))
        else:
            lbl_logo.setText("TriaGO")
            lbl_logo.setStyleSheet("font-size: 28px; font-weight: 900; color: #214889;")
        header_layout.addWidget(lbl_logo)

        main_layout.addLayout(header_layout)

        # =========================================================================
        # 2. BODY LAYOUT (GRID 2x2 PROPORSI SEIMBANG & RESPONSUS)
        # =========================================================================
        content_grid = QGridLayout()
        content_grid.setSpacing(12)

        # A. KOTAK SHAP ANALYSIS (Baris 0, Kolom 0)
        lbl_shap_title = QLabel("SHAP Analysis")
        lbl_shap_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #214889; background: transparent;")
        
        self.box_shap = QFrame()
        self.box_shap.setMinimumHeight(160)  # Kunci tinggi minimum agar tidak tertekan gepeng
        self.box_shap.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        shap_layout = QVBoxLayout(self.box_shap)
        shap_layout.setContentsMargins(4, 4, 4, 4)
        
        self.plot_shap = pg.PlotWidget()
        self.plot_shap.showGrid(x=True, y=False, alpha=0.2)
        self.plot_shap.setLabel('bottom', 'SHAP Value (Dampak Fitur)', color='#555555')
        shap_layout.addWidget(self.plot_shap)

        shap_cell = QVBoxLayout()
        shap_cell.setSpacing(4)
        shap_cell.addWidget(lbl_shap_title)
        shap_cell.addWidget(self.box_shap, stretch=1)
        content_grid.addLayout(shap_cell, 0, 0)

        # B. KOTAK SINYAL ECG (Baris 0, Kolom 1)
        lbl_ecg_title = QLabel("Sinyal ECG")
        lbl_ecg_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #214889; background: transparent;")

        self.box_ecg = QFrame()
        self.box_ecg.setMinimumHeight(160)  # Kunci tinggi minimum
        self.box_ecg.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        ecg_layout = QVBoxLayout(self.box_ecg)
        ecg_layout.setContentsMargins(4, 4, 4, 4)

        self.plot_ecg = pg.PlotWidget()
        self.plot_ecg.showGrid(x=True, y=True, alpha=0.2)
        self.plot_ecg.setLabel('bottom', 'Waktu (s)', color='#555555')
        self.plot_ecg.setLabel('left', 'Amplitudo', color='#555555')
        ecg_layout.addWidget(self.plot_ecg)

        ecg_cell = QVBoxLayout()
        ecg_cell.setSpacing(4)
        ecg_cell.addWidget(lbl_ecg_title)
        ecg_cell.addWidget(self.box_ecg, stretch=1)
        content_grid.addLayout(ecg_cell, 0, 1)

        # C. KOTAK PARAMETER MEDIS (Baris 1, Kolom 0)
        lbl_param_title = QLabel("HASIL PARAMETER")
        lbl_param_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #214889; background: transparent;")

        self.box_parameter = QFrame()
        self.box_parameter.setMinimumHeight(160)
        self.box_parameter.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        param_layout = QGridLayout(self.box_parameter)
        param_layout.setContentsMargins(6, 6, 6, 6)
        param_layout.setSpacing(5)

        self.lbl_temp_val, self.lbl_temp_sub = self._create_param_card(param_layout, "Suhu Tubuh", "-- °C", 0, 0)
        self.lbl_hr_val, _ = self._create_param_card(param_layout, "Denyut Jantung", "-- BPM", 0, 1)
        self.lbl_rr_val, _ = self._create_param_card(param_layout, "Laju Pernapasan", "-- RPM", 1, 0)
        self.lbl_spo2_val, _ = self._create_param_card(param_layout, "Saturasi Oksigen", "-- %", 1, 1)
        self.lbl_bp_val, _ = self._create_param_card(param_layout, "Tekanan Darah", "--/-- mmHg", 2, 0, colspan=2)

        param_cell = QVBoxLayout()
        param_cell.setSpacing(4)
        param_cell.addWidget(lbl_param_title)
        param_cell.addWidget(self.box_parameter, stretch=1)
        content_grid.addLayout(param_cell, 1, 0)

        # D. KOTAK SINYAL PPG IR (Baris 1, Kolom 1)
        lbl_ppg_title = QLabel("Sinyal PPG")
        lbl_ppg_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #214889; background: transparent;")

        self.box_ppg = QFrame()
        self.box_ppg.setMinimumHeight(160)  # Kunci tinggi minimum
        self.box_ppg.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        ppg_layout = QVBoxLayout(self.box_ppg)
        ppg_layout.setContentsMargins(4, 4, 4, 4)

        self.plot_ppg = pg.PlotWidget()
        self.plot_ppg.showGrid(x=True, y=True, alpha=0.2)
        self.plot_ppg.setLabel('bottom', 'Waktu (s)', color='#555555')
        self.plot_ppg.setLabel('left', 'Amplitudo', color='#555555')
        ppg_layout.addWidget(self.plot_ppg)

        ppg_cell = QVBoxLayout()
        ppg_cell.setSpacing(4)
        ppg_cell.addWidget(lbl_ppg_title)
        ppg_cell.addWidget(self.box_ppg, stretch=1)
        content_grid.addLayout(ppg_cell, 1, 1)

        # Mengatur rasio pembagian tinggi dan lebar persis 50% : 50%
        content_grid.setRowStretch(0, 1)
        content_grid.setRowStretch(1, 1)
        content_grid.setColumnStretch(0, 1)
        content_grid.setColumnStretch(1, 1)

        main_layout.addLayout(content_grid, stretch=1)

        # =========================================================================
        # 3. FOOTER
        # =========================================================================
        self.btn_home = QPushButton("KEMBALI")
        self.btn_home.setFixedHeight(42)
        self.btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_home.setStyleSheet("""
            QPushButton { 
                background-color: #214889; 
                color: white; 
                font-size: 16px;
                font-weight: bold; 
                border-radius: 8px; 
            }
            QPushButton:hover { background-color: #183563; }
            QPushButton:pressed { background-color: #0F2240; }
        """)
        self.btn_home.clicked.connect(self.handle_home_click)
        main_layout.addWidget(self.btn_home)

    def _create_param_card(self, grid_layout, title, default_val, row, col, colspan=1):
        """Membuat kartu parameter yang ringkas dan responsif."""
        card = QFrame()
        card.setStyleSheet("QFrame { background-color: #F8FAF6; border: 1px solid #D5E5D0; border-radius: 6px; }")
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(8, 4, 8, 4)  # Margin ringkas agar tidak boros ruang vertikal
        vbox.setSpacing(0)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #555555; border: none; background: transparent;")
        
        lbl_val = QLabel(default_val)
        lbl_val.setStyleSheet("font-size: 18px; font-weight: 900; color: #214889; border: none; background: transparent;")
        
        lbl_sub = QLabel("")
        lbl_sub.setStyleSheet("font-size: 10px; font-weight: 600; color: #778899; border: none; background: transparent;")
        
        vbox.addWidget(lbl_title)
        vbox.addWidget(lbl_val)
        vbox.addWidget(lbl_sub)
        
        grid_layout.addWidget(card, row, col, 1, colspan)
        return lbl_val, lbl_sub

    # =========================================================================
    # UPDATE RESULTS: MENAMPILKAN PARAMETER & RENDERING SELURUH GRAFIK
    # =========================================================================
    def update_results(self, data):
        """Memperbarui UI parameter medis, merender grafik SHAP/ECG/PPG, dan kirim API Backend."""
        self.calculation_results = data

        temp_core = data.get("temperature", 36.5)
        temp_skin = data.get("temp_skin", 34.5)
        temp_burton = data.get("temp_burton", 35.8)
        temp_amb = data.get("temp_ambient", 28.0)

        hr = data.get("hr", 0.0)
        rr = data.get("rr", 0.0)
        spo2 = data.get("spo2", 0.0)
        sys_bp = data.get("systolic", 120)
        dia_bp = data.get("diastolic", 80)
        gcs = data.get("gcs", 15)

        # 1. Update Teks Kartu Parameter Medis
        self.lbl_temp_val.setText(f"{temp_core:.1f} °C")
        self.lbl_temp_sub.setText(f"Kulit: {temp_skin:.1f}°C | Tb (Burton): {temp_burton:.1f}°C")
        self.lbl_hr_val.setText(f"{hr:.1f} BPM")
        self.lbl_rr_val.setText(f"{rr:.1f} RPM")
        if data.get("rr_measured", False):
            self.lbl_rr_val.setToolTip(
                f"Kualitas estimasi RR: {float(data.get('rr_quality', 0.0)):.2f}"
            )
        else:
            self.lbl_rr_val.setToolTip("RR tidak terukur; nilai fallback digunakan")
        self.lbl_spo2_val.setText(f"{spo2:.1f} %")
        self.lbl_bp_val.setText(f"{int(sys_bp)}/{int(dia_bp)} mmHg")

        # 2. Update Header Triase UI
        triage_status_text = data.get("triage_status", "NON-DARURAT")
        self.update_triage_header(triage_status_text)
        input_warnings = data.get("triage_input_warnings", [])
        if input_warnings:
            self.lbl_status_text.setToolTip(
                "Sebagian input memakai fallback:\n- " + "\n- ".join(input_warnings)
            )
        else:
            self.lbl_status_text.setToolTip("")

        # 3. Render Grafik SHAP Analysis
        shap_features = data.get("shap_features", [])
        shap_values = data.get("shap_values", [])
        if len(shap_features) > 0 and len(shap_values) > 0:
            self._render_real_shap(shap_features, shap_values)
        else:
            # Jangan mempertahankan grafik milik pasien sebelumnya ketika
            # explanation untuk model ONNX belum tersedia.
            self.plot_shap.clear()

        # 4. Render Grafik Sinyal ECG 5 Detik
        time_arr = np.array(data.get("time_125", []))
        ecg_arr = np.array(data.get("ecg_smooth", []))
        if len(time_arr) > 0 and len(ecg_arr) > 0:
            sample_count = min(len(time_arr), 125 * 5)
            t_slice = time_arr[-sample_count:]
            ecg_slice = ecg_arr[-sample_count:]
            t_rel = t_slice - t_slice[0]
            
            self.plot_ecg.clear()
            self.plot_ecg.plot(t_rel, ecg_slice, pen=pg.mkPen(color='#214889', width=2))
            self.plot_ecg.setXRange(0, 5, padding=0)
            self.plot_ecg.enableAutoRange(axis='y')

        # 5. Render Grafik Sinyal PPG IR 5 Detik
        ppg_arr = np.array(data.get("ir_clean", []))
        if len(time_arr) > 0 and len(ppg_arr) > 0:
            sample_count = min(len(time_arr), 125 * 5)
            t_slice = time_arr[-sample_count:]
            ppg_slice = ppg_arr[-sample_count:]
            t_rel = t_slice - t_slice[0]
            
            self.plot_ppg.clear()
            self.plot_ppg.plot(t_rel, ppg_slice, pen=pg.mkPen(color='#214889', width=2))
            self.plot_ppg.setXRange(0, 5, padding=0)
            self.plot_ppg.enableAutoRange(axis='y')

        # 6. Pengiriman Payload ke Flask API Backend
        if self.api_client and data.get("triage_valid", False):
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

            is_sent = self.api_client.send_triage_result(
                bed_id=bed_id,
                gcs_score=gcs,
                vitals=vitals_dict,
                classification=triage_cat,
                score=xgb_score
            )
        elif self.api_client:
            print(
                "[WARN API] Hasil triase tidak dikirim karena inferensi model gagal: "
                f"{data.get('triage_error', 'alasan tidak diketahui')}"
            )

    def _map_status_to_color(self, status_text):
        """Konversi dari string teks UI ke standar warna backend/frontend."""
        status_upper = str(status_text).upper()
        if "RESUSITASI" in status_upper or status_upper == "RED":
            return "red"
        elif "DARURAT" in status_upper and "NON" not in status_upper or status_upper == "YELLOW":
            return "yellow"
        else:
            return "green"

    def _render_real_shap(self, features, shap_values):
        """Menggambar Bar Chart SHAP secara Horizontal."""
        self.plot_shap.clear()
        self.plot_shap.getAxis('bottom').enableAutoSIPrefix(False)

        shap_array = np.array(shap_values)
        if len(features) == 0 or len(shap_array) == 0 or np.all(shap_array == 0):
            return

        name_mapping = {
            'temperature_c': 'Suhu',
            'spo2': 'SpO2',
            'respiratory_rate': 'Laju Nafas',
            'heart_rate': 'Heart Rate',
            'systolic_bp': 'Sistolik',
            'diastolic_bp': 'Diastolik',
            'gcs_total': 'Skor GCS',
            'shock_index': 'Shock Index',
            'mean_arterial_pressure': 'MAP',
            'pulse_pressure': 'Pulse Press',
            'modified_shock_index': 'Modified Shock Index',
            'temp_deviation': 'Deviasi Suhu',
            'oxygen_deficit': 'Defisit Oksigen',
            'gcs_deficit': 'Defisit GCS',
            'cardiopulmonary_stress': 'Stres Kardiopulmoner',
            'neuro_hemodynamic_index': 'Indeks Neurohemodinamik',
            'news_vital_score': 'NEWS Vital'
        }

        abs_vals = np.abs(shap_array)
        top_k = min(7, len(features))
        top_indices = np.argsort(abs_vals)[-top_k:] 
        
        top_features = [features[i] for i in top_indices]
        top_values = [shap_array[i] for i in top_indices]

        labels = [name_mapping.get(f, str(f)) for f in top_features]
        y_pos = np.arange(len(top_features))

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

        axis_y = self.plot_shap.getAxis('left')
        ticks = [list(zip(y_pos, labels))]
        axis_y.setTicks(ticks)
        
        self.plot_shap.setYRange(-0.8, len(top_features) - 0.2)
        self.plot_shap.enableAutoRange(axis='x')

    def update_triage_header(self, status):
        status = status.upper()
        if "TIDAK TERSEDIA" in status or "ERROR" in status:
            self.badge_color.setStyleSheet("border-radius: 8px; background-color: #7F8C8D;")
            self.lbl_status_text.setText("TIDAK TERSEDIA")
            self.lbl_status_text.setStyleSheet("font-size: 22px; font-weight: 900; background-color: #E5E8E8; border-radius: 8px; padding-left: 12px; padding-right: 12px; color: #566573;")
        elif "RESUSITASI" in status or status == "RED":
            self.badge_color.setStyleSheet("border-radius: 8px; background-color: #E74C3C;")
            self.lbl_status_text.setText("RESUSITASI")
            self.lbl_status_text.setStyleSheet("font-size: 22px; font-weight: 900; background-color: #FADBD8; border-radius: 8px; padding-left: 12px; padding-right: 12px; color: #E74C3C;")
        elif "DARURAT" in status and "NON" not in status or status == "YELLOW":
            self.badge_color.setStyleSheet("border-radius: 8px; background-color: #F39C12;")
            self.lbl_status_text.setText("DARURAT")
            self.lbl_status_text.setStyleSheet("font-size: 22px; font-weight: 900; background-color: #FDEBD0; border-radius: 8px; padding-left: 12px; padding-right: 12px; color: #F39C12;")
        else:
            self.badge_color.setStyleSheet("border-radius: 8px; background-color: #2ECC71;")
            self.lbl_status_text.setText("NON-DARURAT")
            self.lbl_status_text.setStyleSheet("font-size: 22px; font-weight: 900; background-color: #D5F5E3; border-radius: 8px; padding-left: 12px; padding-right: 12px; color: #2ECC71;")

    def handle_home_click(self):
        
        print("[LOG] Inputs cleared. Returning to home_page...")
        self.home_requested.emit()


# =========================================================================
# UJI MANDIRI LOCAL
# =========================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    test_window = OutputPage()
    test_window.setWindowTitle("TriaGO - Test Output Pengecekan")
    test_window.showMaximized()
    
    fs = 125
    t_dummy = np.linspace(0, 10, 10 * fs)
    ecg_dummy = np.sin(2 * np.pi * 1.5 * t_dummy) + 0.2 * np.random.normal(size=len(t_dummy))
    ir_dummy = 1.2 + 0.4 * np.sin(2 * np.pi * 1.5 * t_dummy)

    dummy_results = {
        "bed": "A1",
        "patient_name": "Budi Santoso",
        "gcs": 15,
        "timestamp": "2026-07-25 10:55:00",
        "temperature": 36.5,
        "temp_skin": 34.2,
        "temp_ambient": 27.5,
        "hr": 110.5,
        "rr": 16.0,
        "spo2": 98.2,
        "systolic": 120,
        "diastolic": 80,
        "time_125": t_dummy,
        "ecg_smooth": ecg_dummy,
        "ir_clean": ir_dummy,
        "triage_status": "DARURAT",
        "xgboost_score": 0.88,
        "shap_features": ["gcs_total", "systolic_bp", "spo2", "heart_rate", "temperature_c"],
        "shap_values": [0.35, -0.22, 0.18, -0.12, 0.05]
    }

    test_window.update_results(dummy_results)
    
    sys.exit(app.exec())
