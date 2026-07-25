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

# Impor API Client dari folder services
try:
    from service.api_client import TriageApiClient
except ImportError:
    # Fallback jika struktur direktori berbeda saat run independen
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
        
        # Inisialisasi API Client
        self.api_client = TriageApiClient() if TriageApiClient else None
        
        self.setup_ui()

    def setup_ui(self):
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background-color: #F6FFEC;")
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 14, 28, 14)
        main_layout.setSpacing(10)

        # =========================================================================
        # 1. HEADER
        # =========================================================================
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)
        header_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(4)
        lbl_title = QLabel("HASIL PENGECEKAN")
        lbl_title.setStyleSheet("font-size: 28px; font-weight: 900; color: #214889; background: transparent;")
        lbl_subtitle = QLabel("Output parameter dan hasil klasifikasi ML")
        lbl_subtitle.setStyleSheet("font-size: 17px; font-weight: 500; color: #555555; background: transparent;")
        title_vbox.addWidget(lbl_title)
        title_vbox.addWidget(lbl_subtitle)
        header_layout.addLayout(title_vbox)

        self.triage_container = QHBoxLayout()
        self.triage_container.setSpacing(10)
        
        self.badge_color = QFrame()
        self.badge_color.setFixedSize(54, 54)
        self.badge_color.setStyleSheet("border-radius: 8px; background-color: #FF5252;")
        
        self.lbl_status_text = QLabel("RESUSITASI")
        self.lbl_status_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status_text.setFixedHeight(52)
        self.lbl_status_text.setMinimumWidth(190)
        self.lbl_status_text.setStyleSheet("""
            font-size: 24px; font-weight: 900; color: #FFFFFF;
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
            lbl_logo.setPixmap(pixmap.scaledToWidth(180, Qt.TransformationMode.SmoothTransformation))
        else:
            lbl_logo.setText("TriaGO")
            lbl_logo.setStyleSheet("font-size: 32px; font-weight: 900; color: #214889;")
        header_layout.addWidget(lbl_logo)

        main_layout.addLayout(header_layout)

        # =========================================================================
        # 2. BODY LAYOUT
        # =========================================================================
        lbl_shap_title = QLabel("SHAP Analysis")
        lbl_shap_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #214889; background: transparent;")
        
        lbl_ecg_title = QLabel("Sinyal ECG (5 Detik)")
        lbl_ecg_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #214889; background: transparent;")
        
        lbl_param_title = QLabel("HASIL PARAMETER")
        lbl_param_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #214889; background: transparent;")
        
        lbl_ppg_title = QLabel("Sinyal PPG IR (5 Detik)")
        lbl_ppg_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #214889; background: transparent;")

        # A. KOTAK SHAP
        self.box_shap = QFrame()
        self.box_shap.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        shap_layout = QVBoxLayout(self.box_shap)
        shap_layout.setContentsMargins(6, 6, 6, 6)
        
        self.plot_shap = pg.PlotWidget()
        self.plot_shap.showGrid(x=True, y=False, alpha=0.2)
        self.plot_shap.setLabel('bottom', 'SHAP Value (Dampak Fitur)', color='#555555')
        shap_layout.addWidget(self.plot_shap)

        # B. KOTAK ECG
        self.box_ecg = QFrame()
        self.box_ecg.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        ecg_layout = QVBoxLayout(self.box_ecg)
        ecg_layout.setContentsMargins(6, 6, 6, 6)

        self.plot_ecg = pg.PlotWidget()
        self.plot_ecg.showGrid(x=True, y=True, alpha=0.2)
        self.plot_ecg.setLabel('bottom', 'Waktu (detik)', color='#555555')
        self.plot_ecg.setLabel('left', 'Amplitudo (mV)', color='#555555')
        ecg_layout.addWidget(self.plot_ecg)

        # C. KOTAK PARAMETER MEDIS
        self.box_parameter = QFrame()
        self.box_parameter.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        param_layout = QGridLayout(self.box_parameter)
        param_layout.setContentsMargins(10, 8, 10, 8)
        param_layout.setSpacing(7)

        self.lbl_temp_val, self.lbl_temp_sub = self._create_param_card(param_layout, "Suhu Tubuh", "-- °C", 0, 0)
        self.lbl_hr_val, _ = self._create_param_card(param_layout, "Denyut Jantung", "-- BPM", 0, 1)
        self.lbl_rr_val, _ = self._create_param_card(param_layout, "Laju Pernapasan", "-- RPM", 1, 0)
        self.lbl_spo2_val, _ = self._create_param_card(param_layout, "Saturasi Oksigen", "-- %", 1, 1)
        self.lbl_bp_val, _ = self._create_param_card(param_layout, "Tekanan Darah", "--/-- mmHg", 2, 0, colspan=2)

        # D. KOTAK PPG IR
        self.box_ppg = QFrame()
        self.box_ppg.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        ppg_layout = QVBoxLayout(self.box_ppg)
        ppg_layout.setContentsMargins(6, 6, 6, 6)

        self.plot_ppg = pg.PlotWidget()
        self.plot_ppg.showGrid(x=True, y=True, alpha=0.2)
        self.plot_ppg.setLabel('bottom', 'Waktu (detik)', color='#555555')
        self.plot_ppg.setLabel('left', 'Amplitudo (a.u.)', color='#555555')
        ppg_layout.addWidget(self.plot_ppg)

        top_row_layout = QHBoxLayout()
        top_row_layout.setSpacing(14)
        
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

        bottom_row_layout = QHBoxLayout()
        bottom_row_layout.setSpacing(14)
        
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
        # 3. FOOTER
        # =========================================================================
        self.btn_home = QPushButton("KEMBALI")
        self.btn_home.setFixedHeight(48)
        self.btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_home.setStyleSheet("""
            QPushButton { 
                background-color: #214889; 
                color: white; 
                font-size: 17px;
                font-weight: bold; 
                border-radius: 8px; 
            }
            QPushButton:hover { background-color: #183563; }
            QPushButton:pressed { background-color: #0F2240; }
        """)
        self.btn_home.clicked.connect(self.handle_home_click)
        main_layout.addWidget(self.btn_home)

    def _create_param_card(self, grid_layout, title, default_val, row, col, colspan=1):
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
    # UPDATE RESULTS & INTEGRASI SINKRONISASI PAYLOAD
    # =========================================================================
    def update_results(self, data):
        """Memperbarui UI dan mengirim data terbaru ke Flask API Backend."""
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
        gcs = data.get("gcs", 15)  # Ambil skor GCS

        # (Logika update tampilan PyQt6 tetap sama...)
        self.lbl_temp_val.setText(f"{temp_core:.1f} °C")
        self.lbl_temp_sub.setText(f"Kulit: {temp_skin:.1f}°C | Tb (Burton): {temp_burton:.1f}°C")
        self.lbl_hr_val.setText(f"{hr:.1f} BPM")
        self.lbl_rr_val.setText(f"{rr:.1f} RPM")
        self.lbl_spo2_val.setText(f"{spo2:.1f} %")
        self.lbl_bp_val.setText(f"{int(sys_bp)}/{int(dia_bp)} mmHg")

        # Update Header Triage UI
        triage_status_text = data.get("triage_status", "NON-DARURAT")
        self.update_triage_header(triage_status_text)

        # ---------------------------------------------------------------------
        # PENGIRIMAN PAYLOAD: SUHU INTI & SKOR GCS
        # ---------------------------------------------------------------------
        if self.api_client:
            bed_id = data.get("bed", "A1")
            
            vitals_dict = {
                "hr": hr,
                "spo2": spo2,
                "rr": rr,
                "temp_core": temp_core,  # Gunakan Suhu Inti hasil estimasi
                "sys": sys_bp,
                "dia": dia_bp
            }
            
            triage_cat = self._map_status_to_color(triage_status_text)
            xgb_score = data.get("xgboost_score", 0.85)

            # Kirim data ke Flask API Server
            self.api_client.send_triage_result(
                bed_id=bed_id,
                gcs_score=gcs,          # Kirim Skor GCS
                vitals=vitals_dict,
                classification=triage_cat,
                score=xgb_score
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
            'gcs_total': 'Skor GCS'
        }

        abs_vals = np.abs(shap_array)
        top_indices = np.argsort(abs_vals)[-7:] 
        
        top_features = [features[i] for i in top_indices]
        top_values = [shap_array[i] for i in top_indices]

        labels = [name_mapping.get(f, f) for f in top_features]
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
        if "RESUSITASI" in status or status == "RED":
            self.badge_color.setStyleSheet("border-radius: 8px; background-color: #E74C3C;")
            self.lbl_status_text.setText("RESUSITASI")
            self.lbl_status_text.setStyleSheet("font-size: 24px; font-weight: 900; background-color: #FADBD8; border-radius: 8px; padding-left: 12px; padding-right: 12px; color: #E74C3C;")
        elif "DARURAT" in status and "NON" not in status or status == "YELLOW":
            self.badge_color.setStyleSheet("border-radius: 8px; background-color: #F39C12;")
            self.lbl_status_text.setText("DARURAT")
            self.lbl_status_text.setStyleSheet("font-size: 24px; font-weight: 900; background-color: #FDEBD0; border-radius: 8px; padding-left: 12px; padding-right: 12px; color: #F39C12;")
        else:
            self.badge_color.setStyleSheet("border-radius: 8px; background-color: #2ECC71;")
            self.lbl_status_text.setText("NON-DARURAT")
            self.lbl_status_text.setStyleSheet("font-size: 24px; font-weight: 900; background-color: #D5F5E3; border-radius: 8px; padding-left: 12px; padding-right: 12px; color: #2ECC71;")

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
        "triage_status": "RESUSITASI",
        "xgboost_score": 0.88,
        "shap_features": ["gcs_total", "systolic_bp", "spo2", "heart_rate", "temperature_c"],
        "shap_values": [0.35, -0.22, 0.18, -0.12, 0.05]
    }

    test_window.update_results(dummy_results)
    
    sys.exit(app.exec())
