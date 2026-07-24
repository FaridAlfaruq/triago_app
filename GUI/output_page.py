import sys
import os
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QGridLayout, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap

# Konfigurasi Global Tema PyQtGraph (Background Putih & Teks Gelap)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', '#214889')


class OutputPage(QWidget):
    # Sinyal untuk kembali ke halaman utama
    home_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.patient_data = {}
        self.calculation_results = {}
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
        lbl_subtitle = QLabel("Output parameter dan hasil klasifikasi")
        lbl_subtitle.setStyleSheet("font-size: 22px; font-weight: 500; color: #214889; background: transparent;")
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

        # Membuat 5 Kartu Parameter
        self.lbl_temp_val = self._create_param_card(param_layout, "Suhu", "-- °C", 0, 0)
        self.lbl_hr_val = self._create_param_card(param_layout, "Denyut Jantung", "-- BPM", 0, 1)
        self.lbl_rr_val = self._create_param_card(param_layout, "Laju Pernapasan", "-- RPM", 1, 0)
        self.lbl_spo2_val = self._create_param_card(param_layout, "Saturasi Oksigen", "-- %", 1, 1)
        self.lbl_bp_val = self._create_param_card(param_layout, "Tekanan Darah", "--/-- mmHg", 2, 0, colspan=2)

        # --- D. KOTAK PPG IR (TANPA LEGEND) ---
        self.box_ppg = QFrame()
        self.box_ppg.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        ppg_layout = QVBoxLayout(self.box_ppg)
        ppg_layout.setContentsMargins(10, 10, 10, 10)

        self.plot_ppg = pg.PlotWidget()
        self.plot_ppg.showGrid(x=True, y=True, alpha=0.2)
        self.plot_ppg.setLabel('bottom', 'Waktu (detik)', color='#555555')
        self.plot_ppg.setLabel('left', 'Amplitudo (a.u.)', color='#555555')
        # Legend sengaja dihilangkan
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
        vbox.setSpacing(2)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #555555; border: none; background: transparent;")
        
        lbl_val = QLabel(default_val)
        lbl_val.setStyleSheet("font-size: 20px; font-weight: 900; color: #214889; border: none; background: transparent;")
        
        vbox.addWidget(lbl_title)
        vbox.addWidget(lbl_val)
        grid_layout.addWidget(card, row, col, 1, colspan)
        return lbl_val

    # =========================================================================
    # FUNGSI UTAMA: MENERIMA DATA & PLOTTING
    # =========================================================================
    def update_results(self, data):
        """Dipanggil dari MainGUI untuk mengisi 5 parameter medis dan plot sinyal (5 Detik)."""
        self.calculation_results = data

        # 1. Update 5 Parameter Medis
        temp = data.get("temperature", data.get("suhu", 36.5))
        hr = data.get("hr", data.get("heart_rate", 0.0))
        rr = data.get("rr", data.get("respiration_rate", 0.0))
        spo2 = data.get("spo2", 0.0)
        sys_bp = data.get("systolic", data.get("sys", 120))
        dia_bp = data.get("diastolic", data.get("dia", 80))

        self.lbl_temp_val.setText(f"{temp} °C")
        self.lbl_hr_val.setText(f"{hr} BPM")
        self.lbl_rr_val.setText(f"{rr} RPM")
        self.lbl_spo2_val.setText(f"{spo2} %")
        self.lbl_bp_val.setText(f"{sys_bp}/{dia_bp} mmHg")

        # 2. Potong Sinyal Menjadi 5 Detik Pertama (fs = 125 Hz -> 625 samples)
        fs = 125
        max_samples = 5 * fs

        time_x = data.get("time_125", np.array([]))
        ecg_y = data.get("ecg_smooth", data.get("ecg_125", np.array([])))
        ir_y = data.get("ir_clean", data.get("ir_smooth", np.array([])))

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

        # 5. Render Dummy Grafik SHAP Analysis
        self._render_dummy_shap()

    def _render_dummy_shap(self):
        """Membuat grafik horizontal bar dummy untuk SHAP Analysis."""
        self.plot_shap.clear()
        
        features = ["GCS", "Tekanan Darah", "SpO2", "HR", "Suhu"]
        shap_values = [0.42, -0.35, 0.28, -0.18, 0.08]
        y_pos = np.arange(len(features))

        for y, val in zip(y_pos, shap_values):
            color = '#E74C3C' if val < 0 else '#2ECC71'
            bar = pg.BarGraphItem(x0=0, y=y, height=0.45, width=val, brush=color, pen=color)
            self.plot_shap.addItem(bar)

        axis_y = self.plot_shap.getAxis('left')
        ticks = [list(zip(y_pos, features))]
        axis_y.setTicks(ticks)

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
    
    # Generate Dummy Data 10 Detik
    fs = 125
    t_dummy = np.linspace(0, 10, 10 * fs)
    ecg_dummy = np.sin(2 * np.pi * 1.5 * t_dummy) + 0.2 * np.random.normal(size=len(t_dummy))
    ir_dummy = 1.2 + 0.4 * np.sin(2 * np.pi * 1.5 * t_dummy)

    dummy_results = {
        "temperature": 36.5,
        "hr": 110.5,
        "rr": 16.0,
        "spo2": 98.2,
        "systolic": 120,
        "diastolic": 80,
        "time_125": t_dummy,
        "ecg_smooth": ecg_dummy,
        "ir_clean": ir_dummy,
        "triage_status": "DARURAT"
    }

    test_window.update_results(dummy_results)
    test_window.update_triage_header(dummy_results["triage_status"]) 
    
    sys.exit(app.exec())