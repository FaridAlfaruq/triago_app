import sys
import os
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QGridLayout, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap

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
        # 1. SETTING BACKGROUND HALAMAN (Hijau Muda Khas TriaGO)
        self.setStyleSheet("background-color: #F6FFEC;")
        
        # Layout utama vertikal halaman (Master Manager)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 20, 40, 20)
        main_layout.setSpacing(15)

        # =========================================================================
        # 2. BAGIAN HEADER: JUDUL, KOTAK STATUS TRIASE, & LOGO
        # =========================================================================
        header_layout = QHBoxLayout()
        header_layout.setSpacing(20)
        header_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Kiri: Judul Halaman & Subtitle
        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(4)
        lbl_title = QLabel("HASIL PENGECEKAN")
        lbl_title.setStyleSheet("font-size: 32px; font-weight: 900; color: #214889; background: transparent;")
        lbl_subtitle = QLabel("Output parameter dan hasil klasifikasi")
        lbl_subtitle.setStyleSheet("font-size: 22px; font-weight: 500; color: #555555; background: transparent;")
        title_vbox.addWidget(lbl_title)
        title_vbox.addWidget(lbl_subtitle)
        header_layout.addLayout(title_vbox)

        # Tengah: Indikator Triase (Square Badge + Text Capsule)
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
        # 3. BAGIAN BODY: LINIER BOX LAYOUT (SOLUSI TOTAL OVERFLOW)
        # =========================================================================
        # Deklarasi teks judul luar kotak
        lbl_shap_title = QLabel("SHAP Analysis")
        lbl_shap_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #214889; background: transparent;")
        
        lbl_ecg_title = QLabel("Sinyal ECG")
        lbl_ecg_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #214889; background: transparent;")
        
        lbl_param_title = QLabel("HASIL PARAMETER")
        lbl_param_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #214889; background: transparent;")
        
        lbl_ppg_title = QLabel("Sinyal PPG")
        lbl_ppg_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #214889; background: transparent;")

        # Deklarasi 4 Kotak Putih Utama (QFrame)
        self.box_shap = QFrame()
        self.box_shap.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        shap_layout = QVBoxLayout(self.box_shap)
        shap_layout.setContentsMargins(15, 15, 15, 15)
        
        self.box_ecg = QFrame()
        self.box_ecg.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        ecg_layout = QVBoxLayout(self.box_ecg)
        ecg_layout.setContentsMargins(15, 15, 15, 15)

        self.box_parameter = QFrame()
        self.box_parameter.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        param_layout = QVBoxLayout(self.box_parameter)
        param_layout.setContentsMargins(15, 15, 15, 15)

        self.box_ppg = QFrame()
        self.box_ppg.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        ppg_layout = QVBoxLayout(self.box_ppg)
        ppg_layout.setContentsMargins(15, 15, 15, 15)

        # --- BARIS 1: TOP ROW (SHAP + ECG) ---
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

        # --- BARIS 2: BOTTOM ROW (PARAMETER + PPG) ---
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
        # 4. BAGIAN FOOTER: BUTTON KEMBALI
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

    def update_triage_header(self, status):
        """Fungsi dinamis untuk mengubah warna header sesuai hasil klasifikasi"""
        status = status.upper()
        if status == "RESUSITASI":
            self.badge_color.setStyleSheet("border-radius: 8px; background-color: #E74C3C;")
            self.lbl_status_text.setText("RESUSITASI")
            self.lbl_status_text.setStyleSheet("font-size: 32px; font-weight: 900; color: #FFFFFF; background-color: #FADBD8; border-radius: 8px; padding-left: 20px; padding-right: 20px; color: #E74C3C;")
        elif status == "DARURAT":
            self.badge_color.setStyleSheet("border-radius: 8px; background-color: #F39C12;")
            self.lbl_status_text.setText("DARURAT")
            self.lbl_status_text.setStyleSheet("font-size: 32x; font-weight: 900; color: #FFFFFF; background-color: #FDEBD0; border-radius: 8px; padding-left: 20px; padding-right: 20px; color: #F39C12;")
        elif status == "NON-DARURAT":
            self.badge_color.setStyleSheet("border-radius: 8px; background-color: #2ECC71;")
            self.lbl_status_text.setText("NON-DARURAT")
            self.lbl_status_text.setStyleSheet("font-size: 32px; font-weight: 900; color: #FFFFFF; background-color: #D5F5E3; border-radius: 8px; padding-left: 20px; padding-right: 20px; color: #2ECC71;")

    def handle_home_click(self):
        # Tempat untuk clearing input (akan diisi penuh nanti)
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
    test_window.update_triage_header("DARURAT") 
    
    sys.exit(app.exec())