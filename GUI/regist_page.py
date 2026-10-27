import os
import sys
import csv
from datetime import datetime
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QApplication,
                             QSizePolicy)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, pyqtSignal

class RegistrationPage(QWidget):
    # Signal untuk mengirim data (GCS dan pain score) saat tombol start diklik.
    # Bed tidak dipilih di sini: backend mengalokasikannya sesuai hasil triase.
    measurement_started = pyqtSignal(dict)

    # Ukuran untuk layar Waveshare 1280x800. Ruang bekas grid bed dipakai untuk
    # tombol skor yang lebih besar agar mudah disentuh.
    SCORE_BUTTON_HEIGHT = 84
    SCORE_BUTTON_STYLE = """
        QPushButton { background-color: #FFFFFF; border: 2px solid #C2D5BB; border-radius: 12px; font-size: 26px; font-weight: bold; color: #A0B09C; }
        QPushButton:hover { border-color: #214889; color: #214889; }
    """
    SCORE_SELECTED_STYLE = """
        QPushButton {{ background-color: {color}; border: none; border-radius: 12px; font-size: 28px; font-weight: bold; color: #FFFFFF; }}
    """
    START_STYLE = """
        QPushButton {{ background-color: {color}; color: #FFFFFF; font-size: 24px; font-weight: bold; border-radius: 12px; border: none; letter-spacing: 1.5px; }}
        QPushButton:hover {{ background-color: {hover}; }}
    """

    def __init__(self):
        super().__init__()
        self.selected_gcs = None
        self.selected_pain = None
        self.gcs_buttons = {}
        self.pain_buttons = {}
        self.setup_ui()

    def setup_ui(self):
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("""
            QWidget { 
                background-color: #F6FFEC; 
                color: #214889; 
                font-family: 'Segoe UI', Arial, sans-serif; 
            }
            QLabel { 
                font-weight: bold; 
                color: #214889; 
            }
        """)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(32, 12, 32, 12)
        main_layout.setSpacing(14)
        
        # =====================================================================
        # HEADER (Judul & Logo)
        # =====================================================================
        header_layout = QHBoxLayout()
        
        title_vbox = QVBoxLayout()
        lbl_title = QLabel("GCS & PAIN SCORE")
        # Ukuran font judul diperbesar menjadi 40px
        lbl_title.setStyleSheet("font-size: 34px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;")
        
        lbl_subtitle = QLabel("Glasgow Coma Score dan skala tingkat nyeri pasien di area triase")
        # Ukuran font subtitle diperbesar menjadi 20px
        lbl_subtitle.setStyleSheet("font-size: 18px; font-weight: normal; color: #556B85;")
        
        title_vbox.addWidget(lbl_title)
        title_vbox.addWidget(lbl_subtitle)
        
        lbl_logo = QLabel()
        lbl_logo.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.abspath(os.path.join(current_dir, "..", "asset", "logo.png"))
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            lbl_logo.setPixmap(pixmap.scaledToWidth(230, Qt.TransformationMode.SmoothTransformation))
        else:
            lbl_logo.setText("TriaGO")
            lbl_logo.setStyleSheet("font-size: 42px; font-weight: 900; color: #214889;")
            
        header_layout.addLayout(title_vbox)
        header_layout.addStretch()
        header_layout.addWidget(lbl_logo)
        
        main_layout.addLayout(header_layout)
        
        # =====================================================================
        # KONTEN TENGAH (GCS, Pain Score, Tombol Start)
        # =====================================================================
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        center_layout.setSpacing(8)
        
        # --- Input GCS ---
        gcs_vbox = QVBoxLayout()
        gcs_vbox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        lbl_gcs_title = QLabel("Input Glasgow Coma Score")
        lbl_gcs_title.setStyleSheet("font-size: 26px; font-weight: bold; color: #214889;")
        gcs_vbox.addWidget(lbl_gcs_title, alignment=Qt.AlignmentFlag.AlignLeft)
        
        hbox_gcs = QHBoxLayout()
        hbox_gcs.setSpacing(10)
        hbox_gcs.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        for score in range(3, 16):
            btn_gcs = QPushButton(str(score))
            btn_gcs.setCheckable(True)
            # Dimensi tombol GCS
            btn_gcs.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_gcs.setMinimumSize(64, self.SCORE_BUTTON_HEIGHT)
            btn_gcs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn_gcs.setStyleSheet(self.SCORE_BUTTON_STYLE)
            btn_gcs.clicked.connect(lambda checked, s=score: self.handle_gcs_selection(s))
            hbox_gcs.addWidget(btn_gcs)
            self.gcs_buttons[score] = btn_gcs
            
        gcs_vbox.addLayout(hbox_gcs)
        center_layout.addLayout(gcs_vbox)
        
        center_layout.addSpacing(24) # Ruang antara GCS dan Pain Score

        # --- Input Pain Score (Skala 0 - 10; 0 tetap pilihan eksplisit) ---
        pain_vbox = QVBoxLayout()
        pain_vbox.setAlignment(Qt.AlignmentFlag.AlignLeft)

        lbl_pain_title = QLabel("Input Pain Score")
        lbl_pain_title.setStyleSheet("font-size: 26px; font-weight: bold; color: #214889;")
        pain_vbox.addWidget(lbl_pain_title, alignment=Qt.AlignmentFlag.AlignLeft)

        hbox_pain = QHBoxLayout()
        hbox_pain.setSpacing(10)
        hbox_pain.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for p_score in range(0, 11):
            btn_pain = QPushButton(str(p_score))
            btn_pain.setCheckable(True)
            btn_pain.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_pain.setMinimumSize(64, self.SCORE_BUTTON_HEIGHT)
            btn_pain.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn_pain.setStyleSheet(self.SCORE_BUTTON_STYLE)
            btn_pain.clicked.connect(lambda checked, s=p_score: self.handle_pain_selection(s))
            hbox_pain.addWidget(btn_pain)
            self.pain_buttons[p_score] = btn_pain

        pain_vbox.addLayout(hbox_pain)
        center_layout.addLayout(pain_vbox)

        center_layout.addSpacing(32)

        # --- Tombol Start Measurement ---
        self.btn_start = QPushButton("Mulai")
        self.btn_start.setFixedSize(440, 72)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.clicked.connect(self.save_and_emit_data)
        center_layout.addWidget(self.btn_start, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Tanpa grid bed, input GCS & nyeri diletakkan di tengah layar.
        main_layout.addStretch()
        main_layout.addLayout(center_layout)
        main_layout.addStretch()
        self.validate_form()

    def handle_gcs_selection(self, score):
        self.selected_gcs = score
        for s, btn in self.gcs_buttons.items():
            btn.setStyleSheet(self.SCORE_BUTTON_STYLE)
            
        if 13 <= score <= 15:
            gcs_color = "#34D980"  # Normal: Hijau
        elif 9 <= score <= 12:
            gcs_color = "#F09C00"  # Sedang: Jingga
        else:
            gcs_color = "#F12A2A"  # Parah: Merah
            
        self.gcs_buttons[score].setStyleSheet(self.SCORE_SELECTED_STYLE.format(color=gcs_color))
        self.validate_form()

    def handle_pain_selection(self, score):
        self.selected_pain = score
        for s, btn in self.pain_buttons.items():
            btn.setChecked(s == score)
            btn.setStyleSheet(self.SCORE_BUTTON_STYLE)

        if 0 <= score <= 3:
            pain_color = "#34D980"  # Ringan: Hijau
        elif 4 <= score <= 6:
            pain_color = "#F09C00"  # Sedang: Jingga
        else:
            pain_color = "#F12A2A"  # Berat: Merah

        self.pain_buttons[score].setStyleSheet(self.SCORE_SELECTED_STYLE.format(color=pain_color))
        self.validate_form()

    def validate_form(self):
        ready = bool(self.selected_gcs) and self.selected_pain is not None
        self.btn_start.setEnabled(ready)
        if ready:
            self.btn_start.setStyleSheet(self.START_STYLE.format(color="#214889", hover="#163264"))
        else:
            self.btn_start.setStyleSheet(self.START_STYLE.format(color="#A0B09C", hover="#A0B09C"))

    def reset_inputs(self):
        """Kosongkan pilihan GCS & nyeri untuk pasien berikutnya."""
        self.selected_gcs = None
        self.selected_pain = None
        for btn in list(self.gcs_buttons.values()) + list(self.pain_buttons.values()):
            btn.setChecked(False)
            btn.setStyleSheet(self.SCORE_BUTTON_STYLE)
        self.validate_form()

    def save_and_emit_data(self):
        measurement_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "gcs": self.selected_gcs,
            "pain_score": self.selected_pain if self.selected_pain is not None else 0
        }

        # 1. Tentukan folder penyimpanan 'data_pengukuran'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.abspath(os.path.join(current_dir, "..", "data_pengukuran"))
        os.makedirs(folder_path, exist_ok=True)  # Buat folder otomatis jika belum ada

        # 2. Lokasi lengkap file CSV di dalam folder data_pengukuran
        # File baru tanpa kolom bed; data_pendaftaran_pasien_pain.csv lama tetap utuh.
        csv_file = os.path.join(folder_path, "data_registrasi_triase.csv")
        file_exists = os.path.isfile(csv_file)
        
        try:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=measurement_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(measurement_data)
            print(f"[LOG] Data registrasi triase berhasil disimpan ke {csv_file}")
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan ke CSV: {e}")
            
        self.measurement_started.emit(measurement_data)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    test_window = RegistrationPage()
    test_window.resize(1280, 800)  # Layar Waveshare
    test_window.setWindowTitle("Test Run - Halaman Setup TriaGO")
    test_window.show()
    sys.exit(app.exec())
