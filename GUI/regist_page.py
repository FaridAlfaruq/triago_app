import os
import sys
import csv
from datetime import datetime
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QGridLayout, QFrame, QApplication,
                             QSizePolicy)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, pyqtSignal

class RegistrationPage(QWidget):
    # Signal untuk mengirim data (hanya bed dan gcs) saat tombol start diklik
    measurement_started = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.selected_bed = None
        self.selected_gcs = None
        self.bed_buttons = {}
        self.gcs_buttons = {}
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
        main_layout.setContentsMargins(32, 16, 32, 16)
        main_layout.setSpacing(14)
        
        # =====================================================================
        # HEADER (Judul & Logo)
        # =====================================================================
        header_layout = QHBoxLayout()
        
        title_vbox = QVBoxLayout()
        lbl_title = QLabel("LOKASI KASUR & GCS")
        # Ukuran font judul diperbesar menjadi 40px
        lbl_title.setStyleSheet("font-size: 34px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;")
        
        lbl_subtitle = QLabel("Manajemen tata letak kasur dan input Glasgow Coma Score")
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
        # KONTEN TENGAH (Kasur, GCS, Tombol Start)
        # =====================================================================
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # --- Tata Letak Kasur ---
        bed_frame = QFrame()
        bed_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        bed_frame.setStyleSheet("QFrame { border: 2px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        
        bed_layout = QVBoxLayout(bed_frame)
        bed_layout.setContentsMargins(24, 16, 24, 20)
        bed_layout.setSpacing(10)
        
        lbl_bed_title = QLabel("Tata Letak Kasur")
        lbl_bed_title.setStyleSheet("QLabel { font-size: 18px; font-weight: 600; color: #556B85; border: none; background: transparent; }")
        bed_layout.addWidget(lbl_bed_title)
        
        grid_bed = QGridLayout()
        grid_bed.setContentsMargins(0, 0, 0, 0)
        grid_bed.setVerticalSpacing(12)
        grid_bed.setHorizontalSpacing(14)
        for column in range(6):
            grid_bed.setColumnStretch(column, 1)
        
        for i in range(1, 13):
            row = 0 if i <= 6 else 1
            col = (i - 1) % 6
            bed_str = f"{i:02d}"
            
            btn_bed = QPushButton(bed_str)
            btn_bed.setCheckable(True)
            btn_bed.setMinimumHeight(66)
            btn_bed.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn_bed.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_bed.setStyleSheet("""
                QPushButton { background-color: #FFFFFF; border: 2px solid #214889; border-radius: 10px; font-size: 36px; font-weight: bold; color: #214889; }
                QPushButton:hover { background-color: #F0F4FF; }
            """)
            btn_bed.clicked.connect(lambda checked, b=bed_str: self.handle_bed_selection(b))
            grid_bed.addWidget(btn_bed, row, col)
            self.bed_buttons[bed_str] = btn_bed
            
        bed_layout.addLayout(grid_bed)
        center_layout.addWidget(bed_frame)
        
        center_layout.addSpacing(5) # Ruang antara Kasur dan GCS
        
        # --- Input GCS ---
        gcs_vbox = QVBoxLayout()
        gcs_vbox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        lbl_gcs_title = QLabel("Input Glasgow Coma Score")
        lbl_gcs_title.setStyleSheet("font-size: 23px; font-weight: bold; color: #214889;")
        gcs_vbox.addWidget(lbl_gcs_title, alignment=Qt.AlignmentFlag.AlignLeft)
        
        hbox_gcs = QHBoxLayout()
        hbox_gcs.setSpacing(10)
        hbox_gcs.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        for score in range(3, 16):
            btn_gcs = QPushButton(str(score))
            btn_gcs.setCheckable(True)
            # Dimensi tombol GCS
            btn_gcs.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_gcs.setMinimumSize(54, 54)
            btn_gcs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn_gcs.setStyleSheet("""
                QPushButton { background-color: #FFFFFF; border: 2px solid #C2D5BB; border-radius: 10px; font-size: 19px; font-weight: bold; color: #A0B09C; }
                QPushButton:hover { border-color: #214889; color: #214889; }
            """)
            btn_gcs.clicked.connect(lambda checked, s=score: self.handle_gcs_selection(s))
            hbox_gcs.addWidget(btn_gcs)
            self.gcs_buttons[score] = btn_gcs
            
        gcs_vbox.addLayout(hbox_gcs)
        center_layout.addLayout(gcs_vbox)
        
        center_layout.addSpacing(10)
        
        # --- Tombol Start Measurement ---
        self.btn_start = QPushButton("Mulai")
        self.btn_start.setFixedSize(380, 56)
        self.btn_start.setEnabled(False)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.setStyleSheet("""
            QPushButton { background-color: #A0B09C; color: #FFFFFF; font-size: 21px; font-weight: bold; border-radius: 10px; border: none; letter-spacing: 1.5px;}
        """)
        self.btn_start.clicked.connect(self.save_and_emit_data)
        center_layout.addWidget(self.btn_start, alignment=Qt.AlignmentFlag.AlignCenter)
        
        main_layout.addLayout(center_layout)
        main_layout.addStretch()

    def handle_bed_selection(self, bed_id):
        self.selected_bed = bed_id
        for bid, btn in self.bed_buttons.items():
            # Update style mereset ke ukuran dan font baru
            btn.setStyleSheet("""
                QPushButton { background-color: #FFFFFF; border: 2px solid #214889; border-radius: 10px; font-size: 36px; font-weight: bold; color: #214889; }
                QPushButton:hover { background-color: #CEF9B6; }
            """)
            
        self.bed_buttons[bed_id].setStyleSheet("""
            QPushButton { background-color: #CEF9B6; border: 3px solid #214889; border-radius: 10px; font-size: 36px; font-weight: bold; color: #214889; }
        """)
        self.validate_form()

    def handle_gcs_selection(self, score):
        self.selected_gcs = score
        for s, btn in self.gcs_buttons.items():
            btn.setStyleSheet("""
                QPushButton { background-color: #FFFFFF; border: 2px solid #C2D5BB; border-radius: 10px; font-size: 19px; font-weight: bold; color: #A0B09C; }
                QPushButton:hover { border-color: #214889; color: #214889; }
            """)
            
        if 13 <= score <= 15:
            gcs_color = "#34D980"  # Normal: Hijau
        elif 9 <= score <= 12:
            gcs_color = "#F09C00"  # Sedang: Jingga
        else:
            gcs_color = "#F12A2A"  # Parah: Merah
            
        self.gcs_buttons[score].setStyleSheet(f"""
            QPushButton {{
                background-color: {gcs_color};
                border: none;
                border-radius: 10px;
                font-size: 20px;
                font-weight: bold;
                color: #FFFFFF;
            }}
        """)
        self.validate_form()

    def validate_form(self):
        if self.selected_bed and self.selected_gcs:
            self.btn_start.setEnabled(True)
            self.btn_start.setStyleSheet("""
                QPushButton { 
                    background-color: #214889; 
                    color: #FFFFFF; 
                    font-size: 21px;
                    font-weight: bold; 
                    border-radius: 10px;
                    border: none;
                    letter-spacing: 1.5px;
                }
                QPushButton:hover { background-color: #163264; }
            """)
        else:
            self.btn_start.setEnabled(False)
            self.btn_start.setStyleSheet("""
                QPushButton { background-color: #A0B09C; color: #FFFFFF; font-size: 21px; font-weight: bold; border-radius: 10px; border: none; letter-spacing: 1.5px;}
            """)

    def save_and_emit_data(self):
        measurement_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bed": self.selected_bed,
            "gcs": self.selected_gcs
        }

        csv_file = "data_pendaftaran_pasien.csv"
        file_exists = os.path.isfile(csv_file)
        
        try:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=measurement_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(measurement_data)
            print(f"[LOG] Data Kasur {measurement_data['bed']} berhasil disimpan ke {csv_file}")
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan ke CSV: {e}")
            
        self.measurement_started.emit(measurement_data)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    test_window = RegistrationPage()
    test_window.resize(1920, 1080)  
    test_window.setWindowTitle("Test Run - Halaman Setup TriaGO")
    test_window.show()
    sys.exit(app.exec())
