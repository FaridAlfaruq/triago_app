import os
import sys
import csv
from datetime import datetime
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QGridLayout, QFrame, 
                             QRadioButton, QButtonGroup, QApplication)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, pyqtSignal

class RegistrationPage(QWidget):
    # Custom signal untuk mengirim data ke main_gui saat tombol start diklik
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
        # 1. Background utama hijau muda (#F6FFEC) dan teks biru (#214889)
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
        
        # Layout Utama Induk secara Vertikal
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(50, 40, 50, 40)
        main_layout.setSpacing(20)
        
        # =====================================================================
        # PREPARASI ELEMEN HEADER (TANPA QHBOXLAYOUT LAMA)
        # =====================================================================
        # Header Kiri (Judul & Subtitle)
        title_vbox = QVBoxLayout()
        lbl_title = QLabel("LOKASI KASUR PASIEN")
        lbl_title.setStyleSheet("font-size: 32px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;")
        
        lbl_subtitle = QLabel("Manajemen kasur dan status pasien")
        lbl_subtitle.setStyleSheet("font-size: 16px; font-weight: normal; color: #556B85;")
        
        title_vbox.addWidget(lbl_title)
        title_vbox.addWidget(lbl_subtitle)
        
        # Logo TriaGO (Set Aligment Rata Kanan)
        lbl_logo = QLabel()
        lbl_logo.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(current_dir, r"C:\Users\Adyty\Documents\Farid ITS\TriaGo\asset\logo.png") 
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            lbl_logo.setPixmap(pixmap.scaledToWidth(260, Qt.TransformationMode.SmoothTransformation))
        else:
            lbl_logo.setText("TriaGO")
            lbl_logo.setStyleSheet("font-size: 36px; font-weight: 900; color: #214889;")
        
        # =====================================================================
        # KONTANER KIRI: Tata Letak Kasur & Kotak GCS
        # =====================================================================
        # ---------------------------------------------------------------------
        # KONTANER KIRI: Tata Letak Kasur & Kotak GCS (VERSI MAKSIMAL & PROPORSIONAL)
        # ---------------------------------------------------------------------
        left_content = QVBoxLayout()
        left_content.setSpacing(0) 
        
        # Tata Letak Kasur Box Frame
        bed_frame = QFrame()
        bed_frame.setStyleSheet("QFrame { border: 1px solid #C2D5BB; border-radius: 8px; background-color: #FFFFFF; }")
        
        bed_layout = QVBoxLayout(bed_frame)
        bed_layout.setContentsMargins(25, 25, 25, 25)
        bed_layout.setSpacing(15)
        
        lbl_bed_title = QLabel("Tata Letak Kasur")
        lbl_bed_title.setStyleSheet("QLabel { font-size: 16px; font-weight: 600; color: #556B85; border: none; background: transparent; margin-bottom: 10px; padding: 0px; margin-left: 0px; }")
        bed_layout.addWidget(lbl_bed_title)
        
        grid_bed = QGridLayout()
        grid_bed.setContentsMargins(0, 0, 0, 0)
        grid_bed.setSpacing(15)
        
        for i in range(1, 13):
            row = 0 if i <= 6 else 1
            col = (i - 1) % 6
            bed_str = f"{i:02d}"
            
            btn_bed = QPushButton(bed_str)
            btn_bed.setCheckable(True)
            
            # 1. PERBESAR TINGGI BOX DI SINI (Dari 120 ke 175)
            btn_bed.setFixedSize(120, 175)  
            
            btn_bed.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_bed.setStyleSheet("QPushButton { background-color: #FFFFFF; border: 1.5px solid #214889; border-radius: 8px; font-size: 32px; font-weight: bold; color: #214889; } QPushButton:hover { background-color: #F0F4FF; }")
            btn_bed.clicked.connect(lambda checked, b=bed_str: self.handle_bed_selection(b))
            grid_bed.addWidget(btn_bed, row, col)
            self.bed_buttons[bed_str] = btn_bed
            
        bed_layout.addLayout(grid_bed)
        left_content.addWidget(bed_frame)
        
        # 2. GANTI STRETCH DENGAN SPACING PASTI AGAR TIDAK ADA GAP KOSONG MELAR
        left_content.addSpacing(25) 
        
        # Input Glasgow Coma Score Box
        gcs_vbox = QVBoxLayout()
        lbl_gcs_title = QLabel("Input Glasgow Coma Score")
        lbl_gcs_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #214889; margin-bottom: 10px;")
        gcs_vbox.addWidget(lbl_gcs_title)
        
        hbox_gcs = QHBoxLayout()
        hbox_gcs.setSpacing(10)
        hbox_gcs.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        for score in range(3, 16):
            btn_gcs = QPushButton(str(score))
            btn_gcs.setCheckable(True)
            btn_gcs.setFixedSize(55, 55)  
            btn_gcs.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_gcs.setStyleSheet("QPushButton { background-color: #FFFFFF; border: 1px solid #C2D5BB; border-radius: 8px; font-size: 18px; font-weight: bold; color: #A0B09C; } QPushButton:hover { border-color: #214889; color: #214889; }")
            btn_gcs.clicked.connect(lambda checked, s=score: self.handle_gcs_selection(s))
            hbox_gcs.addWidget(btn_gcs)
            self.gcs_buttons[score] = btn_gcs
            
        gcs_vbox.addLayout(hbox_gcs)
        left_content.addLayout(gcs_vbox)
        
        # ---------------------------------------------------------------------
        # KONTANER KANAN: Kotak Identitas Data Pasien
        # ---------------------------------------------------------------------
        right_content = QVBoxLayout()
        right_content.setSpacing(0)
        right_content.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        form_frame = QFrame()
        form_frame.setFixedSize(417, 600)
        form_frame.setStyleSheet("QFrame { border: 1px solid #C2D5BB; border-radius: 8px; background-color: #FFFFFF; }")
        
        form_layout = QVBoxLayout(form_frame)
        form_layout.setContentsMargins(25, 15, 25, 15)
        form_layout.setSpacing(12)
        
        lbl_data_title = QLabel("Data Pasien")
        lbl_data_title.setStyleSheet("font-size: 22px; font-weight: bold; color: #556B85; border: none; background: transparent;")
        form_layout.addWidget(lbl_data_title)
        
        lbl_nama = QLabel("Nama")
        lbl_nama.setStyleSheet("QLabel { font-size: 20px; font-weight: bold; border: none; background: transparent; margin-top: -8px; }")
        form_layout.addWidget(lbl_nama)
        
        self.input_nama = QLineEdit()
        self.input_nama.setPlaceholderText("Pandu")
        self.input_nama.setStyleSheet("QLineEdit { border: 1px solid #C2D5BB; border-radius: 6px; padding: 8px; font-size: 16px; color: #214889; background: #FFFFFF; }")
        form_layout.addWidget(self.input_nama)
        
        lbl_umur = QLabel("Umur")
        lbl_umur.setStyleSheet("font-size: 20px; font-weight: bold; border: none; background: transparent;")
        form_layout.addWidget(lbl_umur)
        
        self.input_umur = QLineEdit()
        self.input_umur.setPlaceholderText("22")
        self.input_umur.setStyleSheet("QLineEdit { border: 1px solid #C2D5BB; border-radius: 6px; padding: 8px; font-size: 16px; color: #214889; background: #FFFFFF; }")
        form_layout.addWidget(self.input_umur)
        
        radio_style = """
        QRadioButton { font-size: 16px; font-weight: normal; color: #214889; background: transparent; border: none; padding-left: 8px; }
        QRadioButton::indicator { width: 20px; height: 20px; border: 1px solid #C2D5BB; border-radius: 10px; background-color: #FFFFFF; }
        QRadioButton::indicator:checked { border: 1px solid #71CC44; background-color: #71CC44; }
        """
        
        lbl_jk = QLabel("Jenis Kelamin")
        lbl_jk.setStyleSheet("font-size: 20px; font-weight: bold; border: none; background: transparent;")
        form_layout.addWidget(lbl_jk)
        
        gender_container = QWidget()
        gender_container.setStyleSheet("border: none; background: transparent;")
        layout_gender = QHBoxLayout(gender_container)
        layout_gender.setContentsMargins(0, 0, 0, 0)
        
        self.radio_laki = QRadioButton("Laki-laki")
        self.radio_perempuan = QRadioButton("Perempuan")
        self.radio_laki.setStyleSheet(radio_style)
        self.radio_perempuan.setStyleSheet(radio_style)
        
        self.group_gender = QButtonGroup(self)
        self.group_gender.addButton(self.radio_laki)
        self.group_gender.addButton(self.radio_perempuan)
        
        layout_gender.addWidget(self.radio_laki)
        layout_gender.addWidget(self.radio_perempuan)
        form_layout.addWidget(gender_container)
        
        lbl_kk = QLabel("Kategori Kasus")
        lbl_kk.setStyleSheet("font-size: 20px; font-weight: bold; border: none; background: transparent;")
        form_layout.addWidget(lbl_kk)
        
        kasus_container = QWidget()
        kasus_container.setStyleSheet("border: none; background: transparent;")
        layout_kasus = QHBoxLayout(kasus_container)
        layout_kasus.setContentsMargins(0, 0, 0, 0)
        
        self.radio_trauma = QRadioButton("Trauma")
        self.radio_nontrauma = QRadioButton("Non-Trauma")
        self.radio_trauma.setStyleSheet(radio_style)
        self.radio_nontrauma.setStyleSheet(radio_style)
        
        self.group_kasus = QButtonGroup(self)
        self.group_kasus.addButton(self.radio_trauma)
        self.group_kasus.addButton(self.radio_nontrauma)
        
        layout_kasus.addWidget(self.radio_trauma)
        layout_kasus.addWidget(self.radio_nontrauma)
        form_layout.addWidget(kasus_container)
        
        form_layout.addSpacing(10)
        
        self.btn_start = QPushButton("MULAI")
        self.btn_start.setEnabled(False)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.setFixedHeight(50)
        self.btn_start.setStyleSheet("QPushButton { background-color: #A0B09C; color: #FFFFFF; font-size: 20px; font-weight: bold; border-radius: 8px; border: none; }")
        self.btn_start.clicked.connect(self.save_and_emit_data)
        form_layout.addWidget(self.btn_start)
        
        right_content.addWidget(form_frame)
        right_content.addStretch()
        
        # =====================================================================
        # PERAKITAN STRUKTUR MAKRO DENGAN QGRIDLAYOUT (VERSI CENTERED)
        # =====================================================================
        macro_grid = QGridLayout()
        macro_grid.setHorizontalSpacing(40)  # Mengunci jarak antar kolom tetap 40px
        macro_grid.setVerticalSpacing(25)    # Mengunci jarak vertikal tetap 25px
        
        # Hapus AlignLeft, sisakan AlignTop agar konten tetap menempel ke atas secara vertikal
        macro_grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Mengatur sistem kembaran stretch (pembagi ruang kosong kanan-kiri)
        macro_grid.setColumnStretch(0, 1) # Kolom 0: Penyangga ruang kosong KIRI
        macro_grid.setColumnStretch(1, 0) # Kolom 1: Konten Kiri (Ukuran pas sesuai isi)
        macro_grid.setColumnStretch(2, 0) # Kolom 2: Konten Kanan (Ukuran pas sesuai isi)
        macro_grid.setColumnStretch(3, 1) # Kolom 3: Penyangga ruang kosong KANAN
        
        # Plot komponen digeser ke koordinat Kolom 1 dan Kolom 2
        # Format: addLayout/addWidget(variabel, baris, kolom, [alignment])
        macro_grid.addLayout(title_vbox, 0, 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        macro_grid.addWidget(lbl_logo, 0, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        macro_grid.addLayout(left_content, 1, 1)
        macro_grid.addLayout(right_content, 1, 2)
        
        # Terapkan Grid Makro ke Layout Utama Halaman
        main_layout.addLayout(macro_grid)
        main_layout.addStretch()

        # Koneksi sinyal validasi data form secara real-time
        self.input_nama.textChanged.connect(self.validate_form)
        self.input_umur.textChanged.connect(self.validate_form)
        self.radio_laki.toggled.connect(self.validate_form)
        self.radio_perempuan.toggled.connect(self.validate_form)
        self.radio_trauma.toggled.connect(self.validate_form)
        self.radio_nontrauma.toggled.connect(self.validate_form)

    def handle_bed_selection(self, bed_id):
        self.selected_bed = bed_id
        for bid, btn in self.bed_buttons.items():
            # Terapkan tinggi 175 untuk kondisi normal/hover
            btn.setStyleSheet("""
                QPushButton { background-color: #FFFFFF; border: 1.5px solid #214889; border-radius: 8px; font-size: 32px; font-weight: bold; color: #214889; }
                QPushButton:hover { background-color: #CEF9B6; }
            """)
            btn.setFixedSize(120, 175) 
            
        # Terapkan tinggi 175 untuk kondisi kasur terpilih
        self.bed_buttons[bed_id].setStyleSheet("""
            QPushButton { background-color: #CEF9B6; border: 2.5px solid #214889; border-radius: 8px; font-size: 32px; font-weight: bold; color: #214889; }
        """)
        self.bed_buttons[bed_id].setFixedSize(120, 175)
        self.validate_form()

    def handle_gcs_selection(self, score):
        self.selected_gcs = score
        for s, btn in self.gcs_buttons.items():
            btn.setStyleSheet("""
                QPushButton { background-color: #FFFFFF; border: 1px solid #C2D5BB; border-radius: 8px; font-size: 18px; font-weight: bold; color: #A0B09C; }
                QPushButton:hover { border-color: #214889; color: #214889; }
            """)
            
        # Dinamika perubahan warna tombol GCS saat di-klik
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
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
                color: #FFFFFF;
            }}
        """)
        self.validate_form()

    def validate_form(self):
        gender_ok = self.radio_laki.isChecked() or self.radio_perempuan.isChecked()
        kasus_ok = self.radio_trauma.isChecked() or self.radio_nontrauma.isChecked()
        nama_ok = len(self.input_nama.text().strip()) > 0
        umur_ok = len(self.input_umur.text().strip()) > 0
        
        if nama_ok and umur_ok and gender_ok and kasus_ok and self.selected_bed and self.selected_gcs:
            self.btn_start.setEnabled(True)
            self.btn_start.setStyleSheet("""
                QPushButton { 
                    background-color: #214889; 
                    color: #FFFFFF; 
                    font-size: 20px; 
                    font-weight: bold; 
                    border-radius: 8px; 
                    border: none;
                }
                QPushButton:hover { background-color: #163264; }
            """)
        else:
            self.btn_start.setEnabled(False)
            self.btn_start.setStyleSheet("""
                QPushButton { background-color: #A0B09C; color: #FFFFFF; font-size: 20px; font-weight: bold; border-radius: 8px; border: none; }
            """)

    def save_and_emit_data(self):
        gender = "Laki-laki" if self.radio_laki.isChecked() else "Perempuan"
        kasus = "Trauma" if self.radio_trauma.isChecked() else "Non-Trauma"
        
        patient_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bed": self.selected_bed,
            "nama": self.input_nama.text().strip(),
            "umur": self.input_umur.text().strip(),
            "gender": gender,
            "kasus": kasus,
            "gcs": self.selected_gcs
        }
        
        # Logika penyimpanan CSV lokal otomatis
        csv_file = "data_pendaftaran_pasien.csv"
        file_exists = os.path.isfile(csv_file)
        
        try:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=patient_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(patient_data)
            print(f"[LOG] Data pasien '{patient_data['nama']}' berhasil disimpan ke {csv_file}")
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan ke CSV: {e}")
            
        self.measurement_started.emit(patient_data)

# =====================================================================
# BLOK INDEPENDEN TEST RUN (UKURAN LAYAR 1920 x 1080)
# =====================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    test_window = RegistrationPage()
    test_window.resize(1920, 1080)  
    test_window.setWindowTitle("Test Run - Halaman Registrasi Pasien TriaGO")
    test_window.show()
    sys.exit(app.exec())