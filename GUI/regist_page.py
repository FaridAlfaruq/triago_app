from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QGridLayout, QFrame, 
                             QRadioButton, QSizePolicy, QButtonGroup)
from PyQt6.QtCore import Qt, pyqtSignal

class RegistrationPage(QWidget):
    # Buat custom signal untuk mengirim data ke main_gui saat tombol start diklik
    measurement_started = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.selected_gcs = None
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 20, 30, 20)
        main_layout.setSpacing(15)
        
        # --- HEADER ---
        header_label = QLabel("TriaGO — PENDAFTARAN & TRIASE AWAL")
        header_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2ECC71; letter-spacing: 1px;")
        main_layout.addWidget(header_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        body_layout = QHBoxLayout()
        body_layout.setSpacing(30)
        
        # --- KOLOM KIRI: FORM IDENTITAS ---
        form_container = QFrame()
        form_container.setStyleSheet("background-color: #1E1E1E; border-radius: 8px; padding: 20px;")
        form_layout = QVBoxLayout(form_container)
        form_layout.setSpacing(15)
        
        form_layout.addWidget(QLabel("<font size='4' color='#BDC3C7'><b>Identitas Pasien</b></font>"))
        
        form_layout.addWidget(QLabel("Nama Lengkap Pasien:"))
        self.input_nama = QLineEdit(placeholderText="Masukkan nama...")
        self.input_nama.setStyleSheet("background-color: #2C2C2C; border: 1px solid #444; padding: 10px; border-radius: 4px; color: white; font-size: 14px;")
        form_layout.addWidget(self.input_nama)
        
        form_layout.addWidget(QLabel("Umur Pasien (Tahun):"))
        self.input_umur = QLineEdit(placeholderText="Masukkan umur...")
        self.input_umur.setStyleSheet("background-color: #2C2C2C; border: 1px solid #444; padding: 10px; border-radius: 4px; color: white; font-size: 14px;")
        form_layout.addWidget(self.input_umur)
        
        radio_style = """
        QRadioButton { color: white; font-size: 14px; padding-left: 8px; min-height: 30px; }
        QRadioButton::indicator { width: 18px; height: 18px; border: 2px solid #555555; border-radius: 10px; background-color: #2C2C2C; }
        QRadioButton::indicator:hover { border-color: #7F8C8D; }
        QRadioButton::indicator:checked { border-color: #3498DB; background-color: #2980B9; image: url(none); }
        """
        
        form_layout.addWidget(QLabel("Jenis Kelamin:"))
        layout_gender = QHBoxLayout()
        layout_gender.setSpacing(20)
        self.radio_laki = QRadioButton("Laki-laki")
        self.radio_perempuan = QRadioButton("Perempuan")
        self.radio_laki.setStyleSheet(radio_style)
        self.radio_perempuan.setStyleSheet(radio_style)
        
        self.group_gender = QButtonGroup(self)
        self.group_gender.addButton(self.radio_laki)
        self.group_gender.addButton(self.radio_perempuan)
        
        layout_gender.addWidget(self.radio_laki)
        layout_gender.addWidget(self.radio_perempuan)
        layout_gender.addStretch()
        form_layout.addLayout(layout_gender)
        
        form_layout.addWidget(QLabel("Kategori Kasus Pasien:"))
        layout_kasus = QHBoxLayout()
        layout_kasus.setSpacing(20)
        self.radio_nontrauma = QRadioButton("Non-Trauma")
        self.radio_trauma = QRadioButton("Trauma")
        self.radio_nontrauma.setStyleSheet(radio_style)
        self.radio_trauma.setStyleSheet(radio_style)
        
        self.group_kasus = QButtonGroup(self)
        self.group_kasus.addButton(self.radio_nontrauma)
        self.group_kasus.addButton(self.radio_trauma)
        
        layout_kasus.addWidget(self.radio_nontrauma)
        layout_kasus.addWidget(self.radio_trauma)
        layout_kasus.addStretch()
        form_layout.addLayout(layout_kasus)
        
        form_layout.addStretch()
        body_layout.addWidget(form_container, stretch=4)
        
        # --- KOLOM KANAN: GCS GRID ---
        gcs_container = QFrame()
        gcs_container.setStyleSheet("background-color: #1E1E1E; border-radius: 8px; padding: 20px;")
        gcs_layout = QVBoxLayout(gcs_container)
        
        gcs_layout.addWidget(QLabel("<font size='4' color='#BDC3C7'><b>Glasgow Coma Scale (GCS)</b></font>"))
        gcs_layout.addWidget(QLabel("<font size='2' color='#7F8C8D'>Pilih skor kesadaran fisik pasien:</font>"))
        gcs_layout.addSpacing(15)
        
        grid_gcs = QGridLayout()
        grid_gcs.setSpacing(10)
        self.gcs_buttons = {}
        
        for index, score in enumerate(range(3, 16)):
            row = index // 4
            col = index % 4
            btn = QPushButton(str(score))
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            btn.setMinimumSize(70, 50)
            btn.setStyleSheet(
                "QPushButton { background-color: #2C2C2C; border: 2px solid #3A3A3A; border-radius: 6px; font-size: 16px; font-weight: bold; color: #E0E0E0; }"
                "QPushButton:hover { background-color: #3D3D3D; border-color: #7F8C8D; }"
            )
            btn.clicked.connect(lambda checked, s=score: self.handle_gcs_selection(s))
            grid_gcs.addWidget(btn, row, col)
            self.gcs_buttons[score] = btn
            
        for r in range(4): grid_gcs.setRowStretch(r, 1)
        for c in range(4): grid_gcs.setColumnStretch(c, 1)
            
        gcs_layout.addLayout(grid_gcs)
        body_layout.addWidget(gcs_container, stretch=5)
        main_layout.addLayout(body_layout)
        
        # --- ACTION BUTTON ---
        self.btn_start = QPushButton("MULAI PENGUKURAN")
        self.btn_start.setEnabled(False)
        self.btn_start.setStyleSheet("QPushButton { background-color: #7F8C8D; color: #333333; font-size: 18px; font-weight: bold; padding: 15px; border-radius: 6px; }")
        self.btn_start.clicked.connect(self.emit_start_signal)
        main_layout.addWidget(self.btn_start)
        
        # SIGNALS CONNECT
        self.input_nama.textChanged.connect(self.validate_form)
        self.input_umur.textChanged.connect(self.validate_form)
        self.radio_laki.toggled.connect(self.validate_form)
        self.radio_perempuan.toggled.connect(self.validate_form)
        self.radio_nontrauma.toggled.connect(self.validate_form)
        self.radio_trauma.toggled.connect(self.validate_form)

    def handle_gcs_selection(self, selected_score):
        self.selected_gcs = selected_score
        for score, btn in self.gcs_buttons.items():
            btn.setStyleSheet("QPushButton { background-color: #2C2C2C; border: 2px solid #3A3A3A; border-radius: 6px; font-size: 16px; font-weight: bold; color: #E0E0E0; } QPushButton:hover { background-color: #3D3D3D; border-color: #7F8C8D; }")
        
        if 13 <= selected_score <= 15: active_color, border_color = "#2ECC71", "#27AE60"
        elif 9 <= selected_score <= 12: active_color, border_color = "#F39C12", "#D35400"
        else: active_color, border_color = "#E74C3C", "#C0392B"
            
        self.gcs_buttons[selected_score].setStyleSheet(f"QPushButton {{ background-color: {active_color}; border: 2px solid {border_color}; border-radius: 6px; font-size: 16px; font-weight: bold; color: #FFFFFF; padding: 5px; }}")
        self.validate_form()

    def validate_form(self):
        gender_checked = self.radio_laki.isChecked() or self.radio_perempuan.isChecked()
        kasus_checked = self.radio_nontrauma.isChecked() or self.radio_trauma.isChecked()
        
        if self.input_nama.text().strip() and self.input_umur.text().strip() and self.selected_gcs is not None and gender_checked and kasus_checked:
            self.btn_start.setEnabled(True)
            self.btn_start.setStyleSheet("QPushButton { background-color: #2ECC71; color: white; font-size: 18px; font-weight: bold; padding: 15px; border-radius: 6px; } QPushButton:hover { background-color: #27AE60; }")
        else:
            self.btn_start.setEnabled(False)
            self.btn_start.setStyleSheet("QPushButton { background-color: #7F8C8D; color: #333333; font-size: 18px; font-weight: bold; padding: 15px; border-radius: 6px; }")

    def emit_start_signal(self):
        gender = "Laki-laki" if self.radio_laki.isChecked() else "Perempuan"
        kasus = "Non-Trauma" if self.radio_nontrauma.isChecked() else "Trauma"
        
        patient_data = {
            "nama": self.input_nama.text(),
            "umur": self.input_umur.text(),
            "gender": gender,
            "kasus": kasus,
            "gcs": self.selected_gcs
        }
        # Lepaskan sinyal data pasien ke main_gui
        self.measurement_started.emit(patient_data)