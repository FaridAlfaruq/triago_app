import sys
import os
import csv
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QGridLayout, QScrollArea,
                             QSizePolicy, QApplication)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPdfWriter, QPainter, QTextDocument, QPageSize


class LockedViewBox(pg.ViewBox):
    """
    ViewBox kustom yang mengunci fitur drag/pan (geser dengan mouse),
    namun tetap membiarkan zoom in/out lewat scroll wheel berfungsi normal.
    """

    def mouseDragEvent(self, ev, axis=None):
        ev.ignore()

    def mouseClickEvent(self, ev):
        ev.ignore()

    def wheelEvent(self, ev, axis=None):
        super().wheelEvent(ev, axis=axis)


class OutputPage(QWidget):
    # Sinyal untuk meriset sistem dan kembali ke halaman pendaftaran awal
    home_requested = pyqtSignal()

    DISPLAY_DURATION_SEC = 5
    SAMPLE_RATE_HZ = 400

    def __init__(self):
        super().__init__()
        self.patient_data = {}
        self.calculation_results = {}
        self.setup_ui()

    def setup_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(25, 20, 25, 20)
        outer_layout.setSpacing(15)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                background: #1E1E1E; width: 10px; border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #3A3A3A; border-radius: 5px; min-height: 30px;
            }
            QScrollBar::handle:vertical:hover { background: #4A4A4A; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(15)

        # =========================================================================
        # 1. HEADER: BANNER STATUS TRIASE MASIF
        # =========================================================================
        self.triage_banner = QFrame()
        self.triage_banner.setFixedHeight(60)
        self.triage_banner.setStyleSheet("background-color: #2C2C2C; border-radius: 8px;")
        banner_layout = QVBoxLayout(self.triage_banner)

        self.lbl_triage_status = QLabel("MENUNGGU DATA HASIL TRIASE...")
        self.lbl_triage_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_triage_status.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: white; letter-spacing: 1px;"
        )
        banner_layout.addWidget(self.lbl_triage_status)
        content_layout.addWidget(self.triage_banner)

        # =========================================================================
        # 2. BODY SECTION: SPLIT VIEW (KIRI: DATA | KANAN: GRAFIK DATA ASLI)
        # =========================================================================
        body_layout = QHBoxLayout()
        body_layout.setSpacing(20)

        # --- KOLOM KIRI: RINGKASAN DATA & TANDA VITAL PASIEN ---
        left_container = QFrame()
        left_container.setStyleSheet("background-color: #1E1E1E; border-radius: 8px; padding: 15px;")
        left_layout = QVBoxLayout(left_container)
        left_layout.setSpacing(12)

        left_layout.addWidget(QLabel("<font size='4' color='#BDC3C7'><b>Ringkasan Medis Pasien</b></font>"))

        info_grid = QGridLayout()
        info_grid.setSpacing(10)

        self.lbl_out_nama = QLabel("Nama: -")
        self.lbl_out_umur = QLabel("Umur: -")
        self.lbl_out_gender = QLabel("Jenis Kelamin: -")
        self.lbl_out_kasus = QLabel("Kategori Kasus: -")
        self.lbl_out_gcs = QLabel("Skor GCS: -")
        self.lbl_out_temp = QLabel("Suhu Tubuh: -")

        self.lbl_out_bp = QLabel("Tekanan Darah: --/-- mmHg")
        self.lbl_out_hr = QLabel("Detak Jantung (HR): -- BPM")
        self.lbl_out_spo2 = QLabel("Saturasi Oksigen (SpO2): -- %")

        labels_to_style = [self.lbl_out_nama, self.lbl_out_umur, self.lbl_out_gender,
                           self.lbl_out_kasus, self.lbl_out_gcs, self.lbl_out_bp,
                           self.lbl_out_hr, self.lbl_out_spo2, self.lbl_out_temp]
        for lbl in labels_to_style:
            lbl.setStyleSheet("font-size: 14px; color: #E0E0E0;")

        self.lbl_out_bp.setStyleSheet("font-size: 14px; color: #E0E0E0;")
        self.lbl_out_hr.setStyleSheet("font-size: 14px; color: #E0E0E0;")
        self.lbl_out_spo2.setStyleSheet("font-size: 14px; color: #E0E0E0;")

        info_grid.addWidget(self.lbl_out_nama, 0, 0)
        info_grid.addWidget(self.lbl_out_umur, 0, 1)
        info_grid.addWidget(self.lbl_out_gender, 1, 0)
        info_grid.addWidget(self.lbl_out_kasus, 1, 1)
        info_grid.addWidget(self.lbl_out_gcs, 2, 0)
        info_grid.addWidget(self.lbl_out_temp, 2, 1)

        left_layout.addLayout(info_grid)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #333;")
        left_layout.addWidget(line)

        left_layout.addWidget(QLabel("<font size='3' color='#BDC3C7'><b>Hasil Ekstraksi Parameter Klinis:</b></font>"))
        left_layout.addWidget(self.lbl_out_bp)
        left_layout.addWidget(self.lbl_out_hr)
        left_layout.addWidget(self.lbl_out_spo2)
        left_layout.addStretch()

        body_layout.addWidget(left_container, stretch=4)

        # --- KOLOM KANAN: OUTPUT GRAFIK (DISESUAIKAN) ---
        right_container = QFrame()
        right_container.setStyleSheet("background-color: #1E1E1E; border-radius: 8px; padding: 12px;")
        right_layout = QVBoxLayout(right_container)

        self.win_preview = pg.GraphicsLayoutWidget()
        self.win_preview.setBackground('#1E1E1E')
        self.win_preview.setMinimumHeight(320)
        self.win_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.win_preview)

        # Plot ECG (Hijau) - Judul disederhanakan
        self.p_ecg = self.win_preview.addPlot(title="Sinyal ECG", viewBox=LockedViewBox())
        self.p_ecg.showGrid(x=True, y=True)
        self.p_ecg.setMouseEnabled(x=True, y=True)  
        self.p_ecg.setMenuEnabled(False)
        self.p_ecg.setLabel('left', 'Amplitudo')
        self.curve_ecg = self.p_ecg.plot(pen=pg.mkPen('g', width=1.5))

        self.win_preview.nextRow()

        # Plot PPG (Merah) - Judul disederhanakan
        self.p_ppg = self.win_preview.addPlot(title="Sinyal PPG", viewBox=LockedViewBox())
        self.p_ppg.showGrid(x=True, y=True)
        self.p_ppg.setMouseEnabled(x=True, y=True)
        self.p_ppg.setMenuEnabled(False)
        self.p_ppg.setLabel('left', 'Amplitudo')
        self.p_ppg.setLabel('bottom', 'Waktu', units='s')
        self.curve_ppg = self.p_ppg.plot(pen=pg.mkPen('r', width=1.5))
        self.p_ppg.setXLink(self.p_ecg)

        body_layout.addWidget(right_container, stretch=5)

        content_layout.addLayout(body_layout, stretch=1)

        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area, stretch=1)

        # =========================================================================
        # 3. FOOTER SECTION: BUTTON NAVIGASI (EMOTICON DIHAPUS)
        # =========================================================================
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(15)

        self.btn_export_pdf = QPushButton("CETAK STRIP TRIASE (PDF)")
        self.btn_export_pdf.setFixedHeight(48)
        self.btn_export_pdf.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export_pdf.setStyleSheet(
            "QPushButton { background-color: #3498DB; color: white; font-size: 15px; font-weight: bold; border-radius: 8px; }"
            "QPushButton:hover { background-color: #2980B9; }"
            "QPushButton:pressed { background-color: #21618C; }"
        )
        self.btn_export_pdf.clicked.connect(self.export_to_pdf)

        self.btn_home = QPushButton("SELESAI & PASIEN BARU")
        self.btn_home.setFixedHeight(48)
        self.btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_home.setStyleSheet(
            "QPushButton { background-color: #2ECC71; color: white; font-size: 15px; font-weight: bold; border-radius: 8px; }"
            "QPushButton:hover { background-color: #27AE60; }"
            "QPushButton:pressed { background-color: #1E8449; }"
        )
        self.btn_home.clicked.connect(self.handle_home_click)

        footer_layout.addWidget(self.btn_export_pdf, stretch=1)
        footer_layout.addWidget(self.btn_home, stretch=1)

        outer_layout.addLayout(footer_layout)

    def display_results(self, patient_info, ml_results):
        """Menerima dan merender seluruh parameter klinis dari berkas data asli"""
        self.patient_data = patient_info
        self.calculation_results = ml_results

        self.lbl_out_nama.setText(f"<b>Nama:</b> {patient_info['nama']}")
        self.lbl_out_umur.setText(f"<b>Umur:</b> {patient_info['umur']} Tahun")
        self.lbl_out_gender.setText(f"<b>Gender:</b> {patient_info['gender']}")
        self.lbl_out_kasus.setText(f"<b>Kasus:</b> {patient_info['kasus']}")
        self.lbl_out_gcs.setText(f"<b>Skor GCS:</b> {patient_info['gcs']}")

        self.lbl_out_bp.setText(f"Tekanan Darah: {ml_results['systolic']}/{ml_results['diastolic']} mmHg")
        self.lbl_out_hr.setText(f"Detak Jantung (HR): {ml_results['heart_rate']} BPM")
        self.lbl_out_spo2.setText(f"Saturasi Oksigen (SpO2): {ml_results['spo2']} %")

        csv_path = ml_results.get("csv_path", "")
        time_data, ecg_data, ppg_data, avg_temp = self.load_raw_csv_data(csv_path)

        self.lbl_out_temp.setText(f"<b>Suhu Tubuh:</b> {avg_temp:.2f} °C")

        if time_data:
            self.curve_ecg.setData(time_data, ecg_data)
            self.curve_ppg.setData(time_data, ppg_data)
            x_start = time_data[0]
            x_end = x_start + self.DISPLAY_DURATION_SEC
            self.p_ecg.setXRange(x_start, x_end, padding=0)
            self.p_ecg.enableAutoRange(axis='y')
            self.p_ppg.enableAutoRange(axis='y')

        status = ml_results["triage_status"].upper()
        if status == "SEVERE":
            self.lbl_triage_status.setText("RESUSITASI")
            self.triage_banner.setStyleSheet("background-color: #E74C3C; border-radius: 8px;")
        elif status == "MODERATE":
            self.lbl_triage_status.setText("DARURAT")
            self.triage_banner.setStyleSheet("background-color: #F39C12; border-radius: 8px;")
        else:
            self.lbl_triage_status.setText("NON-DARURAT")
            self.triage_banner.setStyleSheet("background-color: #2ECC71; border-radius: 8px;")

    def load_raw_csv_data(self, csv_filepath):
        time_data = []
        ecg_data = []
        ppg_data = []
        temp_sum = 0
        row_count = 0

        if not csv_filepath or not os.path.exists(csv_filepath):
            return [], [], [], 36.60

        max_samples = int(self.DISPLAY_DURATION_SEC * self.SAMPLE_RATE_HZ)

        try:
            with open(csv_filepath, mode='r') as file:
                reader = csv.reader(file)
                header = next(reader)  

                t_idx = header.index("Time (s)")
                ppg_idx = header.index("PPG_Red")
                ecg_idx = header.index("ECG")
                temp_idx = header.index("Temp_Object")

                for row in reader:
                    if not row:
                        continue

                    row_count += 1
                    t_val = float(row[t_idx])
                    t_obj = float(row[temp_idx])
                    temp_sum += t_obj

                    if len(time_data) < max_samples:
                        time_data.append(t_val)
                        ecg_data.append(float(row[ecg_idx]))
                        ppg_data.append(float(row[ppg_idx]))

            avg_temp = temp_sum / row_count if row_count > 0 else 36.60
            return time_data, ecg_data, ppg_data, avg_temp

        except Exception as e:
            print(f"[ERROR] Gagal memproses visualisasi CSV: {e}")
            return [], [], [], 36.60

    def export_to_pdf(self):
        """Membuat berkas PDF format A4 dengan skala layout yang proporsional dan tidak pipih"""
        if not self.patient_data:
            return

        filename = f"Strip_Triase_{self.patient_data['nama']}.pdf"
        pdf_writer = QPdfWriter(filename)
        
        # 1. Set Kertas ke A4
        pdf_writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        
        # 2. Atur Margin (20mm di setiap sisi agar layout seimbang)
        from PyQt6.QtCore import QMarginsF
        pdf_writer.setPageMargins(QMarginsF(20.0, 20.0, 20.0, 20.0))

        painter = QPainter(pdf_writer)
        doc = QTextDocument()

        # =========================================================================
        # SOLUSI MUTLAK: PENERAPAN SKALA RESOLUSI KANVAS A4
        # Mengunci kerapatan piksel dokumen agar font dan tabel membesar proporsional.
        # =========================================================================
        dpi = pdf_writer.resolution()
        scale_factor = dpi / 96.0  # Konversi skala dari standar layar ke DPI printer
        
        # Ambil total lebar area cetak pixel yang tersedia dari printer
        page_rect = pdf_writer.pageLayout().paintRectPixels(dpi)
        
        # Atur lebar internal dokumen HTML (dalam basis skala rasional)
        doc.setTextWidth(page_rect.width() / scale_factor)
        
        # Paksa painter untuk melakukan scaling sebelum menggambar konten HTML
        painter.scale(scale_factor, scale_factor)
        status = self.calculation_results.get("triage_status", "").upper()
        if status == "SEVERE":
            bg_color = "#E74C3C"  # Merah (Resusitasi)
        elif status == "MODERATE":
            bg_color = "#F39C12"  # Oranye/Kuning (Darurat)
        else:
            bg_color = "#2ECC71"  # Hijau (Non-Darurat)
        suhu_text = self.lbl_out_temp.text().replace("<b>Suhu Tubuh:</b> ", "")

        # 3. HTML Content: Menggunakan ukuran standar px yang sekarang sudah ter-scaling otomatis
        html_content = f"""
        <div style='font-family: Arial, sans-serif; font-size: 14px; color: #000000; line-height: 1.6;'>
            <h1 style='text-align: center; color: #2C3E50; margin-bottom: 5px; font-size: 26px;'>TriaGO MEDICAL REPORT</h1>
            <hr style='border: 1px solid #34495E; margin-bottom: 20px;'>
            <h2 style='color: #2980B9; border-bottom: 2px solid #BDC3C7; padding-bottom: 5px; font-size: 18px; margin-top: 25px;'>1. Identitas Pasien</h2>
            <table width='100%' cellpadding='8' style='margin-bottom: 20px; font-size: 14px;'>
                <tr>
                    <td width='50%'><b>Nama Pasien:</b> {self.patient_data['nama']}</td>
                    <td width='50%'><b>Umur:</b> {self.patient_data['umur']} Tahun</td>
                </tr>
                <tr>
                    <td><b>Jenis Kelamin:</b> {self.patient_data['gender']}</td>
                    <td><b>Kategori Kasus:</b> {self.patient_data['kasus']}</td>
                </tr>
                <tr>
                    <td><b>Skor GCS:</b> {self.patient_data['gcs']}</td>
                    <td><b>Rata-rata Suhu Tubuh:</b> {suhu_text}</td>
                </tr>
            </table>
            
            <h2 style='color: #2980B9; border-bottom: 2px solid #BDC3C7; padding-bottom: 5px; font-size: 18px; margin-top: 25px;'>2. Hasil Parameter Tanda Vital</h2>
            <table width='100%' cellpadding='10' style='margin-bottom: 30px; font-size: 14px; border-collapse: collapse;'>
                <tr style='background-color: #F8F9F9;'>
                    <td width='40%' style='border-bottom: 1px solid #ddd;'><b>Parameter Klinis</b></td>
                    <td width='60%' style='border-bottom: 1px solid #ddd;'><b>Nilai Hasil Ukur</b></td>
                </tr>
                <tr>
                    <td style='border-bottom: 1px solid #ddd;'>Tekanan Darah</td>
                    <td style='border-bottom: 1px solid #ddd;'>{self.calculation_results['systolic']}/{self.calculation_results['diastolic']} mmHg</td>
                </tr>
                <tr style='background-color: #F8F9F9;'>
                    <td style='border-bottom: 1px solid #ddd;'>Detak Jantung (HR)</td>
                    <td style='border-bottom: 1px solid #ddd;'>{self.calculation_results['heart_rate']} BPM</td>
                </tr>
                <tr>
                    <td style='border-bottom: 1px solid #ddd;'>Saturasi Oksigen (SpO2)</td>
                    <td style='border-bottom: 1px solid #ddd;'>{self.calculation_results['spo2']}%</td>
                </tr>
            </table>
            
            <br><br>
            <div style='text-align: center; padding: 20px; background-color: #EAEDED; border: 1px solid #BDC3C7; border-radius: 6px; margin-top: 40px;'>
                <span style='font-size: 16px; font-weight: bold; color: #2C3E50;'>
                    STATUS AKHIR:<br><br>
                    <span style='font-size: 18px; color: #FFFFFF; background-color: {bg_color}; padding: 10px 18px; border-radius: 4px;'>
                        {self.lbl_triage_status.text()}
                    </span>
                </span>
            </div>
        </div>
        """
        doc.setHtml(html_content)
        
        # Render HTML ke kanvas printer yang sudah diskalakan
        doc.drawContents(painter)
        painter.end()
        print(f"[INFO] Sukses cetak PDF format A4 proporsional ke: {filename}")

    def handle_home_click(self):
        self.curve_ecg.clear()
        self.curve_ppg.clear()
        self.home_requested.emit()

# =========================================================================
# BLOK PENGETESAN MANDIRI (LOCAL TESTING BLOCK)
# =========================================================================
if __name__ == "__main__":
    # Inisialisasi aplikasi PyQt
    app = QApplication(sys.argv)
    
    # Set style gelap agar serupa dengan main_gui utama
    app.setStyleSheet("background-color: #121212; color: #FFFFFF; font-family: 'Segoe UI', Arial, sans-serif;")
    
    # Buat instance widget output page
    test_window = OutputPage()
    test_window.setWindowTitle("TriaGO - Output Page Test Environment")
    test_window.resize(1024, 650)
    
    # 1. Siapkan struktur data dummy registrasi pasien (Halaman 1)
    dummy_patient_info = {
        "nama": "Aisyah Test",
        "umur": "24",
        "gender": "Perempuan",
        "kasus": "Trauma",
        "gcs": "12"
    }
    
    # 2. Siapkan struktur data dummy hasil pemrosesan model (Halaman 3)
    # Pastikan file 'data_Test3_20260718_163310.csv' berada di folder yang sama dengan script ini
    dummy_ml_results = {
        "systolic": 118,
        "diastolic": 76,
        "heart_rate": 82,
        "spo2": 99,
        "triage_status": "SEVERE", # Coba ganti: "MILD", "MODERATE", atau "SEVERE"
        "csv_path": "data_Test4_20260718_163546.csv" 
    }
    
    # Hubungkan sinyal tombol home/kembali untuk melihat respon di terminal
    test_window.home_requested.connect(lambda: print("[TEST] Tombol Selesai & Pasien Baru Diklik!"))
    
    # Eksekusi fungsi display untuk langsung merender data asli dari CSV
    test_window.display_results(dummy_patient_info, dummy_ml_results)
    
    # Tampilkan jendela pengujian
    test_window.show()
    sys.exit(app.exec())