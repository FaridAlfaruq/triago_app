import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PyQt6.QtCore import Qt

# System Path Integration
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import halaman kustom
from regist_page import RegistrationPage
from plot_page import PlotPage
from loading_page import LoadingPage
from output_page import OutputPage # <-- TAMBAHKAN INI

class TriaGoApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TriaGO - Automated Medical Triage Kiosk")
        self.resize(1024, 650)
        self.setStyleSheet("background-color: #121212; color: #FFFFFF; font-family: 'Segoe UI', Arial, sans-serif;")
        
        # Menyimpan cache sementara data pendaftaran pasien
        self.current_patient_info = {}
        
        # 1. Kontainer Utama QStackedWidget
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # 2. Inisialisasi Semua Instance Halaman Asli
        self.page_registration = RegistrationPage()
        self.page_live_data = PlotPage() 
        self.page_loading = LoadingPage()
        self.page_output = OutputPage() # <-- GANTI PLACEHOLDER LAMA DENGAN INI
        
        # 3. Hubungkan Referensi Parent & Sinyal Antar Halaman
        self.page_loading.parent_main_win = self 
        self.page_registration.measurement_started.connect(self.handle_start_measurement)
        self.page_live_data.recording_finished.connect(self.handle_processing_phase)
        self.page_output.home_requested.connect(self.reset_to_gatekeeper) # Deteksi tombol kembali ke home
        
        # 4. Daftarkan Semua Halaman ke dalam Stack Widget
        self.stacked_widget.addWidget(self.page_registration) # Index 0
        self.stacked_widget.addWidget(self.page_live_data)    # Index 1
        self.stacked_widget.addWidget(self.page_loading)      # Index 2
        self.stacked_widget.addWidget(self.page_output)       # Index 3
        
        self.stacked_widget.setCurrentIndex(0)

    def handle_start_measurement(self, patient_data):
        """Menerima data pendaftaran awal dari Halaman 1"""
        self.current_patient_info = patient_data # Simpan ke memori global aplikasi utama
        self.stacked_widget.setCurrentIndex(1)
        self.page_live_data.start_session(patient_data)

    def handle_processing_phase(self, csv_filepath):
        """Menerima konfirmasi dari Halaman 2 bahwa perekaman sukses"""
        self.stacked_widget.setCurrentIndex(2)
        self.page_loading.start_processing(csv_filepath)

    def handle_output_phase(self, calculation_results):
        """Menangkap hasil komputasi dari Halaman Loading dan mengoper ke Halaman Akhir"""
        print(f"[MAIN LOG] Mengirim data ke tampilan output...")
        
        # Kirim data pasien global & data kalkulasi ke Halaman 4
        self.page_output.display_results(self.current_patient_info, calculation_results)
        
        # Lompat ke tampilan halaman hasil akhir (Index 3)
        self.stacked_widget.setCurrentIndex(3)

    def reset_to_gatekeeper(self):
        """Fungsi sanitasi untuk mengosongkan memori lama dan kembali ke gerbang utama aplikasi"""
        self.current_patient_info.clear()
        # Opsional: di sini kamu bisa panggil fungsi reset form di regist_page jika dibutuhkan
        self.stacked_widget.setCurrentIndex(0)

    def closeEvent(self, event):
        self.page_live_data.close_threads()
        self.page_loading.close_threads()
        event.accept()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TriaGoApplication()
    window.show()
    sys.exit(app.exec())