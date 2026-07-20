import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# System Path Integration
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import window TriaGO
from GUI.home_page import HomePage
from regist_page import RegistrationPage
from plot_page import PlotPage
from loading_page import LoadingPage
from output_page import OutputPage

# =====================================================================
# WORKER ASSISTANT: Khusus Proses Ekstraksi Fitur (Durasi 5 Detik)
# =====================================================================
class ExtractionWorker(QThread):
    """Worker tambahan untuk menghandle loading ekstraksi fitur selama 5 detik"""
    progress_updated = pyqtSignal(str, int)
    extraction_finished = pyqtSignal(dict)

    def __init__(self, csv_filepath):
        super().__init__()
        self.csv_filepath = csv_filepath

    def run(self):
        total_duration = 5.0  
        steps = 100
        interval = total_duration / steps

        for current_step in range(1, steps + 1):
            self.msleep(int(interval * 1000))
            self.progress_updated.emit("Mengekstraksi fitur biosinyal ML...", current_step)
        
        dummy_ml_results = {
            "systolic": 120,
            "diastolic": 80,
            "heart_rate": 110,
            "respiration_rate": 14,
            "temperature": 36.00,
            "spo2": 98,
            "triage_status": "RESUSITASI", 
            "csv_path": self.csv_filepath
        }
        self.extraction_finished.emit(dummy_ml_results)


# =====================================================================
# INTI CORE APLIKASI: TriaGoApplication
# =====================================================================
class TriaGoApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TriaGO - Automated Medical Triage Kiosk")
        self.showMaximized()
        
        self.current_patient_info = {}
        self.saved_csv_path = ""
        
        # 1. Kontainer Utama Stacked Widget
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # 2. Inisialisasi Instance Halaman
        self.page_home = HomePage()
        self.page_registration = RegistrationPage()
        self.page_loading = LoadingPage()
        self.page_live_data = PlotPage() 
        self.page_output = OutputPage()
        
        # 3. Hubungkan Sistem Komunikasi Sinyal (Signals & Slots)
        self.page_home.start_requested.connect(self.go_to_registration)
        self.page_registration.measurement_started.connect(self.handle_start_stabilization_phase)
        
        # --- SINKRONISASI UTAMA: Hubungkan data warmup PlotPage ke LoadingPage ---
        self.page_live_data.warmup_progress.connect(self.page_loading.update_ui_state)
        self.page_live_data.warmup_finished.connect(self.go_to_live_data_page)
        
        # Sinyal pasca-perekaman dan tombol kembali
        self.page_live_data.recording_finished.connect(self.handle_extraction_phase)
        self.page_output.home_requested.connect(self.reset_to_gatekeeper)
        
        # 4. Daftarkan Halaman ke Stacked Widget
        self.stacked_widget.addWidget(self.page_home)          # Index 0
        self.stacked_widget.addWidget(self.page_registration)  # Index 1
        self.stacked_widget.addWidget(self.page_loading)       # Index 2
        self.stacked_widget.addWidget(self.page_live_data)     # Index 3
        self.stacked_widget.addWidget(self.page_output)        # Index 4
        
        self.stacked_widget.setCurrentIndex(0)

    def go_to_registration(self):
        self.stacked_widget.setCurrentIndex(1)

    def handle_start_stabilization_phase(self, patient_data):
        """Fase Pertama: Membuka loading screen dan langsung menyalakan data stream STM32"""
        self.current_patient_info = patient_data 
        
        # Pindah ke Halaman Loading terlebih dahulu
        self.stacked_widget.setCurrentIndex(2)
        self.page_loading.progress_bar.setValue(0)
        self.page_loading.lbl_status.setText("Menstabilkan sensor....")
        
        # Jalankan session data STM32. Proses 2 detik awal otomatis men-drive progress bar
        self.page_live_data.start_session(patient_data)

    def go_to_live_data_page(self):
        """Callback Otomatis: Dipanggil saat data warmup ke-800 dari PlotPage selesai diterima"""
        print("[LOG SUCCESS] Detik ke-2 tercapai secara riil. Membuka halaman plot sinyal.")
        self.stacked_widget.setCurrentIndex(3)

    def handle_extraction_phase(self, csv_filepath):
        """Fase Kedua Loading: Masuk loading kembali selama 5 detik untuk ekstraksi data"""
        self.saved_csv_path = csv_filepath
        
        # Kembalikan tampilan ke Halaman Loading (Index 2)
        self.stacked_widget.setCurrentIndex(2)
        self.page_loading.progress_bar.setValue(0)
        
        # Inisialisasi thread asinkronus ekstraksi fitur 5 detik
        self.extraction_worker = ExtractionWorker(csv_filepath)
        self.extraction_worker.progress_updated.connect(self.page_loading.update_ui_state)
        self.extraction_worker.extraction_finished.connect(self.handle_output_phase)
        self.extraction_worker.start()

    def handle_output_phase(self, calculation_results):
        self.page_output.update_triage_header(calculation_results["triage_status"])
        self.stacked_widget.setCurrentIndex(4)

    def reset_to_gatekeeper(self):
        self.current_patient_info.clear()
        self.saved_csv_path = ""
        
        self.page_registration.input_nama.clear()
        self.page_registration.input_umur.clear()
        
        if self.page_registration.group_gender.checkedButton():
            self.page_registration.group_gender.setExclusive(False)
            self.page_registration.group_gender.checkedButton().setChecked(False)
            self.page_registration.group_gender.setExclusive(True)
            
        if self.page_registration.group_kasus.checkedButton():
            self.page_registration.group_kasus.setExclusive(False)
            self.page_registration.group_kasus.checkedButton().setChecked(False)
            self.page_registration.group_kasus.setExclusive(True)
            
        self.page_registration.selected_bed = None
        self.page_registration.selected_gcs = None
        
        for btn in self.page_registration.bed_buttons.values():
            btn.setChecked(False)
            btn.setStyleSheet("QPushButton { background-color: #FFFFFF; border: 1.5px solid #214889; border-radius: 8px; font-size: 32px; font-weight: bold; color: #214889; } QPushButton:hover { background-color: #F0F4FF; }")
            
        for btn in self.page_registration.gcs_buttons.values():
            btn.setChecked(False)
            btn.setStyleSheet("QPushButton { background-color: #FFFFFF; border: 1px solid #C2D5BB; border-radius: 8px; font-size: 18px; font-weight: bold; color: #A0B09C; } QPushButton:hover { border-color: #214889; color: #214889; }")
            
        self.page_registration.validate_form()
        self.stacked_widget.setCurrentIndex(1)

    def closeEvent(self, event):
        if hasattr(self.page_live_data, 'worker') and self.page_live_data.worker is not None:
            self.page_live_data.worker.stop()
        if hasattr(self, 'extraction_worker') and self.extraction_worker.isRunning():
            self.extraction_worker.quit()
            self.extraction_worker.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TriaGoApplication()
    window.show()
    sys.exit(app.exec())