import sys
import os
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PyQt6.QtCore import Qt

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
        
        # Sambungkan reference parent ke LoadingPage
        self.page_loading.parent_main_win = self
        
        # 3. Hubungkan Sistem Komunikasi Sinyal (Signals & Slots)
        self.page_home.start_requested.connect(self.go_to_registration)
        self.page_registration.measurement_started.connect(self.handle_start_stabilization_phase)
        
        # --- SINKRONISASI UTAMA: Warmup PlotPage ke LoadingPage ---
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
        """Fase Pertama: Membuka loading screen dan menyalakan data stream STM32"""
        self.current_patient_info = patient_data 
        
        # Pindah ke Halaman Loading
        self.stacked_widget.setCurrentIndex(2)
        self.page_loading.progress_bar.setValue(0)
        self.page_loading.lbl_status.setText("Menstabilkan sensor....")
        
        # Jalankan session data STM32
        self.page_live_data.start_session(patient_data)

    def go_to_live_data_page(self):
        """Callback Otomatis: Dipanggil saat data warmup ke-800 dari PlotPage selesai diterima"""
        print("[LOG SUCCESS] Detik ke-2 tercapai secara riil. Membuka halaman plot sinyal.")
        self.stacked_widget.setCurrentIndex(3)

    def handle_extraction_phase(self, csv_filepath):
        """Fase Kedua: Membaca CSV hasil perekaman & memproses sinyal riil di LoadingPage"""
        self.saved_csv_path = csv_filepath
        
        # Pindah tampilan ke Halaman Loading (Index 2)
        self.stacked_widget.setCurrentIndex(2)
        
        try:
            # 1. Membaca data CSV yang baru saja direkam dari PlotPage
            df = pd.read_csv(csv_filepath)
            
            # Mapping kolom CSV secara otomatis
            raw_time = df['time'].values if 'time' in df.columns else df.iloc[:, 0].values
            raw_ecg = df['ecg'].values if 'ecg' in df.columns else df.iloc[:, 1].values
            raw_red = df['red'].values if 'red' in df.columns else (df.iloc[:, 2].values if df.shape[1] > 2 else None)
            raw_ir = df['ir'].values if 'ir' in df.columns else (df.iloc[:, 3].values if df.shape[1] > 3 else None)

            # 2. Jalankan pemrosesan sinyal riil (ECG & PPG) via LoadingPage
            self.page_loading.start_processing(
                raw_ecg=raw_ecg,
                raw_time=raw_time,
                raw_red=raw_red,
                raw_ir=raw_ir,
                fs_orig=400  # Sesuaikan dengan sampling rate hardware STM32
            )

            # 3. Hubungkan sinyal selesai ke handle_output_phase
            self.page_loading.worker.processing_finished.connect(self.handle_output_phase)

        except Exception as e:
            print(f"[ERROR] Gagal membaca CSV / Memproses data: {e}")

    def handle_output_phase(self, calculation_results):
        print("[LOG SUCCESS] Mengirimkan data sinyal dan parameter ke Output Page...")
        
        # Kirim seluruh dictionary (termasuk array sinyal) ke OutputPage
        if hasattr(self.page_output, "update_results"):
            self.page_output.update_results(calculation_results)
            
        triage_status = calculation_results.get("triage_status", "NON URGENSI")
        self.page_output.update_triage_header(triage_status)
        
        # Pindah ke Halaman Output
        self.stacked_widget.setCurrentIndex(4)

    def reset_to_gatekeeper(self):
        self.current_patient_info.clear()
        self.saved_csv_path = ""
            
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
        if hasattr(self.page_loading, 'worker') and self.page_loading.worker is not None and self.page_loading.worker.isRunning():
            self.page_loading.worker.quit()
            self.page_loading.worker.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TriaGoApplication()
    window.show()
    sys.exit(app.exec())