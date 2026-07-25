import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PyQt6.QtCore import Qt

# System Path Integration
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import seluruh halaman TriaGO
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
        
        # 1. Kontainer Utama Stacked Widget
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # 2. Inisialisasi Instance Halaman
        self.page_home = HomePage()
        self.page_registration = RegistrationPage()
        self.page_loading = LoadingPage()
        self.page_live_data = PlotPage() 
        self.page_output = OutputPage()
        
        # Connect reference parent ke LoadingPage
        self.page_loading.parent_main_win = self
        
        # 3. Hubungkan Sistem Komunikasi Sinyal (Signals & Slots)
        self.page_home.start_requested.connect(self.go_to_registration)
        self.page_registration.measurement_started.connect(self.handle_start_stabilization_phase)
        
        # --- SINKRONISASI WARMUP: PlotPage ke LoadingPage ---
        self.page_live_data.warmup_progress.connect(self.page_loading.update_ui_state)
        self.page_live_data.warmup_finished.connect(self.go_to_live_data_page)
        
        # --- SINKRONISASI PEREKAMAN: Stream data RAM ke LoadingPage ---
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
        """Pindah ke Halaman Registrasi (Kasur & GCS)"""
        self.stacked_widget.setCurrentIndex(1)

    def handle_start_stabilization_phase(self, patient_data):
        """Fase 1: Membuka loading screen dan menyalakan data stream STM32 (Warmup 2 detik)"""
        self.current_patient_info = patient_data 
        
        # Pindah ke Halaman Loading
        self.stacked_widget.setCurrentIndex(2)
        self.page_loading.progress_bar.setValue(0)
        self.page_loading.lbl_status.setText("Menstabilkan sensor....")
        
        # Jalankan session data STM32 pada PlotPage
        self.page_live_data.start_session(patient_data)

    def go_to_live_data_page(self):
        """Callback Otomatis: Dipanggil saat detik ke-2 ( warm-up 800 sampel) tercapai"""
        print("[LOG SUCCESS] Detik ke-2 tercapai secara riil. Membuka halaman plot sinyal.")
        self.stacked_widget.setCurrentIndex(3)

    def handle_extraction_phase(self, raw_data_list):
        """Fase 2: Membaca list paket data mentah dari RAM dan mengolahnya di LoadingPage"""
        self.stacked_widget.setCurrentIndex(2)
        
        try:
            sampling_rate = 400.0
            sampling_interval = 1.0 / sampling_rate
            n_samples = len(raw_data_list)
            
            raw_time = np.array([i * sampling_interval for i in range(n_samples)])
            raw_ecg = np.array([p["ecg"] for p in raw_data_list])
            raw_red = np.array([p["ppg"]["red"] for p in raw_data_list])
            raw_ir = np.array([p["ppg"]["ir"] for p in raw_data_list])

            # Debug log verifikasi data suhu yang dihitung oleh plot_page.py
            skin_temp = self.current_patient_info.get("temp_skin")
            amb_temp = self.current_patient_info.get("temp_ambient")
            print(f"[LOG MAIN_GUI] Data Suhu Pasien Siap -> Kulit: {skin_temp}°C | Lingkungan: {amb_temp}°C")

            # Oper data ke LoadingPage
            self.page_loading.start_processing(
                raw_ecg=raw_ecg,
                raw_time=raw_time,
                raw_red=raw_red,
                raw_ir=raw_ir,
                patient_info=self.current_patient_info,
                fs_orig=400
            )

        except Exception as e:
            print(f"[ERROR] Gagal mengolah data RAM di LoadingPage: {e}")

    def handle_output_phase(self, calculation_results):
        """Fase 3: Menyimpan 1 File CSV Konsolidasi Tunggal & Buka Halaman Output"""
        print("[LOG SUCCESS] Memproses fase output akhir...")
        
        # 1. SIMPAN 1 FILE CSV KONSOLIDASI MASTER
        self.save_consolidated_csv(calculation_results)
        
        # 2. Update Halaman Output (Grafik 5s, Parameter, SHAP, & JSON IoT)
        if hasattr(self.page_output, "update_results"):
            self.page_output.update_results(calculation_results)
            
        triage_status = calculation_results.get("triage_status", "NON-DARURAT")
        self.page_output.update_triage_header(triage_status)
        
        # 3. Pindah ke Halaman Output (Index 4)
        self.stacked_widget.setCurrentIndex(4)

    def save_consolidated_csv(self, results):
        """Menyimpan seluruh data registrasi, sinyal bersih, dan fitur ekstraksi ke 1 CSV master."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        bed_id = results.get("bed", "00")
        
        # FORMAT NAMA FILE MASTER (Bisa disesuaikan jika ingin lokasi folder tertentu)
        filename = f"TriaGO_FullData_Bed{bed_id}_{timestamp_str}.csv"
        
        try:
            time_arr = results.get("time_125", [])
            ecg_clean = results.get("ecg_smooth", [])
            ir_clean = results.get("ir_clean", [])
            
            # Gabungkan Sinyal Time-Series + Metadata & Hasil Ekstraksi
            df = pd.DataFrame({
                "Time_s": time_arr,
                "ECG_Clean": ecg_clean,
                "PPG_IR_Clean": ir_clean,
                
                # Metadata & Input Registrasi
                "Bed_Location": results.get("bed", "00"),
                "GCS_Score": results.get("gcs", 15),
                "Timestamp": results.get("timestamp", ""),
                
                # Parameter Hasil Ekstraksi & ML
                "Suhu_C": results.get("temperature", 36.5),
                "SpO2_Pct": results.get("spo2", 0.0),
                "RR_RPM": results.get("rr", 0.0),
                "HR_BPM": results.get("hr", 0.0),
                "BP_Systolic": results.get("systolic", 120),
                "BP_Diastolic": results.get("diastolic", 80),
                "PI_Red_Pct": results.get("pi_red", 0.0),
                "PI_IR_Pct": results.get("pi_ir", 0.0),
                "Triage_Status": results.get("triage_status", "")
            })
            
            df.to_csv(filename, index=False)
            print(f"[LOG SUCCESS] File CSV Konsolidasi Berhasil Disimpan: {filename}")
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan CSV konsolidasi: {e}")

    def reset_to_gatekeeper(self):
        """Reset seluruh input data pasien dan kembalikan tampilan ke Halaman Registrasi"""
        self.current_patient_info.clear()
            
        self.page_registration.selected_bed = None
        self.page_registration.selected_gcs = None
        
        for btn in self.page_registration.bed_buttons.values():
            btn.setChecked(False)
            btn.setStyleSheet("""
                QPushButton { 
                    background-color: #FFFFFF; 
                    border: 2px solid #214889; 
                    border-radius: 12px; 
                    font-size: 48px; 
                    font-weight: bold; 
                    color: #214889; 
                } 
                QPushButton:hover { background-color: #F0F4FF; }
            """)
            
        for btn in self.page_registration.gcs_buttons.values():
            btn.setChecked(False)
            btn.setStyleSheet("""
                QPushButton { 
                    background-color: #FFFFFF; 
                    border: 2px solid #C2D5BB; 
                    border-radius: 12px; 
                    font-size: 28px; 
                    font-weight: bold; 
                    color: #A0B09C; 
                } 
                QPushButton:hover { border-color: #214889; color: #214889; }
            """)
            
        self.page_registration.validate_form()
        self.stacked_widget.setCurrentIndex(1)

    def closeEvent(self, event):
        """Pastikan seluruh background thread ditutup dengan aman saat aplikasi keluar."""
        if hasattr(self.page_live_data, 'worker') and self.page_live_data.worker is not None:
            self.page_live_data.worker.stop()
        if hasattr(self.page_loading, 'worker') and self.page_loading.worker is not None and self.page_loading.worker.isRunning():
            self.page_loading.worker.quit()
            self.page_loading.worker.wait()
        event.accept()

# =====================================================================
# EXECUTION ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TriaGoApplication()
    window.show()
    sys.exit(app.exec())