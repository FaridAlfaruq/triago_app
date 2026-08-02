import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PyQt6.QtCore import Qt, QTimer

# System Path Integration
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import seluruh halaman TriaGO
from GUI.home_page import HomePage
from GUI.loading_page import LoadingPage
from GUI.output_page import OutputPage
from GUI.plot_page import PlotPage
from GUI.regist_page import RegistrationPage


# =====================================================================
# INTI CORE APLIKASI: TriaGoApplication
# =====================================================================
class TriaGoApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TriaGO - Automated Medical Triage Kiosk")
        self.resize(1280, 800)
        
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
        self.page_live_data.sensor_error.connect(self.handle_sensor_error)
        
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

    def handle_sensor_error(self, message):
        """Tampilkan kegagalan koneksi tanpa mengunci pengguna di loading."""
        print(f"[ERROR SENSOR] {message}")
        self.page_loading.progress_bar.setValue(0)
        self.page_loading.lbl_status.setText(
            "Sensor tidak terhubung. Kembali ke halaman registrasi..."
        )
        QTimer.singleShot(2500, self.go_to_registration)

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

            skin_temp = self.current_patient_info.get("temp_skin", 36.5)
            amb_temp = self.current_patient_info.get("temp_ambient", 31.52)
            
            raw_temp_obj = np.array([
                p.get("temp_skin", p.get("temp_obj", p.get("temperature", {}).get("object", skin_temp)))
                for p in raw_data_list
            ]) if raw_data_list else np.array([])
            
            raw_temp_amb = np.array([
                p.get("temp_ambient", p.get("temp_amb", p.get("temperature", {}).get("ambient", amb_temp)))
                for p in raw_data_list
            ]) if raw_data_list else np.array([])

            self.current_patient_info["raw_temp_obj"] = raw_temp_obj
            self.current_patient_info["raw_temp_amb"] = raw_temp_amb
            print(f"[LOG MAIN_GUI] Data Suhu Pasien Siap -> Kulit: {skin_temp}°C | Lingkungan: {amb_temp}°C (Populated {len(raw_temp_obj)} samples)")

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
        """Fase 3: Evaluasi SQA, Menyimpan 2 File (CSV & JSON), & Buka Halaman Output."""
        print("[LOG SUCCESS] Memproses fase output akhir...")
        
        # PERIKSA HASIL SQA 10S WINDOW (STRIDE 2S)
        if not calculation_results.get("sqa_passed", True):
            from PyQt6.QtWidgets import QMessageBox
            err_msg = calculation_results.get(
                "sqa_error",
                "Tidak ada segmen sinyal 10s yang lolos SQA (Artefak/Noise tinggi).\nSilakan pastikan sensor terpasang baik dan lakukan pengambilan data ulang."
            )
            print(f"[WARN MAIN_GUI SQA REJECTED] {err_msg}")
            QMessageBox.warning(self, "Pengambilan Data Ulang (SQA Gagal)", err_msg)
            self.reset_to_gatekeeper()
            return

        # 1. SIMPAN 2 FILE KONSOLIDASI (CSV & JSON)
        self.save_consolidated_csv(calculation_results)
        
        # 2. Update Halaman Output (Grafik 5s, Parameter, SHAP, & JSON IoT)
        if hasattr(self.page_output, "update_results"):
            self.page_output.update_results(calculation_results)
            
        triage_status = calculation_results.get("triage_status", "NON-DARURAT")
        self.page_output.update_triage_header(triage_status)
        
        # 3. Pindah ke Halaman Output (Index 4)
        self.stacked_widget.setCurrentIndex(4)

    def save_consolidated_csv(self, results):
        """Menyimpan 2 file per pengukuran: CSV (10 kolom sinyal) dan JSON (metadata & fitur)."""
        import json
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        bed_id = results.get("bed", "00")
        
        folder_path = os.path.join(project_root, "data_pengukuran")
        os.makedirs(folder_path, exist_ok=True)
        
        base_name = f"{timestamp_str}_Bed{bed_id}"
        csv_filename = os.path.join(folder_path, f"{base_name}.csv")
        json_filename = os.path.join(folder_path, f"{base_name}.json")
        
        try:
            # 1. Simpan CSV (10 Kolom)
            raw_time = results.get("raw_time", [])
            raw_red = results.get("raw_red", [])
            raw_ir = results.get("raw_ir", [])
            raw_ecg = results.get("raw_ecg", [])
            
            time_125 = results.get("time_125", [])
            red_clean = results.get("red_clean", [])
            ir_clean = results.get("ir_clean", [])
            ecg_clean = results.get("ecg_smooth", [])
            
            temp_obj = results.get("temp_skin", results.get("temperature", 36.5))
            temp_amb = results.get("temp_ambient", 31.52)
            
            n_raw = len(raw_time) if len(raw_time) > 0 else (len(time_125) if len(time_125) > 0 else 1)
            
            raw_temp_obj = results.get("raw_temp_obj")
            if raw_temp_obj is None or len(raw_temp_obj) == 0:
                raw_temp_obj = np.full(n_raw, temp_obj)
                
            raw_temp_amb = results.get("raw_temp_amb")
            if raw_temp_amb is None or len(raw_temp_amb) == 0:
                raw_temp_amb = np.full(n_raw, temp_amb)
            
            df = pd.DataFrame({
                "Time (s)": pd.Series(raw_time),
                "PPG_Red": pd.Series(raw_red),
                "PPG_IR": pd.Series(raw_ir),
                "ECG_Raw": pd.Series(raw_ecg),
                "Resample Time (s)": pd.Series(time_125),
                "PPG_Red_Clean": pd.Series(red_clean),
                "PPG_IR_Clean": pd.Series(ir_clean),
                "ECG_Clean": pd.Series(ecg_clean),
                "Suhu Obj": pd.Series(raw_temp_obj),
                "Suhu Amb": pd.Series(raw_temp_amb)
            })
            df.to_csv(csv_filename, index=False)
            print(f"[LOG SUCCESS] File CSV Sinyal (10 Kolom) Berhasil Disimpan: {csv_filename}")
            
            # 2. Simpan JSON (Hasil Ekstraksi Fitur & Metadata)
            metadata_json = {
                "Timestamp": results.get("timestamp", timestamp_str),
                "Bed": str(results.get("bed", bed_id)),
                "GCS Score": int(results.get("gcs", 15)),
                "HR": float(results.get("hr", 0.0)),
                "RR": float(results.get("rr", 0.0)),
                "SpO2": float(results.get("spo2", 0.0)),
                "PI Red": float(results.get("pi_red", 0.0)),
                "PI IR": float(results.get("pi_ir", 0.0)),
                "SBP": float(results.get("systolic", 120)),
                "DBP": float(results.get("diastolic", 80)),
                "Suhu Core": float(results.get("temperature", 36.5)),
                "Suhu Skin": float(results.get("temp_skin", temp_obj)),
                "Suhu Amb": float(results.get("temp_ambient", temp_amb)),
                "Triage Status": str(results.get("triage_status", "DARURAT")),
                "Triage Confidence": float(results.get("xgboost_score", 0.0))
            }
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(metadata_json, f, indent=4)
            print(f"[LOG SUCCESS] File JSON Metadata Berhasil Disimpan: {json_filename}")
            
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan file pengukuran CSV/JSON: {e}")

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

    def keyPressEvent(self, event):
        """Esc keluar dari fullscreen; F11 mengaktifkan/menonaktifkan fullscreen."""
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.showNormal()
            self.resize(1280, 800)
            return
        if event.key() == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showNormal()
                self.resize(1280, 800)
            else:
                self.showFullScreen()
            return
        super().keyPressEvent(event)

# =====================================================================
# EXECUTION ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TriaGoApplication()
    window.showFullScreen()
    sys.exit(app.exec())
