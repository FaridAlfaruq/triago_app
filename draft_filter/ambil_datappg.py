import sys
import os
import csv
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QLineEdit, QMessageBox, QProgressBar)
from PyQt6.QtCore import QThread, pyqtSignal
import pyqtgraph as pg

# Import custom modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from akuisisi_data.get_stm32 import stream_stm32_data
from processing_data.preprocessing_LiveData import LiveSignalFilter

class STM32Worker(QThread):
    data_received = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        for packet in stream_stm32_data():
            if not self.running:
                break
            if packet["status"] == "OK":
                self.data_received.emit(packet)

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STM32 Bio-Signal Data Acquisition for Calibration")
        self.resize(800, 750)

        # Konfigurasi Parameter Waktu & Sampel (Sesuai Logika Fungsi Baru Anda)
        self.SAMPLE_RATE_HZ = 400
        self.WARMUP_DURATION_SEC = 2.0
        self.RECORD_DURATION_SEC = 15.0
        
        # Total target sampel = (2 detik warmup + 60 detik rekaman) * 400 Hz = 24800 sampel
        self.total_target_samples = int((self.WARMUP_DURATION_SEC + self.RECORD_DURATION_SEC) * self.SAMPLE_RATE_HZ)

        # State Control Variables
        self.is_recording = False
        self.recorded_data = []
        self.max_plot_points = 1300
        self.live_filter = LiveSignalFilter()

        # Buffers untuk Grafik
        self.time_buffer = []
        self.ecg_buffer = []
        self.ppg_red_buffer = []

        self.init_ui()
        self.start_worker_thread()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. Status Label & Hint Label
        self.status_label = QLabel("Ready for recording...")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #0055ff;")
        self.lbl_graph_hint = QLabel("Waiting for sensor stabilization...")
        self.lbl_graph_hint.setStyleSheet("font-size: 12px; color: #777777; font-style: italic;")
        
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.lbl_graph_hint)

        # 2. Temperature & Ground Truth Input Layout
        meta_layout = QHBoxLayout()
        self.lbl_temp_obj = QLabel("Body Temp: -- °C")
        self.lbl_temp_obj.setStyleSheet("font-size: 13px; font-weight: bold; color: #555555;")
        self.lbl_temp_amb = QLabel("Ambient Temp: -- °C")
        self.lbl_temp_amb.setStyleSheet("font-size: 13px; font-weight: bold; color: #555555; margin-left: 15px;")
        
        lbl_gt = QLabel("SpO2 Ground Truth (%):")
        lbl_gt.setStyleSheet("font-size: 13px; font-weight: bold; margin-left: 30px;")
        self.input_gt_spo2 = QLineEdit()
        self.input_gt_spo2.setPlaceholderText("e.g. 98")
        self.input_gt_spo2.setFixedWidth(60)
        self.input_gt_spo2.setEnabled(False) 
        
        meta_layout.addWidget(self.lbl_temp_obj)
        meta_layout.addWidget(self.lbl_temp_amb)
        meta_layout.addWidget(lbl_gt)
        meta_layout.addWidget(self.input_gt_spo2)
        meta_layout.addStretch() 
        main_layout.addLayout(meta_layout)

        # 3. Progress Bar (Baru)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        # 4. Graphics Layout Widget
        self.win = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.win)

        self.p1 = self.win.addPlot(title="ECG Signal")
        self.p1.showGrid(x=True, y=True)
        self.ecg_curve = self.p1.plot(pen=pg.mkPen('y', width=1.5))

        self.win.nextRow()

        self.p2 = self.win.addPlot(title="PPG Signal")
        self.p2.showGrid(x=True, y=True)
        self.ppg_curve = self.p2.plot(pen=pg.mkPen('r', width=1.5))
        self.p2.setXLink(self.p1)

        # 5. Control Buttons
        button_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start (Record 1 Min)")
        self.btn_start.setStyleSheet("background-color: #2ea44f; color: white; font-weight: bold; padding: 10px; border-radius: 4px;")
        self.btn_start.clicked.connect(self.start_recording)
        
        self.btn_reset = QPushButton("Ulangi (Reset)")
        self.btn_reset.setStyleSheet("background-color: #cb2431; color: white; font-weight: bold; padding: 10px; border-radius: 4px;")
        self.btn_reset.clicked.connect(self.reset_recording)
        
        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_reset)
        main_layout.addLayout(button_layout)

    def start_worker_thread(self):
        self.worker = STM32Worker()
        self.worker.data_received.connect(self.handle_new_packet) # Menggunakan fungsi logika baru Anda
        self.worker.start()

    def update_temperature_ui(self, temp_object, temp_ambient):
        """ Mengatur teks dan warna kondisional berdasarkan nilai suhu tubuh pasien """
        self.lbl_temp_obj.setText(f"Body Temp: {temp_object:.2f} °C")
        self.lbl_temp_amb.setText(f"Ambient Temp: {temp_ambient:.2f} °C")
        
        # Contoh visual warna kondisional demam / normal
        if temp_object > 37.5:
            self.lbl_temp_obj.setStyleSheet("font-size: 13px; font-weight: bold; color: #cb2431;") # Merah jika demam
        else:
            self.lbl_temp_obj.setStyleSheet("font-size: 13px; font-weight: bold; color: #2ea44f;") # Hijau jika normal

    def handle_new_packet(self, packet):
        """ Implementasi penuh logika pemisahan Warmup dan Riil Rekaman """
        current_time = packet["timestamp"]
        ecg_val = packet["ecg"]
        ppg_red_val = packet["ppg"]["red"]
        
        # 1. Update Teks & Warna Kondisional Suhu Tubuh Pasien
        temp_object = packet["temperature"]["object"]
        temp_ambient = packet["temperature"]["ambient"]
        self.update_temperature_ui(temp_object, temp_ambient)

        clean_ecg = self.live_filter.filter_ecg(ecg_val)
        clean_ppg = self.live_filter.filter_ppg(ppg_red_val)

        if self.is_recording:
            self.recorded_data.append(packet)
            
            warmup_samples = int(self.WARMUP_DURATION_SEC * self.SAMPLE_RATE_HZ)
            current_samples_count = len(self.recorded_data)

            # FASE A: Masa Stabilisasi / Pemanasan Sensor (0 - 2 Detik Awal)
            if current_samples_count <= warmup_samples:
                self.progress_bar.setValue(0)
                self.status_label.setText("STABILIZING SENSOR... Please wait.")
                self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #d97706;")
                return

            # FASE B: Masa Perekaman Riil (Setelah detik ke-2 / Sinyal Sudah Stabil)
            if self.lbl_graph_hint.isVisible():
                self.lbl_graph_hint.hide() 
                self.status_label.setText("RECORDING ONGOING...")
                self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #cb2431;")

            # Hitung waktu relatif (Detik ke-2 dikonversi menjadi detik ke-0 pada sumbu-X)
            recorded_duration = (current_samples_count - warmup_samples) / self.SAMPLE_RATE_HZ
            
            progress = (recorded_duration / self.RECORD_DURATION_SEC) * 100
            self.progress_bar.setValue(int(progress))

            # Masukkan data hasil filter yang SUDAH STABIL ke dalam buffer visualisasi
            self.time_buffer.append(recorded_duration)
            self.ecg_buffer.append(clean_ecg)
            self.ppg_red_buffer.append(clean_ppg)

            if len(self.time_buffer) > self.max_plot_points:
                self.time_buffer.pop(0)
                self.ecg_buffer.pop(0)
                self.ppg_red_buffer.pop(0)

            self.ecg_curve.setData(self.time_buffer, self.ecg_buffer)
            self.ppg_curve.setData(self.time_buffer, self.ppg_red_buffer)

            if current_samples_count >= self.total_target_samples:
                self.stop_and_save_data()

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recorded_data = [] 
            self.time_buffer.clear()
            self.ecg_buffer.clear()
            self.ppg_red_buffer.clear()
            self.btn_start.setEnabled(False) 
            self.input_gt_spo2.clear()
            self.input_gt_spo2.setEnabled(False)
            self.lbl_graph_hint.show()
            self.progress_bar.setValue(0)

    def reset_recording(self):
        self.is_recording = False
        self.recorded_data = []
        self.time_buffer.clear()
        self.ecg_buffer.clear()
        self.ppg_red_buffer.clear()
        self.ecg_curve.clear()
        self.ppg_curve.clear()
        self.btn_start.setEnabled(True)
        self.input_gt_spo2.clear()
        self.input_gt_spo2.setEnabled(False)
        self.lbl_graph_hint.show()
        self.progress_bar.setValue(0)
        try:
            self.input_gt_spo2.returnPressed.disconnect(self.save_to_csv)
        except TypeError:
            pass
        self.status_label.setText("Recording reset. Ready to start again.")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #0055ff;")

    def stop_and_save_data(self):
        """ Menangani akhir durasi rekaman dan mengaktifkan kolom input SpO2 Ground Truth """
        self.is_recording = False
        self.status_label.setText("Recording Finished! Please input SpO2 Ground Truth to Save.")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2ea44f;")
        
        self.input_gt_spo2.setEnabled(True)
        self.input_gt_spo2.setFocus()
        self.input_gt_spo2.returnPressed.connect(self.save_to_csv)

    def generate_next_filename(self):
        """ Mencari index terakhir di folder untuk format penamaan Data1, Data2, dst. """
        index = 1
        while True:
            filename = f"Data{index}.csv"
            if not os.path.exists(filename):
                return filename
            index += 1

    def save_to_csv(self):
        gt_value = self.input_gt_spo2.text().strip()
        if not gt_value:
            QMessageBox.warning(self, "Input Required", "Mohon masukkan nilai SpO2 pembanding terlebih dahulu!")
            return
        
        try:
            float(gt_value)
        except ValueError:
            QMessageBox.critical(self, "Invalid Input", "Nilai SpO2 pembanding harus berupa angka!")
            return

        filename = self.generate_next_filename() # Menghasilkan nama Data1.csv, Data2.csv, dst.
        sampling_interval = 1.0 / self.SAMPLE_RATE_HZ 

        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time (s)", "PPG_Red", "PPG_IR", "PPG_Green", "ECG", "Temp_Ambient", "Temp_Object", "SpO2_Ground_Truth"])
                
                for index, p in enumerate(self.recorded_data):
                    relative_time_s = index * sampling_interval
                    writer.writerow([
                        f"{relative_time_s:.4f}",
                        p["ppg"]["red"],
                        p["ppg"]["ir"],
                        p["ppg"]["green"],
                        p["ecg"],
                        p["temperature"]["ambient"],
                        p["temperature"]["object"],
                        gt_value
                    ])
            
            QMessageBox.information(self, "Success", f"Data berhasil disimpan ke {filename}")
            self.status_label.setText(f"Finished & Saved to {filename}")
            
            self.btn_start.setEnabled(True)
            self.input_gt_spo2.setEnabled(False)
            self.input_gt_spo2.returnPressed.disconnect(self.save_to_csv)

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Gagal menyimpan file CSV: {e}")
            
    def closeEvent(self, event):
        self.is_recording = False
        if hasattr(self, 'worker'):
            self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())