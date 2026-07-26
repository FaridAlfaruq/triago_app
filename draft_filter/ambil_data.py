import sys
import os
import csv
import time
from collections import deque
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QLineEdit, QMessageBox, QProgressBar)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
import pyqtgraph as pg

# Import custom modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from akusisi_data.get_stm32 import stream_stm32_data
from processing_data.preprocessing_LiveData import LiveSignalFilter

# === OPTIMASI GLOBAL PYQTGRAPH ===
pg.setConfigOptions(antialias=False)  # Matikan antialiasing untuk rendering super cepat
pg.setConfigOption('background', 'k')  # Background hitam standar


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
        self.resize(850, 750)

        # Konfigurasi Parameter Waktu & Sampel
        self.SAMPLE_RATE_HZ = 400
        self.WARMUP_DURATION_SEC = 2.0
        self.RECORD_DURATION_SEC = 60.0
        
        self.total_target_samples = int((self.WARMUP_DURATION_SEC + self.RECORD_DURATION_SEC) * self.SAMPLE_RATE_HZ)

        # State Control Variables
        self.is_recording = False
        self.recorded_data = []
        self.max_plot_points = 1300
        self.live_filter = LiveSignalFilter()

        # === OPTIMASI 1: Ring Buffer O(1) menggunakan deque ===
        self.time_buffer = deque(maxlen=self.max_plot_points)
        self.ecg_buffer = deque(maxlen=self.max_plot_points)
        self.ppg_red_buffer = deque(maxlen=self.max_plot_points)

        # Variables penampung state UI sementara (di-update 400Hz, di-render 30Hz)
        self.latest_temp_obj = 0.0
        self.latest_temp_amb = 0.0
        self.current_progress_val = 0
        self.current_status_text = "Ready for recording..."
        self.current_status_style = "font-size: 14px; font-weight: bold; color: #0055ff;"
        self.is_warmup_phase = False

        self.init_ui()
        self.start_worker_thread()

        # === OPTIMASI 2: QTimer untuk Refresh Rate GUI (30 FPS / ~33 ms) ===
        self.render_timer = QTimer()
        self.render_timer.setInterval(33)  # ~30 FPS
        self.render_timer.timeout.connect(self.update_ui_render)
        self.render_timer.start()

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
        
        # Input SpO2 Ground Truth
        lbl_gt_spo2 = QLabel("SpO2 GT (%):")
        lbl_gt_spo2.setStyleSheet("font-size: 13px; font-weight: bold; margin-left: 20px;")
        self.input_gt_spo2 = QLineEdit()
        self.input_gt_spo2.setPlaceholderText("e.g. 98")
        self.input_gt_spo2.setFixedWidth(50)
        self.input_gt_spo2.setEnabled(False) 

        # Input Heart Rate Ground Truth
        lbl_gt_hr = QLabel("HR GT (bpm):")
        lbl_gt_hr.setStyleSheet("font-size: 13px; font-weight: bold; margin-left: 15px;")
        self.input_gt_hr = QLineEdit()
        self.input_gt_hr.setPlaceholderText("e.g. 75")
        self.input_gt_hr.setFixedWidth(50)
        self.input_gt_hr.setEnabled(False) 
        
        meta_layout.addWidget(self.lbl_temp_obj)
        meta_layout.addWidget(self.lbl_temp_amb)
        meta_layout.addWidget(lbl_gt_spo2)
        meta_layout.addWidget(self.input_gt_spo2)
        meta_layout.addWidget(lbl_gt_hr)
        meta_layout.addWidget(self.input_gt_hr)
        meta_layout.addStretch() 
        main_layout.addLayout(meta_layout)

        # 3. Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        # 4. Graphics Layout Widget
        self.win = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.win)

        self.p1 = self.win.addPlot(title="ECG Signal")
        self.p1.showGrid(x=True, y=True)
        self.p1.setClipToView(True)  # Optimasi render hanya elemen terlihat
        self.ecg_curve = self.p1.plot(pen=pg.mkPen('y', width=1.2))

        self.win.nextRow()

        self.p2 = self.win.addPlot(title="PPG Signal")
        self.p2.showGrid(x=True, y=True)
        self.p2.setClipToView(True)
        self.ppg_curve = self.p2.plot(pen=pg.mkPen('r', width=1.2))
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
        self.worker.data_received.connect(self.handle_new_packet)
        self.worker.start()

    def handle_new_packet(self, packet):
        """Callback ini berjalan cepat 400 Hz HANYA untuk update buffer data & logika perekaman."""
        ecg_val = packet["ecg"]
        ppg_red_val = packet["ppg"]["ir"]  # Menggunakan channel IR untuk grafik PPG
        
        self.latest_temp_obj = packet["temperature"]["object"]
        self.latest_temp_amb = packet["temperature"]["ambient"]

        # Filtering Real-Time
        clean_ecg = self.live_filter.filter_ecg(ecg_val)
        clean_ppg = self.live_filter.filter_ppg(ppg_red_val)

        if self.is_recording:
            self.recorded_data.append(packet)
            
            warmup_samples = int(self.WARMUP_DURATION_SEC * self.SAMPLE_RATE_HZ)
            current_samples_count = len(self.recorded_data)

            if current_samples_count <= warmup_samples:
                self.is_warmup_phase = True
                self.current_progress_val = 0
                self.current_status_text = "STABILIZING SENSOR... Please wait."
                self.current_status_style = "font-size: 14px; font-weight: bold; color: #d97706;"
                return

            self.is_warmup_phase = False
            self.current_status_text = "RECORDING ONGOING..."
            self.current_status_style = "font-size: 14px; font-weight: bold; color: #cb2431;"

            recorded_duration = (current_samples_count - warmup_samples) / self.SAMPLE_RATE_HZ
            self.current_progress_val = int((recorded_duration / self.RECORD_DURATION_SEC) * 100)

            # Fast push ke Ring Buffer deque O(1)
            self.time_buffer.append(recorded_duration)
            self.ecg_buffer.append(clean_ecg)
            self.ppg_red_buffer.append(clean_ppg)

            if current_samples_count >= self.total_target_samples:
                self.stop_and_save_data()

    def update_ui_render(self):
        """Dijalankan 30 FPS oleh QTimer untuk menggambar ulang grafik dan memperbarui teks GUI."""
        # 1. Render Suhu
        self.lbl_temp_obj.setText(f"Body Temp: {self.latest_temp_obj:.2f} °C")
        self.lbl_temp_amb.setText(f"Ambient Temp: {self.latest_temp_amb:.2f} °C")
        if self.latest_temp_obj > 37.5:
            self.lbl_temp_obj.setStyleSheet("font-size: 13px; font-weight: bold; color: #cb2431;")
        else:
            self.lbl_temp_obj.setStyleSheet("font-size: 13px; font-weight: bold; color: #2ea44f;")

        # 2. Render Status & Progress
        if self.is_recording:
            self.progress_bar.setValue(self.current_progress_val)
            self.status_label.setText(self.current_status_text)
            self.status_label.setStyleSheet(self.current_status_style)
            
            if not self.is_warmup_phase and self.lbl_graph_hint.isVisible():
                self.lbl_graph_hint.hide()

            # 3. Render Grafik (Konversi deque ke list untuk pyqtgraph)
            if self.time_buffer:
                t_list = list(self.time_buffer)
                self.ecg_curve.setData(t_list, list(self.ecg_buffer))
                self.ppg_curve.setData(t_list, list(self.ppg_red_buffer))

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recorded_data.clear() 
            self.time_buffer.clear()
            self.ecg_buffer.clear()
            self.ppg_red_buffer.clear()
            self.btn_start.setEnabled(False) 
            
            # Reset dan kunci input GT SpO2 & HR
            self.input_gt_spo2.clear()
            self.input_gt_spo2.setEnabled(False)
            self.input_gt_hr.clear()
            self.input_gt_hr.setEnabled(False)
            
            self.lbl_graph_hint.show()
            self.progress_bar.setValue(0)

    def reset_recording(self):
        self.is_recording = False
        self.recorded_data.clear()
        self.time_buffer.clear()
        self.ecg_buffer.clear()
        self.ppg_red_buffer.clear()
        self.ecg_curve.clear()
        self.ppg_curve.clear()
        self.btn_start.setEnabled(True)
        
        # Reset dan kunci input GT SpO2 & HR
        self.input_gt_spo2.clear()
        self.input_gt_spo2.setEnabled(False)
        self.input_gt_hr.clear()
        self.input_gt_hr.setEnabled(False)
        
        self.lbl_graph_hint.show()
        self.progress_bar.setValue(0)
        
        try:
            self.input_gt_spo2.returnPressed.disconnect(self.save_to_csv)
            self.input_gt_hr.returnPressed.disconnect(self.save_to_csv)
        except TypeError:
            pass
            
        self.status_label.setText("Recording reset. Ready to start again.")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #0055ff;")

    def stop_and_save_data(self):
        """ Menangani akhir durasi rekaman dan mengaktifkan kolom input Ground Truth """
        self.is_recording = False
        self.status_label.setText("Recording Finished! Please input SpO2 & HR Ground Truth to Save.")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2ea44f;")
        
        # Aktifkan kolom input
        self.input_gt_spo2.setEnabled(True)
        self.input_gt_hr.setEnabled(True)
        self.input_gt_spo2.setFocus()
        
        # Menekan Enter di salah satu input akan memicu proses penyimpanan data
        self.input_gt_spo2.returnPressed.connect(self.save_to_csv)
        self.input_gt_hr.returnPressed.connect(self.save_to_csv)

    def generate_next_filename(self):
        index = 1
        while True:
            filename = f"Data{index}.csv"
            if not os.path.exists(filename):
                return filename
            index += 1

    def save_to_csv(self):
        spo2_val = self.input_gt_spo2.text().strip()
        hr_val = self.input_gt_hr.text().strip()
        
        # 1. Validasi Input SpO2
        if not spo2_val:
            QMessageBox.warning(self, "Input Required", "Mohon masukkan nilai SpO2 pembanding terlebih dahulu!")
            self.input_gt_spo2.setFocus()
            return
        try:
            float(spo2_val)
        except ValueError:
            QMessageBox.critical(self, "Invalid Input", "Nilai SpO2 pembanding harus berupa angka!")
            self.input_gt_spo2.setFocus()
            return

        # 2. Validasi Input Heart Rate (HR)
        if not hr_val:
            QMessageBox.warning(self, "Input Required", "Mohon masukkan nilai Heart Rate pembanding terlebih dahulu!")
            self.input_gt_hr.setFocus()
            return
        try:
            float(hr_val)
        except ValueError:
            QMessageBox.critical(self, "Invalid Input", "Nilai Heart Rate pembanding harus berupa angka!")
            self.input_gt_hr.setFocus()
            return

        filename = self.generate_next_filename()
        sampling_interval = 1.0 / self.SAMPLE_RATE_HZ 

        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Menambahkan header kolom HR_Ground_Truth
                writer.writerow(["Time (s)", "PPG_Red", "PPG_IR", "PPG_Green", "ECG", "Temp_Ambient", "Temp_Object", "SpO2_Ground_Truth", "HR_Ground_Truth"])
                
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
                        spo2_val,
                        hr_val
                    ])
            
            QMessageBox.information(self, "Success", f"Data berhasil disimpan ke {filename}")
            self.status_label.setText(f"Finished & Saved to {filename}")
            
            self.btn_start.setEnabled(True)
            self.input_gt_spo2.setEnabled(False)
            self.input_gt_hr.setEnabled(False)
            
            self.input_gt_spo2.returnPressed.disconnect(self.save_to_csv)
            self.input_gt_hr.returnPressed.disconnect(self.save_to_csv)

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Gagal menyimpan file CSV: {e}")
            
    def closeEvent(self, event):
        self.is_recording = False
        if hasattr(self, 'render_timer'):
            self.render_timer.stop()
        if hasattr(self, 'worker'):
            self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())