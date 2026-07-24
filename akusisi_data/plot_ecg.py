import sys
import os
import csv
from collections import deque
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg

# Import generator dari get_stm32.py
from get_stm32 import stream_stm32_data, DEFAULT_PORT, DEFAULT_BAUDRATE

# Parameter Sampling & Tampilan
FS = 400                    # Sampling Frequency (400 Hz presisi dari STM32)
PLOT_DURATION_SEC = 3.0     # Durasi Tampilan Grafik (5 Detik)
MAX_PLOT_POINTS = int(PLOT_DURATION_SEC * FS)  # 2,000 titik sampel
RECORD_DURATION_SEC = 60.0  # Durasi Perekaman (1 Menit)
TOTAL_TARGET_SAMPLES = int(RECORD_DURATION_SEC * FS)  # 24,000 sampel

# === OPTIMASI PYQTGRAPH ===
pg.setConfigOptions(antialias=False)  # Matikan antialiasing agar render cepat
pg.setConfigOption('background', 'k')  # Background hitam standar


class BiosignalReaderWorker(QtCore.QThread):
    """Worker Thread untuk membaca data dari get_stm32.py secara asinkron."""
    data_received = QtCore.pyqtSignal(int, int, int, int, float, float)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUDRATE):
        super().__init__()
        self.port = port
        self.baud = baud
        self.running = True

    def run(self):
        try:
            for packet in stream_stm32_data(port=self.port, baudrate=self.baud):
                if not self.running:
                    break
                
                if packet["status"] == "OK":
                    ppg = packet["ppg"]
                    ecg = packet["ecg"]
                    temp = packet["temperature"]
                    
                    self.data_received.emit(
                        ppg["red"],
                        ppg["ir"],
                        ppg["green"],
                        ecg,
                        temp["ambient"],
                        temp["object"]
                    )
        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop_serial(self):
        self.running = False


class BiosignalPlotterWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PLOT DATA) - ECG, PPG & Suhu")
        self.resize(1000, 750)

        # State Control Variables
        self.is_recording = False
        self.recorded_samples = []  
        
        # === OPTIMASI 1: Ring Buffer O(1) menggunakan deque ===
        self.time_buffer = deque(maxlen=MAX_PLOT_POINTS)
        self.red_buffer = deque(maxlen=MAX_PLOT_POINTS)
        self.ecg_buffer = deque(maxlen=MAX_PLOT_POINTS)
        self.total_received_samples = 0

        # State sementara untuk update label (dikirim dari worker)
        self.latest_red = 0
        self.latest_ecg = 0
        self.latest_t_amb = 0.0
        self.latest_t_obj = 0.0

        self.init_ui()
        self.start_worker_thread()

        # === OPTIMASI 2: QTimer untuk Refresh Rate GUI (30 FPS / ~33 ms) ===
        self.render_timer = QtCore.QTimer()
        self.render_timer.setInterval(33)  # Refresh UI setiap 33ms, bukan 400Hz
        self.render_timer.timeout.connect(self.update_gui_render)
        self.render_timer.start()

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # 1. Header Status Label
        self.status_label = QtWidgets.QLabel(f"Menghubungkan ke {DEFAULT_PORT}...")
        main_layout.addWidget(self.status_label)

        # 2. Temperature Display (Label Biasa)
        temp_layout = QtWidgets.QHBoxLayout()
        self.lbl_obj_val = QtWidgets.QLabel("Suhu Objek: --.-- °C")
        self.lbl_amb_val = QtWidgets.QLabel("Suhu Lingkungan: --.-- °C")

        temp_layout.addWidget(self.lbl_obj_val)
        temp_layout.addWidget(self.lbl_amb_val)
        temp_layout.addStretch()
        main_layout.addLayout(temp_layout)

        # 3. Ground Truth HR Layout
        input_layout = QtWidgets.QHBoxLayout()
        lbl_gt_hr = QtWidgets.QLabel("HR Ground Truth (bpm) [Opsional]:")
        
        self.input_gt_hr = QtWidgets.QLineEdit()
        self.input_gt_hr.setPlaceholderText("e.g. 75")
        self.input_gt_hr.setFixedWidth(180)
        
        input_layout.addWidget(lbl_gt_hr)
        input_layout.addWidget(self.input_gt_hr)
        input_layout.addStretch()
        main_layout.addLayout(input_layout)

        # 4. Progress Bar Perekaman
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        # 5. Graphics Layout Widget (PyQtGraph)
        self.win = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.win)

        # --- PLOT 1: ECG RAW ---
        self.p_ecg = self.win.addPlot(row=0, col=0, title="ECG Signal (RAW 12-bit - 5 Detik)")
        self.p_ecg.showGrid(x=True, y=True)
        self.p_ecg.setLabel('left', 'ECG', units='LSB')
        self.p_ecg.setLabel('bottom', 'Time', units='s')
        self.p_ecg.setClipToView(True)  # Optimasi render hanya area terlihat
        self.ecg_curve = self.p_ecg.plot(pen=pg.mkPen('y', width=1.2))

        # --- PLOT 2: PPG RED RAW ---
        self.p_ppg = self.win.addPlot(row=1, col=0, title="PPG RED Signal (RAW 18-bit - 5 Detik)")
        self.p_ppg.showGrid(x=True, y=True)
        self.p_ppg.setLabel('left', 'PPG RED', units='Counts')
        self.p_ppg.setLabel('bottom', 'Time', units='s')
        self.p_ppg.setClipToView(True)
        
        # Link sumbu X antara Plot ECG dan PPG
        self.p_ppg.setXLink(self.p_ecg)
        self.red_curve = self.p_ppg.plot(pen=pg.mkPen('r', width=1.2))

        # 6. Tombol Control ('Rekam' & 'Reset')
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_record = QtWidgets.QPushButton("Rekam (1 Menit)")
        self.btn_record.clicked.connect(self.start_recording)
        
        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_recording)

        button_layout.addWidget(self.btn_record)
        button_layout.addWidget(self.btn_reset)
        main_layout.addLayout(button_layout)

    def start_worker_thread(self):
        self.worker = BiosignalReaderWorker(port=DEFAULT_PORT, baud=DEFAULT_BAUDRATE)
        self.worker.data_received.connect(self.handle_new_sample)
        self.worker.error_occurred.connect(self.handle_serial_error)
        self.worker.start()

    @QtCore.pyqtSlot(int, int, int, int, float, float)
    def handle_new_sample(self, red_val, ir_val, green_val, ecg_val, t_amb, t_obj):
        """Dijalankan cepat 400 Hz HANYA untuk push data ke buffer (Ringan)."""
        self.total_received_samples += 1
        current_time_s = self.total_received_samples / FS

        # Simpan nilai terbaru untuk di-render nanti oleh QTimer
        self.latest_red = red_val
        self.latest_ecg = ecg_val
        self.latest_t_amb = t_amb
        self.latest_t_obj = t_obj

        # Fast append O(1)
        self.time_buffer.append(current_time_s)
        self.red_buffer.append(red_val)
        self.ecg_buffer.append(ecg_val)

        # Perekaman 1 Menit (24,000 Sampel)
        if self.is_recording:
            self.recorded_samples.append((current_time_s, red_val, ir_val, green_val, ecg_val, t_amb, t_obj))
            if len(self.recorded_samples) >= TOTAL_TARGET_SAMPLES:
                self.stop_and_save_data()

    def update_gui_render(self):
        """Dijalankan stabil 30 FPS oleh QTimer untuk menggambar ulang grafik & label."""
        if not self.time_buffer:
            return

        # Convert deque ke list secara cepat untuk pyqtgraph
        t_data = list(self.time_buffer)
        
        # 1. Update Grafik
        self.ecg_curve.setData(t_data, list(self.ecg_buffer))
        self.red_curve.setData(t_data, list(self.red_buffer))

        # 2. Update Label Suhu
        self.lbl_obj_val.setText(f"Suhu Objek: {self.latest_t_obj:.2f} °C")
        self.lbl_amb_val.setText(f"Suhu Lingkungan: {self.latest_t_amb:.2f} °C")

        # 3. Update Status & Progress Bar
        if self.is_recording:
            count = len(self.recorded_samples)
            progress = int((count / TOTAL_TARGET_SAMPLES) * 100)
            self.progress_bar.setValue(progress)
            
            elapsed_sec = count / FS
            self.status_label.setText(
                f"RECORDING ONGOING... [{elapsed_sec:.1f}s / {RECORD_DURATION_SEC:.0f}s] | "
                f"RED: {self.latest_red} | ECG: {self.latest_ecg} | OBJ: {self.latest_t_obj:.2f}°C"
            )
        else:
            self.status_label.setText(
                f"Streaming Real-Time (400 Hz) | RED: {self.latest_red} | ECG: {self.latest_ecg}"
            )

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recorded_samples.clear()
            self.btn_record.setEnabled(False)
            self.progress_bar.setValue(0)

    def reset_recording(self):
        self.is_recording = False
        self.recorded_samples.clear()
        self.progress_bar.setValue(0)
        self.btn_record.setEnabled(True)
        self.status_label.setText("Perekaman di-reset. Siap untuk rekam ulang.")

    def generate_next_filename(self):
        index = 1
        while True:
            filename = f"Data_{index}.CSV"
            if not os.path.exists(filename):
                return filename
            index += 1

    def stop_and_save_data(self):
        self.is_recording = False
        self.btn_record.setEnabled(True)
        
        hr_gt_val = self.input_gt_hr.text().strip()
        if not hr_gt_val:
            hr_gt_val = "N/A"

        filename = self.generate_next_filename()
        sampling_interval = 1.0 / FS

        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time (s)", "PPG_RED_RAW", "PPG_IR", "PPG_GREEN", "ECG", "T_AMBIENT", "T_OBJECT", "HR_GT"])
                
                for idx, (t_val, red_val, ir_val, green_val, ecg_val, t_amb, t_obj) in enumerate(self.recorded_samples):
                    rel_time = idx * sampling_interval
                    writer.writerow([f"{rel_time:.4f}", red_val, ir_val, green_val, ecg_val, f"{t_amb:.2f}", f"{t_obj:.2f}", hr_gt_val])

            QtWidgets.QMessageBox.information(
                self, "Success", f"Data RAW 1 menit (24.000 sampel) berhasil disimpan ke:\n{filename}"
            )
            self.status_label.setText(f"Perekaman Selesai & Disimpan ke {filename}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Gagal menyimpan file CSV: {e}")

    def handle_serial_error(self, err_msg):
        self.status_label.setText(f"ERROR SERIAL: {err_msg}")

    def closeEvent(self, event):
        self.is_recording = False
        if hasattr(self, 'render_timer'):
            self.render_timer.stop()
        if hasattr(self, 'worker'):
            self.worker.stop_serial()
            self.worker.wait(1000)
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = BiosignalPlotterWindow()
    window.show()
    sys.exit(app.exec())