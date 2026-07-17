import sys
import os
import csv
import time
from datetime import datetime
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel
from PyQt6.QtCore import QThread, pyqtSignal

# import custom modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 3. Sekarang kamu bisa memanggil file di folder lain dengan sangat lancar!
from akuisisi_data.get_stm32 import stream_stm32_data
from processing_data.preprocessing_LiveData import LiveSignalFilter
from akuisisi_data.get_stm32 import stream_stm32_data
from processing_data.preprocessing_LiveData import LiveSignalFilter

class STM32Worker(QThread):
    """
    Worker thread dedicated to reading serial data continuously 
    without blocking the main GUI thread.
    """
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
        self.setWindowTitle("STM32 Bio-Signal Monitor (Split Plots)")
        self.resize(800, 700)

        # Application Reference Time for Relative X-Axis Data
        self.start_app_time = time.time()

        # State Control Variables
        self.is_recording = False
        self.record_start_time = None
        self.recorded_data = []
        self.max_plot_points = 1300
        self.live_filter = LiveSignalFilter()

        # Separate buffers for both plots
        self.time_buffer = []
        self.ecg_buffer = []
        self.ppg_red_buffer = []

        self.init_ui()
        self.start_worker_thread()

    def init_ui(self):
        # 1. Main Widget & Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 2. Status Label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #0055ff;")
        main_layout.addWidget(self.status_label)

        # 2. Temperature only for testing gui
        temp_layout = QHBoxLayout()
        self.lbl_temp_obj = QLabel("Body Temp: -- °C")
        self.lbl_temp_obj.setStyleSheet("font-size: 14px; font-weight: bold; color: #ff5500; margin-left: 10px;")
        self.lbl_temp_amb = QLabel("Ambient Temp: -- °C")
        self.lbl_temp_amb.setStyleSheet("font-size: 14px; font-weight: bold; color: #555555; margin-left: 20px;")
        
        temp_layout.addWidget(self.lbl_temp_obj)
        temp_layout.addWidget(self.lbl_temp_amb)
        temp_layout.addStretch() # Mendorong teks ke kiri agar rapi
        main_layout.addLayout(temp_layout)

        # 3. Graphics Layout Widget (pyqtgraph multi-plot container)
        self.win = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.win)

        # --- PLOT 1: ECG (Top Graph) ---
        self.p1 = self.win.addPlot(title="ECG Signal")
        self.p1.setLabel('left', 'Amplitude')
        self.p1.setLabel('bottom', 'Time', units='s')
        self.p1.showGrid(x=True, y=True)
        self.ecg_curve = self.p1.plot(pen=pg.mkPen('y', width=1.5)) # Yellow

        # Move to the next row inside the graphics layout
        self.win.nextRow()

        # --- PLOT 2: PPG Red (Bottom Graph) ---
        self.p2 = self.win.addPlot(title="PPG Signal")
        self.p2.setLabel('left', 'Light Intensity')
        self.p2.setLabel('bottom', 'Time', units='s')
        self.p2.showGrid(x=True, y=True)
        self.ppg_curve = self.p2.plot(pen=pg.mkPen('r', width=1.5)) # Red

        # Link X-axes together so panning/zooming one shifts both
        self.p2.setXLink(self.p1)

        # 4. Control Buttons Layout
        button_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("Start (Record 1 Min)")
        self.btn_start.setStyleSheet("background-color: #2ea44f; color: white; font-weight: bold; padding: 10px; border-radius: 4px;")
        self.btn_start.clicked.connect(self.start_recording)
        
        self.btn_stop = QPushButton("Stop All")
        self.btn_stop.setStyleSheet("background-color: #cb2431; color: white; font-weight: bold; padding: 10px; border-radius: 4px;")
        self.btn_stop.clicked.connect(self.stop_all)
        
        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_stop)
        main_layout.addLayout(button_layout)

    def start_worker_thread(self):
        self.worker = STM32Worker()
        self.worker.data_received.connect(self.handle_new_data)
        self.worker.start()

    def handle_new_data(self, packet):
        current_time = packet["timestamp"]
        ecg_val = packet["ecg"]
        ppg_red_val = packet["ppg"]["red"]
        
        # Live Temperature Information Update
        temp_ambient = packet["temperature"]["ambient"]
        temp_object = packet["temperature"]["object"]
        self.lbl_temp_obj.setText(f"Body Temp: {temp_object:.2f} °C")
        self.lbl_temp_amb.setText(f"Ambient Temp: {temp_ambient:.2f} °C")

        # Calculate relative time in seconds since the application started
        relative_time = current_time - self.start_app_time

        # --- PART 1: LIVE FILTERING PROCESSED PER SAMPLE ---
        # Bersihkan sampel mentah secara instan menggunakan state filter terbaru
        clean_ecg = self.live_filter.filter_ecg(ecg_val)
        clean_ppg = self.live_filter.filter_ppg(ppg_red_val)

        # Masukkan data yang sudah bersih ke dalam buffer visualisasi grafik
        self.time_buffer.append(relative_time)
        self.ecg_buffer.append(clean_ecg)
        self.ppg_red_buffer.append(clean_ppg)

        # Restrict the buffer size to avoid performance lag (Keep last 800 samples)
        if len(self.time_buffer) > self.max_plot_points:
            self.time_buffer.pop(0)
            self.ecg_buffer.pop(0)
            self.ppg_red_buffer.pop(0)

        # Update curves safely without any tail artifacts or jittering
        self.ecg_curve.setData(self.time_buffer, self.ecg_buffer)
        self.ppg_curve.setData(self.time_buffer, self.ppg_red_buffer)

        # --- PART 2: 1-MINUTE RECORDING LOGIC (Stays Raw) ---
        if self.is_recording:
            self.recorded_data.append(packet)
            
            # Hitung persentase keterisian data berdasarkan jumlah sampel nyata
            progress = (len(self.recorded_data) / self.target_samples) * 100
            self.status_label.setText(f"RECORDING DATA... ({progress:.1f}% gathered)")

            # Jika data sudah terkumpul genap 24.000 sampel, baru simpan ke CSV
            if len(self.recorded_data) >= self.target_samples:
                self.save_to_csv()

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recorded_data = [] 
            self.btn_start.setEnabled(False) 
            
            # 60 detik * 400 Hz = 24,000 sampel target
            self.target_samples = 24000 
            
            self.status_label.setText("RECORDING DATA... (0%)")
            self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #cb2431;")

    def save_to_csv(self):
        self.is_recording = False
        self.btn_start.setEnabled(True)
        self.status_label.setText("Finished! Data Saved")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #0055ff;")

        if not self.recorded_data:
            return

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_{timestamp_str}.csv"
        sampling_interval = 1.0 / 400.0 

        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Sumbu waktu diubah keterangannya menjadi Time (s) di header CSV
                writer.writerow(["Time (s)", "PPG_Red", "PPG_IR", "PPG_Green", "ECG", "Temp_Ambient", "Temp_Object"])
                
                # Loop menggunakan index untuk menghitung waktu relatif yang presisi
                for index, p in enumerate(self.recorded_data):
                    relative_time_s = index * sampling_interval
                    
                    writer.writerow([
                        f"{relative_time_s:.4f}", # Format 4 angka di belakang koma (misal: 0.0025, 0.0050)
                        p["ppg"]["red"],
                        p["ppg"]["ir"],
                        p["ppg"]["green"],
                        p["ecg"],
                        p["temperature"]["ambient"],
                        p["temperature"]["object"]
                    ])
            print(f"[INFO] Data successfully saved to {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV file: {e}")
            
    def stop_all(self):
        print("[INFO] Shutting down application...")
        self.is_recording = False
        if hasattr(self, 'worker'):
            self.worker.stop()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())