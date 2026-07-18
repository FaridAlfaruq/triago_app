import sys
import os
import csv
import time
from datetime import datetime
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import QThread, pyqtSignal, Qt

# Integrasi Custom Module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from akuisisi_data.get_stm32 import stream_stm32_data
from processing_data.preprocessing_LiveData import LiveSignalFilter

class STM32Worker(QThread):
    """Worker thread khusus penangan data serial STM32 tanpa blocking GUI"""
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

class PlotPage(QWidget):
    # Sinyal pemberitahuan ke main_gui ketika durasi selesai direkam
    recording_finished = pyqtSignal(str) 

    SAMPLE_RATE_HZ = 400
    WARMUP_DURATION_SEC = 2.0  # Durasi membuang transien filter (2 detik)
    RECORD_DURATION_SEC = 60.0 # Durasi rekam sinyal bersih (60 detik)

    def __init__(self):
        super().__init__()
        self.patient_data = {}
        self.is_recording = False
        self.recorded_data = []
        self.max_plot_points = 2000
        
        # TARGET TOTAL: 60 detik + 2 detik warm-up = 62 detik total akuisisi data
        self.total_target_samples = int((self.WARMUP_DURATION_SEC + self.RECORD_DURATION_SEC) * self.SAMPLE_RATE_HZ)
        self.live_filter = LiveSignalFilter()
        
        # Buffer visualisasi grafik
        self.time_buffer = []
        self.ecg_buffer = []
        self.ppg_red_buffer = []
        
        self.start_app_time = time.time()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)

        # 1. STATUS & PROGRESS RECORDER PANEL
        status_panel = QHBoxLayout()
        self.lbl_status = QLabel("SIAP MELAKUKAN PEMERIKSAAN")
        self.lbl_status.setStyleSheet("font-size: 16px; font-weight: bold; color: #2ECC71;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #2C2C2C; border: 1px solid #444; border-radius: 4px; text-align: center; color: white; }"
            "QProgressBar::chunk { background-color: #E74C3C; }"
        )
        
        status_panel.addWidget(self.lbl_status, stretch=2)
        status_panel.addWidget(self.progress_bar, stretch=3)
        layout.addLayout(status_panel)

        # 2. MULTI-PLOT CONTAINER GRAPHICS
        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground('#1E1E1E')
        layout.addWidget(self.win, stretch=5)

        # --- PLOT 1: ECG (Top Graph - Hijau) ---
        self.p1 = self.win.addPlot(title="ECG Signal")
        self.p1.setLabel('left', 'Amplitude')
        self.p1.setLabel('bottom', 'Time', units='s')
        self.p1.showGrid(x=True, y=True)
        self.ecg_curve = self.p1.plot(pen=pg.mkPen('g', width=1.5)) 

        self.win.nextRow()

        # --- PLOT 2: PPG (Bottom Graph - Merah) ---
        self.p2 = self.win.addPlot(title="PPG Signal (Infrared)")
        self.p2.setLabel('left', 'Light Intensity')
        self.p2.setLabel('bottom', 'Time', units='s')
        self.p2.showGrid(x=True, y=True)
        self.ppg_curve = self.p2.plot(pen=pg.mkPen('r', width=1.5)) 
        self.p2.setXLink(self.p1)

        # TAMBAHAN PERBAIKAN: Teks Petunjuk Stabilisasi (Overlay Visual)
        self.lbl_graph_hint = QLabel("Mempersiapkan sistem dan menstabilkan sensor...", self.win)
        self.lbl_graph_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_graph_hint.setStyleSheet("""
            background-color: rgba(30, 30, 30, 0.85); 
            color: #F39C12; 
            font-size: 16px; 
            font-weight: bold; 
            border: 1px dashed #F39C12; 
            border-radius: 4px; 
            padding: 10px;
        """)
        self.lbl_graph_hint.hide()

        # 3. KONDISIONAL LIVE TEMPERATURE PANEL
        temp_container = QWidget()
        temp_container.setStyleSheet("background-color: #1E1E1E; border-radius: 6px; padding: 10px;")
        temp_layout = QHBoxLayout(temp_container)
        
        self.lbl_temp_obj = QLabel("Suhu Tubuh Pasien: -- °C")
        self.lbl_temp_obj.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF;")
        
        temp_layout.addWidget(self.lbl_temp_obj)
        temp_layout.addStretch()
        layout.addWidget(temp_container)

    def resizeEvent(self, event):
        """Menjaga teks hint agar selalu berada tepat di tengah grid grafik saat window diubah ukurannya"""
        super().resizeEvent(event)
        if self.lbl_graph_hint.isVisible():
            # Pusatkan teks overlay di area GraphicsLayoutWidget
            w = 400
            h = 50
            x = int((self.win.width() - w) / 2)
            y = int((self.win.height() - h) / 2)
            self.lbl_graph_hint.setGeometry(x, y, w, h)

    def start_session(self, patient_info):
        """Dipanggil oleh main_gui saat perawat menekan start di Halaman 1"""
        self.patient_data = patient_info
        self.recorded_data = []
        self.time_buffer.clear()
        self.ecg_buffer.clear()
        self.ppg_red_buffer.clear()
        self.progress_bar.setValue(0)
        
        self.is_recording = True
        self.start_app_time = time.time()
        
        self.lbl_status.setText(f"RECORDING DATA: {self.patient_data['nama'].upper()}")
        self.lbl_status.setStyleSheet("font-size: 16px; font-weight: bold; color: #E74C3C;")

        # Tampilkan overlay inisialisasi sensor di awal
        w = 400
        h = 50
        x = int((self.win.width() - w) / 2)
        y = int((self.win.height() - h) / 2)
        self.lbl_graph_hint.setGeometry(x, y, w, h)
        self.lbl_graph_hint.show()

        # Jalankan background thread serial
        self.worker = STM32Worker()
        self.worker.data_received.connect(self.handle_new_packet)
        self.worker.start()

    def handle_new_packet(self, packet):
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
                # Sinyal sudah difilter di atas, tapi JANGAN di-plot ke dalam buffer grafik
                self.progress_bar.setValue(0)
                return

            # FASE B: Masa Perekaman Riil (Setelah detik ke-2 / Sinyal Sudah Stabil)
            if self.lbl_graph_hint.isVisible():
                self.lbl_graph_hint.hide() 

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

    def update_temperature_ui(self, t_obj, t_amb):
        """Menerapkan aturan kondisional warna klinis pada label suhu tubuh"""
        if t_obj < 35.0:
            color = "#3498DB"  # Biru (Hipotermia Berat)
        elif 35.0 <= t_obj < 37.5:
            color = "#2ECC71"  # Hijau (Normal)
        elif 37.5 <= t_obj < 38.5:
            color = "#F39C12"  # Oranye (Moderate Fever)
        else:
            color = "#E74C3C"  # Merah (Suhu Tinggi/Hipertermia)
            
        self.lbl_temp_obj.setText(f"Suhu Tubuh Pasien: {t_obj:.2f} °C")
        self.lbl_temp_obj.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")

    def stop_and_save_data(self):
        self.is_recording = False
        if hasattr(self, 'worker'):
            self.worker.stop()
            
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_{self.patient_data['nama']}_{timestamp_str}.csv"
        sampling_interval = 1.0 / self.SAMPLE_RATE_HZ

        # Hitung batasan indeks pemotongan buffer (buang data 2 detik awal)
        warmup_samples = int(self.WARMUP_DURATION_SEC * self.SAMPLE_RATE_HZ)
        clean_data_subset = self.recorded_data[warmup_samples:]

        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time (s)", "PPG_Red", "PPG_IR", "PPG_Green", "ECG", "Temp_Ambient", "Temp_Object"])
                
                # Simpan subset data bersih. Indeks waktu di-reset dari 0.0000s
                for index, p in enumerate(clean_data_subset):
                    relative_time_s = index * sampling_interval
                    writer.writerow([
                        f"{relative_time_s:.4f}",
                        p["ppg"]["red"], p["ppg"]["ir"], p["ppg"]["green"],
                        p["ecg"], p["temperature"]["ambient"], p["temperature"]["object"]
                    ])
            print(f"[INFO] Data saved securely (transient removed) to {filename}")
            self.recording_finished.emit(filename)
        except Exception as e:
            print(f"[ERROR] Save failed: {e}")

    def close_threads(self):
        """Memastikan thread mati jika aplikasi ditutup paksa"""
        if hasattr(self, 'worker'):
            self.worker.stop()