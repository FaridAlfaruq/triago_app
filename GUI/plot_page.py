import sys
import os
import csv
import time
from datetime import datetime
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QApplication, QGridLayout
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPropertyAnimation, QEasingCurve, pyqtProperty, QRectF
from PyQt6.QtGui import QPixmap, QPainter, QPainterPath, QColor, QFont, QPen

# Integrasi dengan Module Hardware & Filter Riil TriaGO
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from akusisi_data.get_stm32 import stream_stm32_data
from processing_data.preprocessing_LiveData import LiveSignalFilter


# =====================================================================
# KUSTOM WIDGET: AnimatedProgressBar dengan Warna Tema TriaGO
# =====================================================================
class AnimatedProgressBar(QWidget):
    """Progress bar kustom berbentuk pill dengan animasi halus, menggunakan tema warna TriaGO."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self.setFixedHeight(30)  # Menjaga konsistensi proporsi tata letak UI

        self._animation = QPropertyAnimation(self, b"value")
        self._animation.setDuration(250)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def getValue(self):
        return self._value

    def setValue(self, v):
        self._value = max(0.0, min(100.0, v))
        self.update()

    value = pyqtProperty(float, fget=getValue, fset=setValue)

    def animate_to(self, target_value: int):
        self._animation.stop()
        self._animation.setStartValue(self._value)
        self._animation.setEndValue(float(target_value))
        self._animation.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Mengurangi koordinat sedikit agar garis border tidak terpotong tepi widget
        rect = QRectF(self.rect()).adjusted(0.75, 0.75, -0.75, -0.75)
        radius = rect.height() / 2

        # 1. Gambar Track Latar Belakang (Putih dengan Border Sage Green #C2D5BB)
        track_path = QPainterPath()
        track_path.addRoundedRect(rect, radius, radius)
        painter.setPen(QPen(QColor("#C2D5BB"), 1.5))
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawPath(track_path)

        # 2. Gambar Isi Progress Bar (Deep Blue #214889)
        full_width = rect.width()
        chunk_width = full_width * (self._value / 100.0)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#214889"))

        if chunk_width > rect.height(): 
            chunk_rect = QRectF(rect.x(), rect.y(), chunk_width, rect.height())
            chunk_path = QPainterPath()
            chunk_path.addRoundedRect(chunk_rect, radius, radius)
            painter.drawPath(chunk_path)
        elif chunk_width > 0:
            # Mengatasi bug kotak 0-5%: langsung berbentuk melengkung lingkaran penuh
            painter.drawEllipse(QRectF(rect.x(), rect.y(), rect.height(), rect.height()))

        # 3. Gambar Teks Persentase di Tengah dengan Warna Kontras Dinamis
        # Mengubah teks menjadi putih jika tertutup chunk biru agar tetap terbaca
        if self._value > 52.0:
            painter.setPen(QColor("#FFFFFF"))
        else:
            painter.setPen(QColor("#214889"))

        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{int(round(self._value))}%")


# =====================================================================
# WORKER THREAD: Murni Mengambil Data Riil dari Sensor STM32
# =====================================================================
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


# =====================================================================
# HALAMAN UTAMA: PlotPage
# =====================================================================
class PlotPage(QWidget):
    recording_finished = pyqtSignal(str) 
    warmup_progress = pyqtSignal(str, int)  # Mengirim teks status & nilai persen (0-100)
    warmup_finished = pyqtSignal()          # Dipanggil tepat saat detik ke-2 selesai
    
    SAMPLE_RATE_HZ = 400
    WARMUP_DURATION_SEC = 2.0  
    RECORD_DURATION_SEC = 60.0

    def __init__(self):
        super().__init__()
        self.patient_data = {}
        self.is_recording = False
        self.recorded_data = []
        self.max_plot_points = 2000
        
        self.total_target_samples = int((self.WARMUP_DURATION_SEC + self.RECORD_DURATION_SEC) * self.SAMPLE_RATE_HZ)
        self.live_filter = LiveSignalFilter()
        
        # Buffer visualisasi grafik
        self.time_buffer = []
        self.ecg_buffer = []
        self.ppg_buffer = []
        
        # Counter untuk membatasi fps menggambar grafik
        self.ui_update_counter = 0
        self.plot_render_interval = 12 
        
        self.setup_ui()

    def setup_ui(self):
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background-color: #F6FFEC;")
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 20, 40, 20)
        main_layout.setSpacing(10)

        # =====================================================================
        # BAGIAN 1: HEADER LAYOUT (JUDUL KIRI + PILL PROGRESS BAR + LOGO KANAN)
        # =====================================================================
        header_layout = QHBoxLayout()
        header_layout.setSpacing(25)
        header_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        
        lbl_title = QLabel("Melakukan Perekaman")
        lbl_title.setStyleSheet("font-size: 28px; font-weight: 800; color: #214889; background: transparent;")
        header_layout.addWidget(lbl_title)
        
        # Menggunakan AnimatedProgressBar Kustom Baru
        self.progress_bar = AnimatedProgressBar()
        self.progress_bar.setMinimumWidth(320)
        header_layout.addWidget(self.progress_bar, stretch=1)
        
        lbl_logo = QLabel()
        lbl_logo.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl_logo.setStyleSheet("background: transparent;")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(current_dir, r"C:\Users\Adyty\Documents\Farid ITS\TriaGo\asset\logo.png") 
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            lbl_logo.setPixmap(pixmap.scaledToWidth(240, Qt.TransformationMode.SmoothTransformation))
        else:
            lbl_logo.setText("TriaGO")
            lbl_logo.setStyleSheet("font-size: 32px; font-weight: 900; color: #214889;")
        header_layout.addWidget(lbl_logo)
        
        main_layout.addLayout(header_layout)

        # =====================================================================
        # BAGIAN 2: PLOTTING CONTAINER 
        # =====================================================================
        # --- PLOT AREA 1: SINYAL ECG ---
        lbl_ecg_tag = QLabel("Sinyal ECG")
        lbl_ecg_tag.setStyleSheet("font-size: 18px; font-weight: bold; color: #214889; background: transparent;")
        main_layout.addWidget(lbl_ecg_tag)
        
        ecg_frame = QFrame()
        ecg_frame.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        ecg_frame_layout = QVBoxLayout(ecg_frame)
        ecg_frame_layout.setContentsMargins(8, 8, 8, 8) 
        
        self.ecg_plot = pg.PlotWidget()
        self.ecg_plot.setBackground('#FFFFFF')
        self.ecg_plot.showAxis('left', True)   
        self.ecg_plot.showAxis('bottom', True)
        self.ecg_plot.setLabel('bottom', 'Waktu (s)', color='#214889')
        self.ecg_plot.setLabel('left', 'Amplitudo', color='#214889')
        self.ecg_plot.getAxis('left').setPen('#214889')
        self.ecg_plot.getAxis('bottom').setPen('#214889')
        self.ecg_plot.getAxis('left').setTextPen('#214889')
        self.ecg_plot.getAxis('bottom').setTextPen('#214889')
        self.ecg_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.ecg_plot.getViewBox().enableAutoRange(axis='y')
        self.ecg_plot.plotItem.layout.setContentsMargins(15, 10, 15, 10)
        
        self.ecg_curve = self.ecg_plot.plot(pen=pg.mkPen(color='#214889', width=2))
        ecg_frame_layout.addWidget(self.ecg_plot)
        main_layout.addWidget(ecg_frame, stretch=1)

        # --- PLOT AREA 2: SINYAL PPG ---
        lbl_ppg_tag = QLabel("Sinyal PPG")
        lbl_ppg_tag.setStyleSheet("font-size: 18px; font-weight: bold; color: #214889; background: transparent;")
        main_layout.addWidget(lbl_ppg_tag)
        
        ppg_frame = QFrame()
        ppg_frame.setStyleSheet("QFrame { border: 1.5px solid #C2D5BB; border-radius: 12px; background-color: #FFFFFF; }")
        ppg_frame_layout = QVBoxLayout(ppg_frame)
        ppg_frame_layout.setContentsMargins(8, 8, 8, 8)
        
        self.ppg_plot = pg.PlotWidget()
        self.ppg_plot.setBackground('#FFFFFF')
        self.ppg_plot.showAxis('left', True)
        self.ppg_plot.showAxis('bottom', True)
        self.ppg_plot.setLabel('bottom', 'Waktu (s)', color='#214889')
        self.ppg_plot.setLabel('left', 'Amplitudo', color='#214889')
        self.ppg_plot.getAxis('left').setPen('#214889')
        self.ppg_plot.getAxis('bottom').setPen('#214889')
        self.ppg_plot.getAxis('left').setTextPen('#214889')
        self.ppg_plot.getAxis('bottom').setTextPen('#214889')
        self.ppg_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.ppg_plot.getViewBox().enableAutoRange(axis='y')
        self.ppg_plot.plotItem.layout.setContentsMargins(15, 10, 15, 10)
        
        self.ppg_curve = self.ppg_plot.plot(pen=pg.mkPen(color='#214889', width=2))
        self.ppg_plot.setXLink(self.ecg_plot) 
        ppg_frame_layout.addWidget(self.ppg_plot)
        main_layout.addWidget(ppg_frame, stretch=1)

    def start_session(self, patient_info):
        self.patient_data = patient_info
        self.recorded_data = []
        self.time_buffer.clear()
        self.ecg_buffer.clear()
        self.ppg_buffer.clear()
        self.progress_bar.setValue(0) # Reset instan ke 0% tanpa animasi lag awal
        self.ui_update_counter = 0
        
        self.is_recording = True
        
        self.worker = STM32Worker()
        self.worker.data_received.connect(self.handle_new_packet)
        self.worker.start()

    def handle_new_packet(self, packet):
        if not self.is_recording:
            return
            
        self.recorded_data.append(packet)
        current_samples_count = len(self.recorded_data)
        warmup_samples = int(self.WARMUP_DURATION_SEC * self.SAMPLE_RATE_HZ)

        # =====================================================================
        # SOLUSI INTI: Alirkan data ke filter SEJAK DETIK KE-0 (Fase Warmup)
        # Langkah ini membuat register internal filter konvergen & stabil duluan
        # =====================================================================
        ecg_val = packet["ecg"]
        ppg_red_val = packet["ppg"]["red"]

        clean_ecg = self.live_filter.filter_ecg(ecg_val)
        clean_ppg = self.live_filter.filter_ppg(ppg_red_val)

        # 1. Logika Sinkronisasi Fase Warmup (Tetap men-drive LoadingPage)
        if current_samples_count <= warmup_samples:
            progress_warmup = (current_samples_count / warmup_samples) * 100
            self.warmup_progress.emit("Menstabilkan sensor....", int(progress_warmup))
            
            if current_samples_count == warmup_samples:
                self.warmup_finished.emit()
            return  # Keluar di sini, data transien awal TIDAK MASUK ke grafik plot

        # =====================================================================
        # 2. FASE PEREKAMAN RIIL (Detik > 2.0 / Sampel > 800)
        # Sinyal yang masuk ke sini dijamin sudah bersih dari transien filter
        # =====================================================================
        # Sumbu X akan mulai bersih dari pecahan detik pertama (0.0025s, 0.0050s, dst)
        recorded_duration = (current_samples_count - warmup_samples) / self.SAMPLE_RATE_HZ
        
        self.time_buffer.append(recorded_duration)
        self.ecg_buffer.append(clean_ecg)
        self.ppg_buffer.append(clean_ppg)

        if len(self.time_buffer) > self.max_plot_points:
            self.time_buffer.pop(0)
            self.ecg_buffer.pop(0)
            self.ppg_buffer.pop(0)

        # PROTEKSI ANTI-FREEZE WINDOW MINIMIZE
        if self.isMinimized():
            if current_samples_count >= self.total_target_samples:
                self.stop_and_save_data()
            return

        # Memicu update animasi progress bar riil (60 detik)
        if current_samples_count % 20 == 0:
            progress = (recorded_duration / self.RECORD_DURATION_SEC) * 100
            self.progress_bar.animate_to(int(progress))

        # Downsampling visualisasi kurva biosinyal ke layar monitor
        self.ui_update_counter += 1
        if self.ui_update_counter % self.plot_render_interval == 0:
            self.ecg_curve.setData(self.time_buffer, self.ecg_buffer)
            self.ppg_curve.setData(self.time_buffer, self.ppg_buffer)

        if current_samples_count >= self.total_target_samples:
            self.stop_and_save_data()

    def stop_and_save_data(self):
        self.is_recording = False
        self.close_threads()
            
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Data_raw_{timestamp_str}.csv"
        sampling_interval = 1.0 / self.SAMPLE_RATE_HZ

        warmup_samples = int(self.WARMUP_DURATION_SEC * self.SAMPLE_RATE_HZ)
        clean_data_subset = self.recorded_data[warmup_samples:]

        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time (s)", "PPG_Red", "PPG_IR", "PPG_Green", "ECG", "Temp_Ambient", "Temp_Object"])
                
                for index, p in enumerate(clean_data_subset):
                    relative_time_s = index * sampling_interval
                    writer.writerow([
                        f"{relative_time_s:.4f}",
                        p["ppg"]["red"], p["ppg"]["ir"], p["ppg"]["green"],
                        p["ecg"], p["temperature"]["ambient"], p["temperature"]["object"]
                    ])
            print(f"[LOG SUCCESS] Data riil berhasil disimpan ke: {filename}")
            
            # Memastikan visual menyentuh 100% penuh di akhir rekaman
            self.progress_bar.animate_to(100)
            
            self.recording_finished.emit(filename)
        except Exception as e:
            print(f"[ERROR WRITE CSV FAILURE]: {e}")

    def close_threads(self):
        self.is_recording = False
        if hasattr(self, 'worker') and self.worker is not None:
            try:
                self.worker.data_received.disconnect(self.handle_new_packet)
            except TypeError:
                pass
            self.worker.stop()
            self.worker = None
            print("[LOG SUCCESS] Thread STM32 di-shutdown dengan aman.")

    def closeEvent(self, event):
        self.close_threads()
        event.accept()


# =====================================================================
# DIRECT RUN STANDALONE TIMING TEST
# =====================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    standalone_window = PlotPage()
    standalone_window.setWindowTitle("Uji Coba Hardware Riil - Halaman Perekaman TriaGO")
    standalone_window.showMaximized()
    
    standalone_window.start_session({"nama": "Pasien Uji Riil"})
    
    sys.exit(app.exec())