import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QApplication, QGraphicsOpacityEffect
)
from PyQt6.QtCore import (
    QThread, pyqtSignal, Qt, QPropertyAnimation, QEasingCurve,
    pyqtProperty, QRectF
)
from PyQt6.QtGui import QPainter, QColor, QPainterPath, QFont, QPixmap

# Menambahkan direktori utama (TriaGo) ke dalam sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modul pemrosesan ECG dan PPG
from processing_data.processing_data import ECGProcessor, PPGProcessor


# =====================================================================
# WORKER: Thread Pemrosesan Sinyal Asynchronous (ECG & PPG)
# =====================================================================
class ProcessingWorker(QThread):
    """Worker thread untuk mengeksekusi pipeline preprocessing & ekstraksi fitur

    ECG dan PPG secara asynchronous agar UI PyQt6 tidak freeze/lag.
    """

    status_updated = pyqtSignal(str, int)  # Sinyal update (teks_status, persentase)
    processing_finished = pyqtSignal(dict)  # Sinyal output dictionary hasil

    def __init__(
        self,
        raw_ecg,
        raw_time,
        raw_red=None,
        raw_ir=None,
        fs_orig=400,
        parent=None,
    ):
        super().__init__(parent)
        self.raw_ecg = raw_ecg
        self.raw_time = raw_time
        self.raw_red = raw_red
        self.raw_ir = raw_ir
        self.fs_orig = fs_orig
        
        # Inisialisasi Processor ECG & PPG
        self.ecg_processor = ECGProcessor(target_fs=125)
        self.ppg_processor = PPGProcessor(target_fs=125)

    def run(self):
        # -----------------------------------------------------------------
        # TAHAP 1: Preprocessing & Filtering Sinyal ECG (0% - 25%)
        # -----------------------------------------------------------------
        self.status_updated.emit("Downsampling ECG ke 125 Hz...", 10)
        self.msleep(150)

        ecg_125, time_125 = self.ecg_processor.downsample(
            self.raw_ecg, self.raw_time, fs=self.fs_orig, fs_target=125
        )

        self.status_updated.emit("Menyaring noise ECG (Notch 50Hz & Detrending)...", 20)
        sig_notch = self.ecg_processor.notch(ecg_125, freq=50.0, fs=125)
        sig_detrend = self.ecg_processor.detrending(sig_notch, fs=125)

        self.status_updated.emit("Menghaluskan sinyal ECG (LPF & Savgol)...", 25)
        sig_lpf = self.ecg_processor.lowpass(sig_detrend, lowcut=35.0, fs=125)
        ecg_smooth = self.ecg_processor.savgol(sig_lpf, window_size=11, poly_order=2)
        self.msleep(150)

        # -----------------------------------------------------------------
        # TAHAP 2: Ekstraksi Fitur ECG (R-Peak, HR, & RR) (25% - 50%)
        # -----------------------------------------------------------------
        self.status_updated.emit("Mendeteksi R-Peak & Menghitung HR (ECG)...", 35)
        r_peaks, noise_peaks = self.ecg_processor.detect_r_peaks(ecg_125, fs=125)
        hr_ecg = self.ecg_processor.calculate_heart_rate(r_peaks, fs=125)
        self.msleep(150)

        self.status_updated.emit("Menganalisis Respiratory Rate (RR)...", 50)
        resp_rate, resp_signal, resp_peaks = (
            self.ecg_processor.calculate_respiration_rate(ecg_125, r_peaks, fs=125)
        )
        self.msleep(150)

        # -----------------------------------------------------------------
        # TAHAP 3: Pemrosesan Sinyal PPG (Red & IR) (50% - 90%)
        # -----------------------------------------------------------------
        if self.raw_red is not None and self.raw_ir is not None and len(self.raw_red) > 0:
            self.status_updated.emit("Menjalankan pipeline preprocessing PPG...", 65)
            self.msleep(150)

            self.status_updated.emit("Menghitung SpO2, PI Red, dan PI IR...", 80)
            # Eksekusi 7 Tahapan Pipeline PPG
            ppg_results = self.ppg_processor.process_ppg(
                raw_time=self.raw_time,
                raw_red=self.raw_red,
                raw_ir=self.raw_ir,
                fs_orig=self.fs_orig
            )
            self.msleep(150)

            spo2 = ppg_results['spo2']
            pi_red = ppg_results['pi_red']
            pi_ir = ppg_results['pi_ir']
            red_clean = ppg_results['red_clean']
            ir_clean = ppg_results['ir_clean']
            ppg_hr = ppg_results['ppg_hr']
        else:
            # Fallback jika data PPG tidak tersedia
            spo2 = 0.0
            pi_red = 0.0
            pi_ir = 0.0
            red_clean = np.array([])
            ir_clean = np.array([])
            ppg_hr = 0.0

        # -----------------------------------------------------------------
        # TAHAP 4: Konsolidasi & Selesai (100%)
        # -----------------------------------------------------------------
        self.status_updated.emit("Pemrosesan Data Selesai!", 100)
        self.msleep(200)

        # Output gabungan seluruh fitur ECG & PPG
        results = {
        # Data Vitalsign (Angka)
        "hr": hr_ecg,
        "rr": resp_rate,
        "spo2": spo2,
        "pi_red": pi_red,
        "pi_ir": pi_ir,
        "ppg_hr": ppg_hr,
        
        # Sinyal Array untuk Plot
        "time_125": time_125,       # Sumbu X (Waktu)
        "ecg_smooth": ecg_smooth,   # Sumbu Y (Sinyal ECG)
        "red_smooth": red_clean,    # Sumbu Y (Sinyal PPG Red)
        "ir_smooth": ir_clean,      # Sumbu Y (Sinyal PPG IR)
        
        # Puncak gelombang (Opsional jika ingin menandai peak di plot)
        "r_peaks": r_peaks,
        }      

        self.processing_finished.emit(results)


# =====================================================================
# CUSTOM PROGRESS BAR: Didesain Sesuai Komponen Figma
# =====================================================================
class AnimatedProgressBar(QWidget):
    """Progress bar kustom berbentuk pill putih dengan teks persentase biru tua."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self.setFixedHeight(34)

        self._animation = QPropertyAnimation(self, b"value")
        self._animation.setDuration(200)
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

        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2

        # 1. Track Latar Belakang
        track_path = QPainterPath()
        track_path.addRoundedRect(rect, radius, radius)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(33, 72, 137, 50))
        painter.drawPath(track_path)

        # 2. Isi Progress Bar (Putih Solid)
        full_width = rect.width()
        chunk_width = full_width * (self._value / 100.0)

        if chunk_width > rect.height():
            chunk_rect = QRectF(rect.x(), rect.y(), chunk_width, rect.height())
            chunk_path = QPainterPath()
            chunk_path.addRoundedRect(chunk_rect, radius, radius)

            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#FFFFFF"))
            painter.drawPath(chunk_path)
            painter.restore()
        elif chunk_width > 0:
            painter.setBrush(QColor("#FFFFFF"))
            painter.drawEllipse(QRectF(rect.x(), rect.y(), rect.height(), rect.height()))

        # 3. Teks Persentase
        painter.setPen(QColor("#214889"))
        font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{int(round(self._value))}%")


# =====================================================================
# HALAMAN UTAMA: LoadingPage
# =====================================================================
class LoadingPage(QWidget):
    def __init__(self):
        super().__init__()
        self._status_effect = None
        self._fade_out_anim = None
        self._fade_in_anim = None
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background-color: #F6FFEC;")
        
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setContentsMargins(50, 50, 50, 50)

        # 1. Logo TriaGO
        self.lbl_logo = QLabel()
        self.lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_logo.setStyleSheet("background: transparent; margin-bottom: 10px;")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(current_dir, r"C:\Users\Adyty\Documents\Farid ITS\TriaGo\asset\logo.png") 
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            self.lbl_logo.setPixmap(pixmap.scaledToWidth(420, Qt.TransformationMode.SmoothTransformation))
        else:
            self.lbl_logo.setText("TriaGO")
            self.lbl_logo.setStyleSheet("font-size: 48px; font-weight: 900; color: #214889; background: transparent;")
        main_layout.addWidget(self.lbl_logo)

        # 2. Container Card
        card_container = QFrame()
        card_container.setStyleSheet("""
            QFrame {
                background-color: #214889; 
                border-radius: 28px; 
            }
        """)
        card_container.setFixedWidth(600)
        card_container.setFixedHeight(140)

        card_layout = QVBoxLayout(card_container)
        card_layout.setContentsMargins(35, 25, 35, 25)
        card_layout.setSpacing(12)

        # 3. Progress Bar
        self.progress_bar = AnimatedProgressBar()
        card_layout.addWidget(self.progress_bar)

        # 4. Status Label
        self.lbl_status = QLabel("Mempersiapkan perangkat...")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("""
            QLabel {
                font-size: 15px;
                font-weight: 600;
                color: #FFFFFF;
                font-style: italic;
                background: transparent;
            }
        """)
        card_layout.addWidget(self.lbl_status)

        # Animasi Fade Text
        self._status_effect = QGraphicsOpacityEffect(self.lbl_status)
        self._status_effect.setOpacity(1.0)
        self.lbl_status.setGraphicsEffect(self._status_effect)

        main_layout.addWidget(card_container)

    def start_processing(
        self,
        raw_ecg,
        raw_time,
        raw_red=None,
        raw_ir=None,
        fs_orig=400,
    ):
        """Memicu proses pemrosesan data ECG, PPG, HR, RR, SpO2, dan PI."""
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Memulai pemrosesan data...")

        self.worker = ProcessingWorker(
            raw_ecg=raw_ecg,
            raw_time=raw_time,
            raw_red=raw_red,
            raw_ir=raw_ir,
            fs_orig=fs_orig,
        )
        self.worker.status_updated.connect(self.update_ui_state)
        self.worker.processing_finished.connect(self.handle_processing_completion)
        self.worker.start()

    def handle_processing_completion(self, results):
        print("[LOG] Pemrosesan data selesai!")
        self.lbl_status.setText("Pemrosesan Data Selesai!")

        # 1. Generate string timestamp saat ini (Format: TahunBulanTanggal_JamMenitDetik)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 2. Buat nama file dinamis menggunakan f-string
        filename = f"ekstraksi_data_{timestamp}.csv"

        # 3. Menyusun data parameter vital sign
        summary_data = {
            "HR_ECG_BPM": [results['hr']],
            "RR_RPM": [results['rr']],
            "SpO2_Percent": [results['spo2']],
            "PI_Red_Percent": [results['pi_red']],
            "PI_IR_Percent": [results['pi_ir']],
            "HR_PPG_BPM": [results['ppg_hr']]
        }
        df_summary = pd.DataFrame(summary_data)
        # Menyimpan file ke folder project
        df_summary.to_csv(filename, index=False)
        print(f"[LOG] Data berhasil disimpan ke: {filename}")

        # Alirkan hasil ke Window Utama
        if hasattr(self, "parent_main_win"):
            self.parent_main_win.processed_results = results
            self.parent_main_win.go_to_live_data_page()
        else:
            self.lbl_status.setText(
                f"Selesai!!!"
            )

    def update_ui_state(self, text, progress_value):
        """Singkronisasi progress bar dan transisi teks status."""
        self.progress_bar.animate_to(progress_value)
        if text != self.lbl_status.text():
            self._fade_to_text(text)

    def _fade_to_text(self, new_text):
        if self._fade_out_anim is not None:
            self._fade_out_anim.stop()
        if self._fade_in_anim is not None:
            self._fade_in_anim.stop()

        self._fade_out_anim = QPropertyAnimation(self._status_effect, b"opacity")
        self._fade_out_anim.setDuration(100)
        self._fade_out_anim.setStartValue(1.0)
        self._fade_out_anim.setEndValue(0.0)

        def _swap_and_fade_in():
            self.lbl_status.setText(new_text)
            self._fade_in_anim = QPropertyAnimation(self._status_effect, b"opacity")
            self._fade_in_anim.setDuration(150)
            self._fade_in_anim.setStartValue(0.0)
            self._fade_in_anim.setEndValue(1.0)
            self._fade_in_anim.start()

        self._fade_out_anim.finished.connect(_swap_and_fade_in)
        self._fade_out_anim.start()

    def close_threads(self):
        """Memastikan thread mati jika aplikasi ditutup paksa."""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()


# =====================================================================
# BLOK TEST RUN INDEPENDEN
# =====================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    test_loader = LoadingPage()
    test_loader.resize(1920, 1080)
    test_loader.setWindowTitle("Pratinjau Kios TriaGO - Loading & Pemrosesan Data")
    test_loader.show()

    # Data Dummy Mentah (400 Hz, 10 Detik)
    fs_dummy = 400
    duration = 10
    t_dummy = np.linspace(0, duration, duration * fs_dummy)

    # Sinyal Sintesis Dummy
    ecg_dummy = (
        np.sin(2 * np.pi * 1.25 * t_dummy)
        + 0.5 * np.sin(2 * np.pi * 50 * t_dummy)
        + np.random.normal(0, 0.1, len(t_dummy))
    )
    ppg_red_dummy = 20000 + 500 * np.sin(2 * np.pi * 1.25 * t_dummy) + np.random.normal(0, 20, len(t_dummy))
    ppg_ir_dummy = 30000 + 600 * np.sin(2 * np.pi * 1.25 * t_dummy) + np.random.normal(0, 20, len(t_dummy))

    # Jalankan proses
    test_loader.start_processing(
        raw_ecg=ecg_dummy,
        raw_time=t_dummy,
        raw_red=ppg_red_dummy,
        raw_ir=ppg_ir_dummy,
        fs_orig=fs_dummy,
    )

    sys.exit(app.exec())