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

# Import modul pemrosesan ECG dan PPG dari processing_data
from processing_data.processing_data import ECGProcessor, PPGProcessor
from ml_xgboost.triage_xgboost import TriageModelError, TriageOnnxPredictor
from ml_xgboost.triage_xgboost.preprocessing import TriagePreprocessingError


# =====================================================================
# WORKER: Thread Pemrosesan Sinyal & Machine Learning Asynchronous
# =====================================================================
class ProcessingWorker(QThread):
    status_updated = pyqtSignal(str, int)   # Sinyal update status & persen GUI
    processing_finished = pyqtSignal(dict) # Sinyal kirim dictionary hasil akhir

    def __init__(
        self,
        raw_ecg,
        raw_time,
        raw_red=None,
        raw_ir=None,
        patient_info=None,
        fs_orig=400,
        triage_predictor=None,
        triage_model_error=None,
        parent=None,
    ):
        super().__init__(parent)
        self.raw_ecg = raw_ecg
        self.raw_time = raw_time
        self.raw_red = raw_red
        self.raw_ir = raw_ir
        self.patient_info = patient_info or {}
        self.fs_orig = fs_orig
        self.triage_predictor = triage_predictor
        self.triage_model_error = triage_model_error
        
        # Inisialisasi Processor ECG & PPG
        self.ecg_processor = ECGProcessor(target_fs=125)
        self.ppg_processor = PPGProcessor(target_fs=125)

    def run(self):
            # -----------------------------------------------------------------
            # TAHAP 1: Preprocessing & Filtering Sinyal ECG (0% - 25%)
            # -----------------------------------------------------------------
            self.status_updated.emit("Downsampling & Filtering Sinyal ECG...", 15)
            self.msleep(100)

            ecg_125, time_125 = self.ecg_processor.downsample(
                self.raw_ecg, self.raw_time, fs=self.fs_orig, fs_target=125
            )

            sig_notch = self.ecg_processor.notch(ecg_125, freq=50.0, fs=125)
            sig_detrend = self.ecg_processor.detrending(sig_notch, fs=125)
            sig_lpf = self.ecg_processor.lowpass(sig_detrend, lowcut=35.0, fs=125)
            ecg_smooth = self.ecg_processor.savgol(sig_lpf, window_size=11, poly_order=2)

            # -----------------------------------------------------------------
            # TAHAP 2: Ekstraksi Fitur ECG (R-Peak, HR, & RR) (25% - 50%)
            # -----------------------------------------------------------------
            self.status_updated.emit("Mendeteksi R-Peak, HR & Respiratory Rate...", 40)
            self.msleep(100)

            r_peaks, noise_peaks = self.ecg_processor.detect_r_peaks(ecg_125, fs=125)
            hr_ecg = self.ecg_processor.calculate_heart_rate(r_peaks, fs=125)
            resp_rate, resp_signal, resp_peaks = (
                self.ecg_processor.calculate_respiration_rate(ecg_125, r_peaks, fs=125)
            )

            # -----------------------------------------------------------------
            # TAHAP 3: Pemrosesan Sinyal PPG 7 Tahap (50% - 75%)
            # -----------------------------------------------------------------
            self.status_updated.emit("Menjalankan Pemrosesan Sinyal PPG (SpO2 & PI)...", 65)
            self.msleep(100)

            if self.raw_red is not None and self.raw_ir is not None and len(self.raw_red) > 0:
                ppg_results = self.ppg_processor.process_ppg(
                    raw_time=self.raw_time,
                    raw_red=self.raw_red,
                    raw_ir=self.raw_ir,
                    fs_orig=self.fs_orig
                )

                spo2 = ppg_results['spo2']
                pi_red = ppg_results['pi_red']
                pi_ir = ppg_results['pi_ir']
                red_clean = ppg_results['red_clean']
                ir_clean = ppg_results['ir_clean']
                ppg_hr = ppg_results['ppg_hr']
            else:
                spo2 = 98.0
                pi_red, pi_ir, ppg_hr = 0.0, 0.0, 0.0
                red_clean, ir_clean = np.array([]), np.array([])

            # -----------------------------------------------------------------
            # TAHAP 4: Machine Learning Triage & SHAP Analysis (75% - 95%)
            # -----------------------------------------------------------------
            self.status_updated.emit("Menjalankan Feature Engineering & ML Triage...", 85)
            self.msleep(100)

            patient = self.patient_info or {}

            # -----------------------------------------------------------------
            # ESTIMASI SUHU TUBUH INTI (T_core) & BURTON (T_b)
            # -----------------------------------------------------------------
            WEIGHT_CORE = 0.64
            WEIGHT_SKIN = 0.36

            # Ambil nilai mentah dari patient_info
            raw_temp_skin = patient.get('temp_skin')
            raw_temp_amb = patient.get('temp_ambient')
            input_warnings = []

            # Logging warning jika salah satu/kedua suhu bernilai None
            if raw_temp_skin is None or raw_temp_amb is None:
                print(f"[WARN WORKER TEMP] ⚠️ Parameter 'temp_skin' ({raw_temp_skin}) atau 'temp_ambient' ({raw_temp_amb}) tidak ditemukan pada patient_info! Memakai nilai default (Kulit: 34.5°C, Amb: 28.0°C).")
                input_warnings.append("temperature memakai nilai fallback")

            temp_skin_val = float(raw_temp_skin if raw_temp_skin is not None else 34.5)
            temp_amb_val = float(raw_temp_amb if raw_temp_amb is not None else 28.0)

            # 1. Hitung Estimasi Suhu Inti (T_core)
            k_env = WEIGHT_SKIN / WEIGHT_CORE  # 0.36 / 0.64 = 0.5625
            t_core_calc = temp_skin_val + k_env * (temp_skin_val - temp_amb_val)

            # 2. Hitung Suhu Rata-rata Tubuh Burton (T_b)
            t_burton_calc = (WEIGHT_CORE * t_core_calc) + (WEIGHT_SKIN * temp_skin_val)

            # Gunakan T_core hasil kalkulasi sebagai temp_val (dikunci pada batas biologis aman)
            temp_val = float(np.clip(t_core_calc, 30.0, 43.0))

            print(f"[LOG BURTON FORMULA] T_skin={temp_skin_val:.2f}°C, T_amb={temp_amb_val:.2f}°C => T_core={temp_val:.2f}°C, T_burton={t_burton_calc:.2f}°C")

            # Parameter vital sign lainnya
            if spo2 <= 0:
                input_warnings.append("spo2 memakai nilai fallback 98")
            if resp_rate <= 0:
                input_warnings.append("respiratory_rate memakai nilai fallback 16")
            if hr_ecg <= 0:
                input_warnings.append("heart_rate memakai nilai fallback 75")
            if patient.get('systolic') is None:
                input_warnings.append("systolic_bp memakai nilai fallback 120")
            if patient.get('diastolic') is None:
                input_warnings.append("diastolic_bp memakai nilai fallback 80")
            if patient.get('gcs') is None:
                input_warnings.append("gcs_total memakai nilai fallback 15")

            spo2_val = float(spo2 if spo2 > 0 else 98.0)
            rr_val = float(resp_rate if resp_rate > 0 else 16.0)
            hr_val = float(hr_ecg if hr_ecg > 0 else 75.0)
            sys_val = float(patient.get('systolic') if patient.get('systolic') is not None else 120)
            dia_val = float(patient.get('diastolic') if patient.get('diastolic') is not None else 80)
            gcs_val = float(patient.get('gcs') if patient.get('gcs') is not None else 15)

            # Susun tujuh tanda vital mentah. Adapter ONNX melakukan clipping
            # dan membangun sepuluh fitur turunan sesuai notebook training.
            raw_data = {
                'temperature_c': temp_val,
                'spo2': spo2_val,
                'respiratory_rate': rr_val,
                'heart_rate': hr_val,
                'systolic_bp': sys_val,
                'diastolic_bp': dia_val,
                'gcs_total': gcs_val,
            }

            triage_label = "TIDAK TERSEDIA"
            triage_valid = False
            triage_error = self.triage_model_error or "Model ONNX belum dimuat."
            triage_score = 0.0
            triage_probabilities = []
            triage_features = {}
            inference_ms = None

            if self.triage_predictor is not None:
                try:
                    prediction = self.triage_predictor.predict(raw_data)
                    triage_label = prediction.label
                    triage_valid = True
                    triage_error = None
                    triage_score = prediction.confidence
                    triage_probabilities = list(prediction.probabilities)
                    triage_features = prediction.feature_values
                    inference_ms = prediction.inference_ms
                    print(
                        f"[ONNX] {prediction.label} "
                        f"(confidence={prediction.confidence:.4f}, "
                        f"inference={prediction.inference_ms:.3f} ms)"
                    )
                except (TriagePreprocessingError, TriageModelError) as exc:
                    triage_error = str(exc)
                    print(f"[ERROR ONNX] {triage_error}")
                except Exception as exc:
                    triage_error = f"Kesalahan tak terduga saat inferensi ONNX: {exc}"
                    print(f"[ERROR ONNX] {triage_error}")

            input_quality = "measured" if not input_warnings else "fallback"
            if input_warnings:
                print("[WARN ONNX INPUT] " + "; ".join(input_warnings))

            # -----------------------------------------------------------------
            # TAHAP 5: Konsolidasi Seluruh Data ke Dictionary RAM (100%)
            # -----------------------------------------------------------------
            self.status_updated.emit("Pemrosesan Data Selesai!", 100)
            self.msleep(150)

            results = {
                # Metadata & Registrasi
                "bed": patient.get("bed", "00"),
                "gcs": gcs_val,
                "timestamp": patient.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),

                # Parameter Medis Suhu (Direct, Core Estimate, & Burton)
                "temperature": temp_val,                  # T_core (digunakan oleh ML Model & Display Utama)
                "temp_skin": round(temp_skin_val, 1),     # Suhu kulit/object dari sensor
                "temp_ambient": round(temp_amb_val, 1),   # Suhu lingkungan dari sensor
                "temp_burton": round(t_burton_calc, 1),   # Suhu Rata-rata Burton (Tb)

                # Parameter Medis Lainnya
                "hr": hr_val,
                "rr": rr_val,
                "spo2": spo2_val,
                "systolic": sys_val,
                "diastolic": dia_val,
                "pi_red": pi_red,
                "pi_ir": pi_ir,
                "ppg_hr": ppg_hr,

                # Output model triase ONNX
                "triage_status": triage_label,
                "triage_valid": triage_valid,
                "triage_error": triage_error,
                "triage_input_quality": input_quality,
                "triage_input_warnings": input_warnings,
                "triage_probabilities": triage_probabilities,
                "triage_features": triage_features,
                "xgboost_score": triage_score,
                "model_backend": "onnxruntime",
                "model_inference_ms": inference_ms,
                # SHAP model lama sengaja tidak digunakan untuk menjelaskan model baru.
                "shap_features": [],
                "shap_values": [],

                # Sinyal Raw & Filtered
                "raw_time": self.raw_time,
                "raw_ecg": self.raw_ecg,
                "raw_red": self.raw_red,
                "raw_ir": self.raw_ir,
                "time_125": time_125,
                "ecg_smooth": ecg_smooth,
                "red_clean": red_clean,
                "ir_clean": ir_clean
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

        # 2. Isi Progress Bar
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
    processing_finished = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._status_effect = None
        self._fade_out_anim = None
        self._fade_in_anim = None
        self.worker = None
        self.triage_predictor = None
        self.triage_model_error = None
        self.setup_ui()
        self._load_triage_model()

    def _load_triage_model(self):
        """Muat dan warm-up sesi ONNX sekali selama umur aplikasi."""
        try:
            self.triage_predictor = TriageOnnxPredictor(warm_up=True)
            print(
                "[SUCCESS ONNX] Model triase siap "
                f"dalam {self.triage_predictor.load_ms:.2f} ms."
            )
        except TriageModelError as exc:
            self.triage_model_error = str(exc)
            print(f"[ERROR ONNX] {self.triage_model_error}")

    def setup_ui(self):
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background-color: #F6FFEC;")
        
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setContentsMargins(32, 24, 32, 24)

        # 1. Logo TriaGO
        self.lbl_logo = QLabel()
        self.lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_logo.setStyleSheet("background: transparent; margin-bottom: 10px;")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.abspath(os.path.join(current_dir, "..", "asset", "logo.png"))
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            self.lbl_logo.setPixmap(pixmap.scaledToWidth(340, Qt.TransformationMode.SmoothTransformation))
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
        card_container.setFixedWidth(540)
        card_container.setFixedHeight(128)

        card_layout = QVBoxLayout(card_container)
        card_layout.setContentsMargins(30, 22, 30, 22)
        card_layout.setSpacing(10)

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
        patient_info=None,
        fs_orig=400,
    ):
        """Memicu pemrosesan sinyal dan klasifikasi ML."""
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Memulai pemrosesan data...")

        self.worker = ProcessingWorker(
            raw_ecg=raw_ecg,
            raw_time=raw_time,
            raw_red=raw_red,
            raw_ir=raw_ir,
            patient_info=patient_info,
            fs_orig=fs_orig,
            triage_predictor=self.triage_predictor,
            triage_model_error=self.triage_model_error,
        )
        self.worker.status_updated.connect(self.update_ui_state)
        self.worker.processing_finished.connect(self.handle_processing_completion)
        self.worker.start()

    def handle_processing_completion(self, results):
        """Dipanggil otomatis ketika pemrosesan data selesai."""
        print("[LOG] Pemrosesan data selesai!")
        self.lbl_status.setText("Pemrosesan Data Selesai!")

        # 1. Generate string timestamp saat ini (Format: TahunBulanTanggal_JamMenitDetik)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 2. Tentukan folder penyimpanan tetap 'data_pengukuran'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.abspath(os.path.join(current_dir, "..", "data_pengukuran"))
        os.makedirs(folder_path, exist_ok=True)  # Membuat folder otomatis jika belum ada

        # 3. Buat path nama file lengkap di dalam folder data_pengukuran
        filename = os.path.join(folder_path, f"ekstraksi_data_{timestamp}.csv")

        # 4. Menyusun data parameter vital sign
        summary_data = {
            "HR_ECG_BPM": [results['hr']],
            "RR_RPM": [results['rr']],
            "SpO2_Percent": [results['spo2']],
            "PI_Red_Percent": [results['pi_red']],
            "PI_IR_Percent": [results['pi_ir']],
            "HR_PPG_BPM": [results['ppg_hr']]
        }
        df_summary = pd.DataFrame(summary_data)
        
        # 5. Menyimpan file ke folder data_pengukuran
        df_summary.to_csv(filename, index=False)
        print(f"[LOG] Data ringkasan berhasil disimpan ke: {filename}")

        # 6. Alirkan hasil ke Window Utama
        if hasattr(self, "parent_main_win"):
            self.parent_main_win.processed_results = results
            self.parent_main_win.handle_output_phase(results)
        else:
            self.lbl_status.setText("Selesai!!!")

    def update_ui_state(self, text, progress_value):
        """Sinkronisasi progress bar dan transisi teks status."""
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
        if self.worker is not None and self.worker.isRunning():
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

    ecg_dummy = (
        np.sin(2 * np.pi * 1.25 * t_dummy)
        + 0.5 * np.sin(2 * np.pi * 50 * t_dummy)
        + np.random.normal(0, 0.1, len(t_dummy))
    )
    ppg_red_dummy = 20000 + 500 * np.sin(2 * np.pi * 1.25 * t_dummy) + np.random.normal(0, 20, len(t_dummy))
    ppg_ir_dummy = 30000 + 600 * np.sin(2 * np.pi * 1.25 * t_dummy) + np.random.normal(0, 20, len(t_dummy))

    dummy_patient = {
        "bed": "03",
        "gcs": 14,
        "temperature": 36.8,
        "systolic": 125,
        "diastolic": 82
    }

    test_loader.start_processing(
        raw_ecg=ecg_dummy,
        raw_time=t_dummy,
        raw_red=ppg_red_dummy,
        raw_ir=ppg_ir_dummy,
        patient_info=dummy_patient,
        fs_orig=fs_dummy,
    )

    sys.exit(app.exec())
