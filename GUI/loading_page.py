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

# Import modul pemrosesan ECG, PPG, Model ONNX Triase, dan Deep Learning BPNet
from processing_data.processing_data import ECGProcessor, PPGProcessor
from model.deployment_inference import TriageOnnxModel
from model.bpnet_inference import RevisedBPPredictor, preprocess_revised_bp
from model.temperature_estimation import estimate_core_temperature, ir_perfusion_percent

AUTOMATIC_TRIAGE_VALIDATED = True


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
        bpnet_predictor=None,
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
        self.bpnet_predictor = bpnet_predictor
        
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
            # TAHAP 2: Ekstraksi Fitur ECG (R-Peak & Heart Rate) (25% - 50%)
            # -----------------------------------------------------------------
            self.status_updated.emit("Mendeteksi R-Peak & Heart Rate ECG...", 40)
            self.msleep(100)

            r_peaks, noise_peaks = self.ecg_processor.detect_r_peaks(ecg_125, fs=125)
            hr_ecg = self.ecg_processor.calculate_heart_rate(r_peaks, fs=125)

            # -----------------------------------------------------------------
            # TAHAP 3: Pemrosesan Sinyal PPG & Multimodal RR Fusion (50% - 75%)
            # -----------------------------------------------------------------
            self.status_updated.emit("Menjalankan Pemrosesan Sinyal PPG (SpO2 & PI)...", 60)
            self.msleep(100)

            if (self.raw_red is not None and self.raw_ir is not None and len(self.raw_red) >= self.fs_orig * 2
                    and np.isfinite(self.raw_red).all() and np.isfinite(self.raw_ir).all()):
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
                ir_peaks = ppg_results.get('ir_peaks')
                ppg_hr = ppg_results['ppg_hr']
            else:
                spo2 = 0.0
                pi_red, pi_ir, ppg_hr = 0.0, 0.0, 0.0
                red_clean, ir_clean = np.array([]), np.array([])
                ir_peaks = None

            # Estimasi Multimodal Respiratory Rate Smart Fusion (ECG + PPG, Birrenkott 2018)
            self.status_updated.emit("Estimasi Multimodal Respiratory Rate (ECG + PPG)...", 70)
            resp_rate, resp_signal, resp_peaks = (
                self.ecg_processor.calculate_respiration_rate(
                    ecg_smooth, r_peaks, fs=125,
                    ppg=ir_clean if len(ir_clean) > 0 else None,
                    ppg_peaks=ir_peaks
                )
            )
            rr_details = self.ecg_processor.last_respiration_details or {}
            rr_quality = float(rr_details.get("quality", 0.0))
            rr_mode = str(rr_details.get("mode", "UNKNOWN"))
            rr_measured = bool(resp_rate > 0)
            print(f"[LOG RR MULTIMODAL] Mode={rr_mode}, RR={resp_rate:.2f} bpm, Quality={rr_quality:.3f}")

            # -----------------------------------------------------------------
            # TAHAP 4: Machine Learning Triage & SHAP Analysis (75% - 95%)
            # -----------------------------------------------------------------
            self.status_updated.emit("Mengestimasi suhu dan tekanan darah...", 85)
            self.msleep(100)

            patient = self.patient_info or {}

            # -----------------------------------------------------------------
            # ESTIMASI SUHU TUBUH INTI (CBT / T_core)
            # -----------------------------------------------------------------
            skin_samples = np.asarray(patient.get("raw_temp_obj", []), dtype=float)
            ambient_samples = np.asarray(patient.get("raw_temp_amb", []), dtype=float)
            valid = np.isfinite(skin_samples) & np.isfinite(ambient_samples)
            temp_skin_val = float(np.mean(skin_samples[valid])) if np.any(valid) else None
            temp_amb_val = float(np.mean(ambient_samples[valid])) if np.any(valid) else None
            temp_pi_ir = ir_perfusion_percent(self.raw_ir, self.fs_orig)
            temp_val = estimate_core_temperature(temp_skin_val, temp_amb_val, temp_pi_ir)
            input_warnings = []

            # Parameter vital sign lainnya
            if spo2 <= 0:
                input_warnings.append("spo2 memakai nilai fallback 98")
            if resp_rate <= 0:
                input_warnings.append("respiratory_rate memakai nilai fallback 16")
            if hr_ecg <= 0:
                input_warnings.append("heart_rate memakai nilai fallback 75")
            if patient.get('gcs') is None:
                input_warnings.append("gcs_total memakai nilai fallback 15")
            if patient.get('pain_score') is None:
                input_warnings.append("pain_score memakai nilai fallback 0")

            spo2_val = float(spo2 if spo2 > 0 else 98.0)
            rr_val = float(resp_rate if resp_rate > 0 else 16.0)
            hr_val = float(hr_ecg if hr_ecg > 0 else 75.0)
            gcs_val = float(patient.get('gcs') if patient.get('gcs') is not None else 15)
            pain_val = float(patient.get('pain_score') if patient.get('pain_score') is not None else 0)


            # -----------------------------------------------------------------
            # TAHAP 3.5: SQA 10s Window (Stride 2s) & Deep Learning BPNet Inference
            # ECG + PPG Inframerah, Tanpa Kalibrasi Manset
            # -----------------------------------------------------------------
            bp_res = {}
            sqa_error = None

            if self.bpnet_predictor is None:
                sqa_error = "Model BP revisi tidak tersedia."

            elif (
                self.raw_ecg is not None
                and self.raw_ir is not None
                and self.raw_time is not None
                and len(self.raw_time) >= 2
            ):
                try:
                    # Preprocessing menggunakan ECG dan PPG inframerah.
                    # Kedua sinyal diproses berdasarkan timestamp masing-masing.
                    bp_ecg, bp_ppg = preprocess_revised_bp(
                        self.raw_time,
                        self.raw_ecg,
                        self.raw_time,
                        self.raw_ir
                    )

                    # Inferensi BP tanpa referensi manset.
                    bp_res = self.bpnet_predictor.predict_recording(
                        bp_ecg,
                        bp_ppg
                    )

                except Exception as exc:
                    sqa_error = f"Gagal inferensi BP: {exc}"
                    print(f"[ERROR BPNET] {sqa_error}")

            else:
                sqa_error = "Sinyal ECG atau PPG inframerah tidak tersedia."

            # Ambil hasil estimasi tekanan darah dan evaluasi kualitas sinyal.
            sqa_passed = bp_res.get("sqa_passed", False)
            passed_segments = bp_res.get("passed_segments", 0)
            total_segments = bp_res.get("total_segments", 0)

            sys_val = bp_res.get("sbp")
            dia_val = bp_res.get("dbp")

            # Status yang diharapkan: uncalibrated apabila inferensi berhasil
            # tanpa referensi manset.
            bp_mode = bp_res.get("bp_mode", "unavailable")

            print(
                f"[BPNET] SBP={sys_val}, DBP={dia_val}, "
                f"Mode={bp_mode}, "
                f"SQA={passed_segments}/{total_segments}"
            )

            if sqa_error:
                print(f"[WARN BPNET] {sqa_error}")


            # Susun tanda vital mentah untuk ONNX XGBoost Triase (11 fitur)
            raw_data = {
                'temperature_c': temp_val,
                'spo2': spo2_val,
                'respiratory_rate': rr_val,
                'heart_rate': hr_val,
                'systolic_bp': sys_val,
                'diastolic_bp': dia_val,
                'gcs_total': gcs_val,
                'pain_score': pain_val,
            }

            triage_label = "TIDAK TERSEDIA"
            triage_valid = False
            triage_error = self.triage_model_error or "Model ONNX belum dimuat."
            triage_score = 0.0
            triage_probabilities = []
            triage_features = {}
            inference_ms = None

            if not AUTOMATIC_TRIAGE_VALIDATED:
                triage_error = "Triase otomatis menunggu validasi suhu, tekanan darah, dan Raspberry Pi."
            
            
            elif temp_val is None:
                triage_error = "Suhu tubuh tidak tersedia."
            elif self.triage_predictor is not None:
                try:
                    label, conf, proba = self.triage_predictor.predict(raw_data)
                    triage_label = label
                    triage_valid = True
                    triage_error = None
                    triage_score = conf
                    triage_probabilities = list(proba)

                    print(f"[ONNX] {label} (confidence={conf:.4f})")
                except Exception as exc:
                    triage_error = f"Kesalahan saat inferensi ONNX: {exc}"
                    print(f"[ERROR ONNX] {triage_error}")
            
            
            elif self.triage_predictor is not None:
                try:
                    label, conf, proba = self.triage_predictor.predict(raw_data)
                    triage_label = label
                    triage_valid = True
                    triage_error = None
                    triage_score = conf
                    triage_probabilities = list(proba)
                    print(f"[ONNX] {label} (confidence={conf:.4f})")
                except Exception as exc:
                    triage_error = f"Kesalahan saat inferensi ONNX: {exc}"
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
                "pain_score": pain_val,
                "timestamp": patient.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),

                # Parameter Medis Suhu (Estimasi Suhu Inti CBT)
                "temperature": temp_val,                  # T_core (digunakan oleh ML Model & Display Utama)
                "temp_skin": temp_skin_val,
                "temp_ambient": temp_amb_val,
                "temperature_pi_percent": temp_pi_ir,

                # Parameter Medis Lainnya
                "hr": hr_val,
                "rr": rr_val,
                "rr_measured": rr_measured,
                "rr_quality": rr_quality,
                "rr_mode": rr_mode,
                "spo2": spo2_val,
                "systolic": sys_val,
                "diastolic": dia_val,
                "bp_mode": bp_mode,
                "pi_red": pi_red,
                "pi_ir": pi_ir,
                "ppg_hr": ppg_hr,

                # Status & Evaluasi SQA
                "sqa_passed": sqa_passed,
                "sqa_error": sqa_error,
                "sqa_passed_segments": passed_segments,
                "sqa_rejected_segments": total_segments - passed_segments,
                "sqa_total_segments": total_segments,
                "sqa_rejections": bp_res.get("rejections", {}),

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
                "raw_temp_obj": patient.get("raw_temp_obj"),
                "raw_temp_amb": patient.get("raw_temp_amb"),
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
        self.bpnet_predictor = None
        self.triage_model_error = None
        self.setup_ui()
        self._load_triage_model()

    def _load_triage_model(self):
        """Muat model ONNX Triase dan Deep Learning BPNet."""
        try:
            self.triage_predictor = TriageOnnxModel()
            print("[SUCCESS ONNX] Model triase dari model/triage_xgboost_model.onnx berhasil dimuat.")
        except Exception as exc:
            self.triage_model_error = str(exc)
            print(f"[ERROR ONNX] {self.triage_model_error}")

        try:
            self.bpnet_predictor = RevisedBPPredictor()
            print(
                "[SUCCESS BPNET] Model Deep Learning BPNet & SQA Pipeline "
                f"berhasil dimuat via {self.bpnet_predictor.interpreter_backend}."
            )
        except Exception as exc:
            print(f"[WARN BPNET] Model BP revisi belum siap ({exc}).")

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
        self.card_container = QFrame()
        self.card_container.setStyleSheet("""
            QFrame {
                background-color: #214889; 
                border-radius: 28px; 
            }
        """)
        self.card_container.setFixedWidth(540)
        self.card_container.setFixedHeight(128)

        card_layout = QVBoxLayout(self.card_container)
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

        main_layout.addWidget(self.card_container)

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
        self.card_container.setStyleSheet("""
            QFrame {
                background-color: #214889; 
                border-radius: 28px; 
            }
        """)
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
            bpnet_predictor=self.bpnet_predictor,
        )
        self.worker.status_updated.connect(self.update_ui_state)
        self.worker.processing_finished.connect(self.handle_processing_completion)
        self.worker.start()

    def handle_processing_completion(self, results):
        """Dipanggil otomatis ketika pemrosesan data selesai."""
        print("[LOG] Pemrosesan data selesai!")
        self._fade_to_text("Pemrosesan Data Selesai!")

        # Alirkan hasil ke Window Utama untuk penyimpanan 2 file (CSV & JSON) dan pengalihan halaman
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
        "pain_score": 2,
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
