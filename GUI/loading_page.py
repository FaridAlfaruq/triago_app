import sys
import os
import numpy as np
import pandas as pd
import joblib
import shap
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


# =============================================================================
# HELPER: FUNGSI FEATURE ENGINEERING KLINIS (18 FITUR KLINIS)
# =============================================================================
def apply_feature_engineering(df_features):
    """Menghasilkan 11 fitur turunan klinis sesuai pipeline training Colab."""
    df = df_features.copy()

    # 1. Fitur Klinis Utama
    df['shock_index'] = df['heart_rate'] / (df['systolic_bp'] + 0.1)
    df['map'] = df['diastolic_bp'] + (1/3 * (df['systolic_bp'] - df['diastolic_bp']))
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']

    # 2. Fitur Indikator Kegawatan
    df['hypoxia'] = (df['spo2'] < 90).astype(int)
    df['tachypnea'] = (df['respiratory_rate'] > 24).astype(int)
    df['abnormal_temp'] = ((df['temperature_c'] >= 38.0) | (df['temperature_c'] <= 35.0)).astype(int)
    df['abnormal_hr'] = ((df['heart_rate'] > 100) | (df['heart_rate'] < 60)).astype(int)

    # Indikator GCS
    df['gcs_squared'] = df['gcs_total'] ** 2
    df['gcs_map_index'] = df['gcs_total'] * df['map']
    df['gcs_shock_index'] = df['shock_index'] / (df['gcs_total'] + 0.1)

    # 3. Total Akumulasi Kegawatan
    df['total_abnormal'] = (
        df['hypoxia'] + df['tachypnea'] + df['abnormal_temp'] + df['abnormal_hr']
    )

    return df


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
        parent=None,
    ):
        super().__init__(parent)
        self.raw_ecg = raw_ecg
        self.raw_time = raw_time
        self.raw_red = raw_red
        self.raw_ir = raw_ir
        self.patient_info = patient_info or {}
        self.fs_orig = fs_orig
        
        # Inisialisasi Processor ECG & PPG
        self.ecg_processor = ECGProcessor(target_fs=125)
        self.ppg_processor = PPGProcessor(target_fs=125)

        # Absolute Path File Model XGBoost
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.abspath(os.path.join(current_dir, "..", "ml_xgboost", "triage_model.joblib"))
        print(f"[INFO] Path Target: {self.model_path}")
        if not os.path.exists(self.model_path):
            print(f"[ERROR] ❌ File model tidak ditemukan pada path yang ditentukan!")
        else:
            print("[SUCCESS]  File model ditemukan.")
            
            try:
                # 2. Cek Memuat Model (Load Test)
                self.model = joblib.load(self.model_path)
                print("[SUCCESS]  Model berhasil dimuat ke memori (joblib.load beroperasi penuh).")
                print(f"[INFO] Tipe objek: {type(self.model).__name__}")
                
                # 3. Test Eksekusi (Dummy Inference)
                # Mendeteksi jumlah fitur yang dibutuhkan secara otomatis
                n_features = getattr(self.model, "n_features_in_", None)
                
                if n_features is not None:
                    print(f"[INFO] Fitur terdeteksi pada model: {n_features} input.")
                    dummy_input = np.zeros((1, n_features))
                    
                    # Jalankan prediksi uji coba
                    test_output = self.model.predict(dummy_input)
                    print(f"[SUCCESS]  Test eksekusi model berhasil!")
                    print(f"[INFO] Output uji coba: {test_output}")
                else:
                    print("[WARN] ⚠️ Jumlah fitur (n_features_in_) tidak terdeteksi otomatis. Melewati test dummy inference.")

            except ModuleNotFoundError as e:
                print(f"[ERROR] ❌ Library pendukung tidak ditemukan saat memuat model: {e}")
                print("[HINT] Pastikan versi `xgboost` dan `scikit-learn` pada environment runtime sesuai dengan environment saat training.")
            except Exception as e:
                print(f"[ERROR] ❌ Gagal membuka atau menjalankan model: {e}")

        print("="*50 + "\n")

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
        
        temp_val = float(patient.get('temperature', 36.5))
        spo2_val = float(spo2 if spo2 > 0 else 98.0)
        rr_val = float(resp_rate if resp_rate > 0 else 16.0)
        hr_val = float(hr_ecg if hr_ecg > 0 else 75.0)
        sys_val = float(patient.get('systolic', 120))
        dia_val = float(patient.get('diastolic', 80))
        gcs_val = float(patient.get('gcs', 15))

        # 1. Susun 7 Fitur Mentah
        raw_data = {
            'temperature_c': [temp_val],
            'spo2': [spo2_val],
            'respiratory_rate': [rr_val],
            'heart_rate': [hr_val],
            'systolic_bp': [sys_val],
            'diastolic_bp': [dia_val],
            'gcs_total': [gcs_val]
        }

        # 2. Jalankan Feature Engineering (18 Fitur)
        df_base = pd.DataFrame(raw_data)
        df_engineered = apply_feature_engineering(df_base)

        EXPECTED_FEATURES = [
            'temperature_c', 'spo2', 'respiratory_rate', 'heart_rate', 
            'systolic_bp', 'diastolic_bp', 'gcs_total', 'shock_index', 
            'map', 'pulse_pressure', 'hypoxia', 'tachypnea', 
            'abnormal_temp', 'abnormal_hr', 'gcs_squared', 
            'gcs_map_index', 'gcs_shock_index', 'total_abnormal'
        ]
        df_input = df_engineered[EXPECTED_FEATURES]

        triage_label = "NON-DARURAT"
        shap_features = EXPECTED_FEATURES
        shap_vals_sample = np.zeros(len(EXPECTED_FEATURES))

        # 3. Prediksi & SHAP Analysis via XGBoost
        if os.path.exists(self.model_path):
            try:
                pipeline = joblib.load(self.model_path)

                # Prediksi Kelas Triase
                pred_class = pipeline.predict(df_input)[0]
                class_mapping = {0: "RESUSITASI", 1: "DARURAT", 2: "NON-DARURAT"}
                triage_label = class_mapping.get(pred_class, "NON-DARURAT")

                # SHAP Value Calculation
                scaler = pipeline.named_steps['scaler']
                model = pipeline.named_steps['model']
                
                X_scaled = scaler.transform(df_input)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_scaled)

                # Penanganan Universal Array SHAP Multi-Class
                if isinstance(shap_values, list):
                    # Format list: [class_0_array, class_1_array, class_2_array]
                    shap_vals_sample = shap_values[pred_class][0]
                elif isinstance(shap_values, np.ndarray):
                    if shap_values.ndim == 3:
                        # Format 3D Array: (n_samples, n_features, n_classes) atau (n_classes, n_samples, n_features)
                        if shap_values.shape[0] == 3:  # Shape (3, 1, 18)
                            shap_vals_sample = shap_values[pred_class][0]
                        else:  # Shape (1, 18, 3)
                            shap_vals_sample = shap_values[0, :, pred_class]
                    elif shap_values.ndim == 2:
                        shap_vals_sample = shap_values[0]

            except Exception as e:
                print(f"[ERROR ML INFERENCE] Gagal memprediksi/menghitung SHAP: {e}")

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

            # Parameter Medis
            "temperature": temp_val,
            "hr": hr_val,
            "rr": rr_val,
            "spo2": spo2_val,
            "systolic": sys_val,
            "diastolic": dia_val,
            "pi_red": pi_red,
            "pi_ir": pi_ir,
            "ppg_hr": ppg_hr,

            # Output ML & SHAP
            "triage_status": triage_label,
            "shap_features": shap_features,
            "shap_values": shap_vals_sample,

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
        )
        self.worker.status_updated.connect(self.update_ui_state)
        self.worker.processing_finished.connect(self.handle_processing_completion)
        self.worker.start()

    def handle_processing_completion(self, results):
        """Dipanggil otomatis ketika pemrosesan data selesai."""
        print("[LOG] Pemrosesan data selesai!")
        self.lbl_status.setText("Pemrosesan Data Selesai!")

        # MENGHAPUS SIMPAN FILE LOKAL CSV
        # Langsung mengalirkan hasil ke Controller Utama (main_gui.py)
        if hasattr(self, "parent_main_win"):
            self.parent_main_win.handle_output_phase(results)

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