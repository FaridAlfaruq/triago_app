# -*- coding: utf-8 -*-
"""GUI akuisisi data STM32 untuk pembuatan dataset kalibrasi.

Menampilkan ECG dan PPG-IR terfilter secara real-time, merekam paket mentah,
serta menyimpan ground truth tanda vital pada setiap baris CSV.
"""

import csv
import os
import sys
import time
from collections import deque

import pyqtgraph as pg
from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# Import custom modules dari root project.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from akusisi_data.get_stm32 import stream_stm32_data
from processing_data.preprocessing_LiveData import LiveSignalFilter


# Render cepat untuk stream 400 Hz. Grafik tetap digambar oleh timer 30 FPS.
pg.setConfigOptions(antialias=False)
pg.setConfigOption("background", "w")


class STM32Worker(QThread):
    """Membaca stream STM32 di thread terpisah agar GUI tidak membeku."""

    data_received = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        for packet in stream_stm32_data():
            if not self.running or self.isInterruptionRequested():
                break
            if packet.get("status") == "OK":
                self.data_received.emit(packet)

    def stop(self):
        self.running = False
        self.requestInterruption()
        self.wait()


class MainWindow(QMainWindow):
    SAMPLE_RATE_HZ = 400
    WARMUP_DURATION_SEC = 3.0
    RECORD_DURATION_SEC = 60.0
    PLOT_WINDOW_SEC = 5.0

    # key: (label, unit, placeholder, minimum, maximum)
    GROUND_TRUTH_FIELDS = {
        "spo2": ("SpO₂", "%", "98", 50.0, 100.0),
        "sbp": ("SBP", "mmHg", "120", 40.0, 300.0),
        "dbp": ("DBP", "mmHg", "80", 20.0, 200.0),
        "body_temp": ("Suhu Tubuh", "°C", "36.8", 25.0, 45.0),
        "respiratory_rate": ("Respiratory Rate", "rpm", "16", 3.0, 80.0),
        "heart_rate": ("Heart Rate", "bpm", "75", 20.0, 250.0),
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("STM32 Bio-Signal Data Acquisition for Calibration")
        self.resize(1180, 880)

        self.total_target_samples = int(
            (self.WARMUP_DURATION_SEC + self.RECORD_DURATION_SEC)
            * self.SAMPLE_RATE_HZ
        )

        self.is_recording = False
        self.recorded_data = []
        self.max_plot_points = int(self.PLOT_WINDOW_SEC * self.SAMPLE_RATE_HZ)
        self.live_filter = LiveSignalFilter()

        # Ring buffer menjaga biaya update tetap konstan.
        self.time_buffer = deque(maxlen=self.max_plot_points)
        self.ecg_buffer = deque(maxlen=self.max_plot_points)
        self.ppg_ir_buffer = deque(maxlen=self.max_plot_points)

        # State ringan yang dirender oleh timer 30 FPS.
        self.latest_temp_obj = 0.0
        self.latest_temp_amb = 0.0
        self.latest_sample_count = 0
        self.latest_elapsed_sec = 0.0
        self.latest_stream_hz = 0.0
        self.current_progress_val = 0
        self.current_status_text = "Ready for recording..."
        self.current_status_style = (
            "font-size: 14px; font-weight: bold; color: #0055ff;"
        )
        self.is_warmup_phase = False

        self._rate_packet_count = 0
        self._rate_window_started = time.monotonic()

        self.init_ui()
        self.start_worker_thread()

        self.render_timer = QTimer(self)
        self.render_timer.setInterval(33)  # sekitar 30 FPS
        self.render_timer.timeout.connect(self.update_ui_render)
        self.render_timer.start()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(10)

        self.status_label = QLabel("Ready for recording...")
        self.status_label.setStyleSheet(self.current_status_style)
        self.lbl_graph_hint = QLabel(
            "Grafik menampilkan rolling window 5 detik sejak sampel pertama, "
            "termasuk fase warmup."
        )
        self.lbl_graph_hint.setStyleSheet(
            "font-size: 12px; color: #777777; font-style: italic;"
        )
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.lbl_graph_hint)

        # Nilai sensor dan statistik stream real-time.
        meta_layout = QHBoxLayout()
        self.lbl_temp_obj = QLabel("Body Temp: -- °C")
        self.lbl_temp_amb = QLabel("Ambient Temp: -- °C")
        self.lbl_sample_rate = QLabel("Stream: -- Hz")
        self.lbl_sample_count = QLabel("Samples: 0")
        self.lbl_elapsed = QLabel("Elapsed: 0.0 s")

        for label in (self.lbl_temp_obj, self.lbl_temp_amb):
            label.setStyleSheet(
                "font-size: 13px; font-weight: bold; color: #555555;"
            )
        for label in (
            self.lbl_sample_rate,
            self.lbl_sample_count,
            self.lbl_elapsed,
        ):
            label.setStyleSheet(
                "font-size: 13px; font-weight: bold; color: #214889;"
            )

        meta_layout.addWidget(self.lbl_temp_obj)
        meta_layout.addWidget(self.lbl_temp_amb)
        meta_layout.addStretch()
        meta_layout.addWidget(self.lbl_sample_rate)
        meta_layout.addWidget(self.lbl_sample_count)
        meta_layout.addWidget(self.lbl_elapsed)
        main_layout.addLayout(meta_layout)

        # Ground truth dapat diisi sebelum merekam atau setelah rekaman selesai.
        gt_group = QGroupBox("Ground Truth / Nilai Referensi")
        gt_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #214889; }"
        )
        gt_layout = QGridLayout(gt_group)
        gt_layout.setHorizontalSpacing(12)
        gt_layout.setVerticalSpacing(8)
        self.ground_truth_inputs = {}

        for index, (key, spec) in enumerate(self.GROUND_TRUTH_FIELDS.items()):
            display_name, unit, example, _, _ = spec
            row = index // 3
            column = (index % 3) * 2
            label = QLabel(f"{display_name} ({unit})")
            field = QLineEdit()
            field.setPlaceholderText(f"contoh: {example}")
            field.setMaximumWidth(150)
            field.returnPressed.connect(self.save_to_csv)
            self.ground_truth_inputs[key] = field
            gt_layout.addWidget(label, row, column)
            gt_layout.addWidget(field, row, column + 1)

        main_layout.addWidget(gt_group)

        # Alias kompatibilitas untuk kode lama yang mungkin mengakses atribut ini.
        self.input_gt_spo2 = self.ground_truth_inputs["spo2"]
        self.input_gt_hr = self.ground_truth_inputs["heart_rate"]

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground("#FFFFFF")
        main_layout.addWidget(self.win, stretch=1)

        self.p1 = self.win.addPlot(title="Sinyal ECG Real-Time (Filtered)")
        self._configure_plot(self.p1, "Amplitudo ECG")
        self.ecg_curve = self.p1.plot(pen=pg.mkPen("#214889", width=1.8))

        self.win.nextRow()

        self.p2 = self.win.addPlot(title="Sinyal PPG-IR Real-Time (Filtered)")
        self._configure_plot(self.p2, "Amplitudo PPG")
        self.ppg_curve = self.p2.plot(pen=pg.mkPen("#D35400", width=1.8))
        self.p2.setXLink(self.p1)

        button_layout = QHBoxLayout()
        self.btn_start = QPushButton(
            f"Start Recording ({self.RECORD_DURATION_SEC:g}s + "
            f"{self.WARMUP_DURATION_SEC:g}s warmup)"
        )
        self.btn_start.setStyleSheet(
            "background-color: #2ea44f; color: white; font-weight: bold; "
            "padding: 10px; border-radius: 4px;"
        )
        self.btn_start.clicked.connect(self.start_recording)

        self.btn_save = QPushButton("Simpan CSV")
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(
            "background-color: #214889; color: white; font-weight: bold; "
            "padding: 10px; border-radius: 4px;"
        )
        self.btn_save.clicked.connect(self.save_to_csv)

        self.btn_reset = QPushButton("Ulangi (Reset)")
        self.btn_reset.setStyleSheet(
            "background-color: #cb2431; color: white; font-weight: bold; "
            "padding: 10px; border-radius: 4px;"
        )
        self.btn_reset.clicked.connect(self.reset_recording)

        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_save)
        button_layout.addWidget(self.btn_reset)
        main_layout.addLayout(button_layout)

    @staticmethod
    def _configure_plot(plot, y_label):
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setClipToView(True)
        plot.setDownsampling(auto=True, mode="peak")
        plot.setLabel(
            "bottom", "Waktu sejak mulai sesi", units="s", color="#214889"
        )
        plot.setLabel("left", y_label, color="#214889")
        plot.getAxis("left").setPen("#214889")
        plot.getAxis("bottom").setPen("#214889")
        plot.getAxis("left").setTextPen("#214889")
        plot.getAxis("bottom").setTextPen("#214889")
        plot.getViewBox().enableAutoRange(axis="y")

    def start_worker_thread(self):
        self.worker = STM32Worker()
        self.worker.data_received.connect(self.handle_new_packet)
        self.worker.start()

    def handle_new_packet(self, packet):
        """Update buffer pada setiap paket; rendering tetap dibatasi 30 FPS."""
        ecg_val = packet["ecg"]
        ppg_ir_val = packet["ppg"]["ir"]

        temperature = packet.get("temperature", {})
        self.latest_temp_obj = float(temperature.get("object", 0.0))
        self.latest_temp_amb = float(temperature.get("ambient", 0.0))

        clean_ecg = self.live_filter.filter_ecg(ecg_val)
        clean_ppg = self.live_filter.filter_ppg(ppg_ir_val)

        # Estimasi sample rate aktual membantu mendeteksi packet drop/serial lag.
        self._rate_packet_count += 1
        rate_elapsed = time.monotonic() - self._rate_window_started
        if rate_elapsed >= 1.0:
            self.latest_stream_hz = self._rate_packet_count / rate_elapsed
            self._rate_packet_count = 0
            self._rate_window_started = time.monotonic()

        if not self.is_recording:
            return

        self.recorded_data.append(packet)
        current_samples_count = len(self.recorded_data)
        session_elapsed = (current_samples_count - 1) / self.SAMPLE_RATE_HZ
        warmup_samples = int(self.WARMUP_DURATION_SEC * self.SAMPLE_RATE_HZ)

        self.latest_sample_count = current_samples_count
        self.latest_elapsed_sec = session_elapsed

        # Grafik aktif sejak sampel pertama, termasuk selama fase warmup.
        self.time_buffer.append(session_elapsed)
        self.ecg_buffer.append(clean_ecg)
        self.ppg_ir_buffer.append(clean_ppg)

        if current_samples_count <= warmup_samples:
            self.is_warmup_phase = True
            self.current_progress_val = 0
            remaining = max(0.0, self.WARMUP_DURATION_SEC - session_elapsed)
            self.current_status_text = (
                f"STABILIZING SENSOR... {remaining:.1f}s remaining"
            )
            self.current_status_style = (
                "font-size: 14px; font-weight: bold; color: #d97706;"
            )
            return

        self.is_warmup_phase = False
        recorded_duration = (
            current_samples_count - warmup_samples
        ) / self.SAMPLE_RATE_HZ
        self.current_progress_val = min(
            100,
            int((recorded_duration / self.RECORD_DURATION_SEC) * 100),
        )
        self.current_status_text = (
            f"RECORDING ONGOING... {recorded_duration:.1f}/"
            f"{self.RECORD_DURATION_SEC:g}s"
        )
        self.current_status_style = (
            "font-size: 14px; font-weight: bold; color: #cb2431;"
        )

        if current_samples_count >= self.total_target_samples:
            self.stop_and_save_data()

    def update_ui_render(self):
        """Menggambar grafik dan teks maksimal sekitar 30 kali per detik."""
        self.lbl_temp_obj.setText(f"Body Temp: {self.latest_temp_obj:.2f} °C")
        self.lbl_temp_amb.setText(f"Ambient Temp: {self.latest_temp_amb:.2f} °C")
        self.lbl_sample_rate.setText(f"Stream: {self.latest_stream_hz:.1f} Hz")
        self.lbl_sample_count.setText(f"Samples: {self.latest_sample_count}")
        self.lbl_elapsed.setText(f"Elapsed: {self.latest_elapsed_sec:.1f} s")

        temp_color = "#cb2431" if self.latest_temp_obj > 37.5 else "#2ea44f"
        self.lbl_temp_obj.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {temp_color};"
        )

        if self.is_recording:
            self.progress_bar.setValue(self.current_progress_val)
            self.status_label.setText(self.current_status_text)
            self.status_label.setStyleSheet(self.current_status_style)

        if self.time_buffer:
            t_values = list(self.time_buffer)
            self.ecg_curve.setData(t_values, list(self.ecg_buffer))
            self.ppg_curve.setData(t_values, list(self.ppg_ir_buffer))

            right_edge = max(self.PLOT_WINDOW_SEC, t_values[-1])
            left_edge = max(0.0, right_edge - self.PLOT_WINDOW_SEC)
            self.p1.setXRange(left_edge, right_edge, padding=0.0)

    def start_recording(self):
        if self.is_recording:
            return

        self.is_recording = True
        self.is_warmup_phase = True
        self.live_filter = LiveSignalFilter()
        self.recorded_data.clear()
        self.time_buffer.clear()
        self.ecg_buffer.clear()
        self.ppg_ir_buffer.clear()
        self.ecg_curve.clear()
        self.ppg_curve.clear()

        self.latest_sample_count = 0
        self.latest_elapsed_sec = 0.0
        self.current_progress_val = 0
        self.current_status_text = "STABILIZING SENSOR..."
        self.progress_bar.setValue(0)

        self.btn_start.setEnabled(False)
        self.btn_save.setEnabled(False)
        for field in self.ground_truth_inputs.values():
            field.setEnabled(False)

    def reset_recording(self):
        self.is_recording = False
        self.is_warmup_phase = False
        self.recorded_data.clear()
        self.time_buffer.clear()
        self.ecg_buffer.clear()
        self.ppg_ir_buffer.clear()
        self.ecg_curve.clear()
        self.ppg_curve.clear()

        self.latest_sample_count = 0
        self.latest_elapsed_sec = 0.0
        self.current_progress_val = 0
        self.progress_bar.setValue(0)

        self.btn_start.setEnabled(True)
        self.btn_save.setEnabled(False)
        for field in self.ground_truth_inputs.values():
            field.clear()
            field.setEnabled(True)

        self.status_label.setText("Recording reset. Ready to start again.")
        self.status_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #0055ff;"
        )

    def stop_and_save_data(self):
        """Mengakhiri rekaman dan membuka input ground truth untuk disimpan."""
        self.is_recording = False
        self.is_warmup_phase = False
        self.current_progress_val = 100
        self.progress_bar.setValue(100)
        self.status_label.setText(
            "Recording selesai. Lengkapi seluruh ground truth lalu simpan CSV."
        )
        self.status_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #2ea44f;"
        )

        self.btn_save.setEnabled(True)
        for field in self.ground_truth_inputs.values():
            field.setEnabled(True)
        self.input_gt_spo2.setFocus()

    def generate_next_filename(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "data_primer")
        os.makedirs(output_dir, exist_ok=True)
        index = 1
        while True:
            filepath = os.path.join(output_dir, f"Data{index}.csv")
            if not os.path.exists(filepath):
                return filepath
            index += 1

    def _read_ground_truth(self):
        """Validasi seluruh nilai referensi dan kembalikan dictionary float."""
        values = {}

        for key, field in self.ground_truth_inputs.items():
            display_name, unit, _, minimum, maximum = self.GROUND_TRUTH_FIELDS[key]
            text_value = field.text().strip().replace(",", ".")

            if not text_value:
                QMessageBox.warning(
                    self,
                    "Input Ground Truth Belum Lengkap",
                    f"Mohon masukkan {display_name} ({unit}).",
                )
                field.setFocus()
                return None

            try:
                value = float(text_value)
            except ValueError:
                QMessageBox.critical(
                    self,
                    "Input Tidak Valid",
                    f"Nilai {display_name} harus berupa angka.",
                )
                field.setFocus()
                return None

            if not minimum <= value <= maximum:
                QMessageBox.warning(
                    self,
                    "Nilai di Luar Rentang",
                    f"{display_name} harus berada pada rentang "
                    f"{minimum:g}–{maximum:g} {unit}.",
                )
                field.setFocus()
                return None

            values[key] = value

        if values["dbp"] >= values["sbp"]:
            QMessageBox.warning(
                self,
                "Tekanan Darah Tidak Valid",
                "Nilai DBP harus lebih rendah daripada SBP.",
            )
            self.ground_truth_inputs["dbp"].setFocus()
            return None

        return values

    def save_to_csv(self):
        if self.is_recording or not self.recorded_data:
            QMessageBox.warning(
                self,
                "Belum Siap",
                "Selesaikan perekaman sebelum menyimpan data.",
            )
            return

        ground_truth = self._read_ground_truth()
        if ground_truth is None:
            return

        filename = self.generate_next_filename()
        sampling_interval = 1.0 / self.SAMPLE_RATE_HZ

        try:
            with open(filename, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Time (s)",
                        "PPG_Red",
                        "PPG_IR",
                        "PPG_Green",
                        "ECG",
                        "Temp_Ambient",
                        "Temp_Object",
                        "SpO2_Ground_Truth",
                        "SBP_Ground_Truth",
                        "DBP_Ground_Truth",
                        "Body_Temperature_Ground_Truth",
                        "Respiratory_Rate_Ground_Truth",
                        "HR_Ground_Truth",
                    ]
                )

                warmup_samples = int(self.WARMUP_DURATION_SEC * self.SAMPLE_RATE_HZ)
                clean_recording = self.recorded_data[warmup_samples:] if len(self.recorded_data) > warmup_samples else self.recorded_data

                for index, packet in enumerate(clean_recording):
                    relative_time_s = index * sampling_interval
                    writer.writerow(
                        [
                            f"{relative_time_s:.4f}",
                            packet["ppg"]["red"],
                            packet["ppg"]["ir"],
                            packet["ppg"]["green"],
                            packet["ecg"],
                            packet["temperature"]["ambient"],
                            packet["temperature"]["object"],
                            ground_truth["spo2"],
                            ground_truth["sbp"],
                            ground_truth["dbp"],
                            ground_truth["body_temp"],
                            ground_truth["respiratory_rate"],
                            ground_truth["heart_rate"],
                        ]
                    )

            QMessageBox.information(
                self, "Success", f"Data berhasil disimpan ke {filename}"
            )
            self.status_label.setText(f"Finished & Saved to {filename}")
            self.btn_start.setEnabled(True)
            self.btn_save.setEnabled(False)
            for field in self.ground_truth_inputs.values():
                field.setEnabled(False)

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Save Error",
                f"Gagal menyimpan file CSV: {exc}",
            )

    def closeEvent(self, event):
        self.is_recording = False
        if hasattr(self, "render_timer"):
            self.render_timer.stop()
        if hasattr(self, "worker"):
            self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
