from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QGraphicsOpacityEffect
)
from PyQt6.QtCore import (
    QThread, pyqtSignal, Qt, QPropertyAnimation, QEasingCurve,
    pyqtProperty, QRectF
)
from PyQt6.QtGui import QPainter, QColor, QLinearGradient, QPen, QPainterPath, QFont


class ProcessingWorker(QThread):
    """Worker thread untuk komputasi berat di latar belakang."""
    status_updated = pyqtSignal(str, int)
    processing_finished = pyqtSignal(dict)

    def __init__(self, csv_filepath):
        super().__init__()
        self.csv_filepath = csv_filepath

    def run(self):
        stages = [
            ("Menginisialisasi berkas data rekaman...", 5),
            ("Mengekstrak matriks sinyal mentah (24.000 sampel)...", 15),
            ("Menjalankan digital filtering pada gelombang ECG...", 25),
            ("Menjalankan digital filtering pada gelombang PPG...", 35),
            ("Mendeteksi puncak R-Peak ECG & perhitungan HRV...", 45),
            ("Mengekstraksi fitur amplitudo & durasi PPG...", 55),
            ("Mengumpankan fitur ke Model Deep Learning Tekanan Darah...", 70),
            ("Mengompilasi data GCS, Suhu, dan Parameter Hemodinamik...", 85),
            ("Menjalankan Model Machine Learning Klasifikasi Triase...", 95),
            ("Sinkronisasi hasil akhir TriaGO...", 100)
        ]

        total_duration = 5.0
        steps = 100
        interval = total_duration / steps

        current_step = 0
        stage_index = 0
        status_text = "Memulai komputasi data..."

        while current_step <= steps:
            self.msleep(int(interval * 1000))
            current_step += 1

            if stage_index < len(stages) and current_step >= stages[stage_index][1]:
                status_text = stages[stage_index][0]
                stage_index += 1

            self.status_updated.emit(status_text, current_step)

        dummy_results = {
            "systolic": 120,
            "diastolic": 80,
            "heart_rate": 78,
            "spo2": 98,
            "triage_status": "MILD",
            "csv_path": self.csv_filepath
        }
        self.processing_finished.emit(dummy_results)


class AnimatedProgressBar(QWidget):
    """
    Progress bar yang digambar manual dengan QPainter.
    Kelebihan dibanding stylesheet QProgressBar bawaan:
    - Ujung kapsul SELALU melengkung sempurna, berapapun persentasenya
      (stylesheet Qt gagal melengkungkan ujung chunk saat chunk masih sempit).
    - Nilai berubah dengan animasi halus (bukan lompat langsung).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._radius = 11
        self.setFixedHeight(22)

        self._animation = QPropertyAnimation(self, b"value")
        self._animation.setDuration(350)
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
        radius = self._radius

        # --- Track (latar belakang kapsul) ---
        track_path = QPainterPath()
        track_path.addRoundedRect(rect, radius, radius)
        painter.setPen(QPen(QColor("#444444"), 1))
        painter.setBrush(QColor("#2C2C2C"))
        painter.drawPath(track_path)

        # --- Chunk terisi (digambar dengan clip path miliknya sendiri
        #     agar kedua ujungnya selalu melengkung, tidak terpotong kotak) ---
        full_width = rect.width()
        chunk_width = full_width * (self._value / 100.0)

        if chunk_width > 0.5:
            chunk_rect = QRectF(rect.x(), rect.y(), chunk_width, rect.height())
            chunk_path = QPainterPath()
            chunk_path.addRoundedRect(chunk_rect, radius, radius)

            painter.save()
            painter.setClipPath(chunk_path)

            gradient = QLinearGradient(rect.x(), 0, rect.x() + full_width, 0)
            gradient.setColorAt(0.0, QColor("#3498DB"))
            gradient.setColorAt(1.0, QColor("#2ECC71"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawRect(chunk_rect)
            painter.restore()

        # --- Teks persentase ---
        painter.setPen(QColor("white"))
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{int(round(self._value))}%")


class LoadingPage(QWidget):
    def __init__(self):
        super().__init__()
        self._status_effect = None
        self._fade_out_anim = None
        self._fade_in_anim = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(50, 50, 50, 50)

        card_container = QFrame()
        card_container.setStyleSheet(
            "background-color: #1E1E1E; border-radius: 12px; padding: 40px;"
        )
        card_container.setFixedWidth(650)

        card_layout = QVBoxLayout(card_container)
        card_layout.setSpacing(30)

        self.lbl_title = QLabel("PROSES ANALISIS MEDIS")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #3498DB; letter-spacing: 2px;"
        )
        card_layout.addWidget(self.lbl_title)

        # Progress bar custom-painted, menggantikan QProgressBar + stylesheet
        self.progress_bar = AnimatedProgressBar()
        card_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Mempersiapkan mesin komputasi...")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("""
            QLabel {
                font-size: 15px;
                color: #BDC3C7;
                font-style: italic;
                min-height: 50px;
                padding: 5px;
            }
        """)
        card_layout.addWidget(self.lbl_status)

        # Opacity effect dipasang sekali di awal, dipakai ulang untuk animasi fade
        self._status_effect = QGraphicsOpacityEffect(self.lbl_status)
        self._status_effect.setOpacity(1.0)
        self.lbl_status.setGraphicsEffect(self._status_effect)

        layout.addWidget(card_container)

    def start_processing(self, csv_filepath):
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Mempersiapkan mesin komputasi...")

        self.worker = ProcessingWorker(csv_filepath)
        self.worker.status_updated.connect(self.update_ui_state)
        self.worker.processing_finished.connect(self.handle_completion)
        self.worker.start()

    def update_ui_state(self, text, progress_value):
        self.progress_bar.animate_to(progress_value)
        if text != self.lbl_status.text():
            self._fade_to_text(text)

    def _fade_to_text(self, new_text):
        # Hentikan animasi sebelumnya jika masih berjalan
        if self._fade_out_anim is not None:
            self._fade_out_anim.stop()
        if self._fade_in_anim is not None:
            self._fade_in_anim.stop()

        self._fade_out_anim = QPropertyAnimation(self._status_effect, b"opacity")
        self._fade_out_anim.setDuration(150)
        self._fade_out_anim.setStartValue(1.0)
        self._fade_out_anim.setEndValue(0.0)
        self._fade_out_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        def _swap_and_fade_in():
            self.lbl_status.setText(new_text)
            self._fade_in_anim = QPropertyAnimation(self._status_effect, b"opacity")
            self._fade_in_anim.setDuration(200)
            self._fade_in_anim.setStartValue(0.0)
            self._fade_in_anim.setEndValue(1.0)
            self._fade_in_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            self._fade_in_anim.start()

        self._fade_out_anim.finished.connect(_swap_and_fade_in)
        self._fade_out_anim.start()

    def handle_completion(self, results):
        if hasattr(self, 'parent_main_win'):
            self.parent_main_win.handle_output_phase(results)

    def close_threads(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()