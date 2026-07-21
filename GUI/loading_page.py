import sys
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QApplication, QGraphicsOpacityEffect
)
from PyQt6.QtCore import (
    QThread, pyqtSignal, Qt, QPropertyAnimation, QEasingCurve,
    pyqtProperty, QRectF
)
from PyQt6.QtGui import QPainter, QColor, QPainterPath, QFont, QPixmap

# =====================================================================
# WORKER 1: Khusus Proses Stabilisasi Sensor Awal (Durasi 2 Detik)
# =====================================================================
class StabilizationWorker(QThread):
    """Worker thread untuk menghandle jeda transisi stabilisasi sensor selama 2 detik."""
    status_updated = pyqtSignal(str, int)
    stabilization_finished = pyqtSignal()

    def run(self):
        total_duration = 2.0  # Target waktu stabilisasi (2 detik)
        steps = 100
        interval = total_duration / steps  # 0.02 detik per step

        for current_step in range(1, steps + 1):
            self.msleep(int(interval * 1000))
            # Kirim sinyal update kemajuan nilai ke UI
            self.status_updated.emit("Menstabilkan sensor....", current_step)
            
        self.stabilization_finished.emit()


# =====================================================================
# CUSTOM PROGRESS BAR: Didesain Sesuai Komponen Figma
# =====================================================================
class AnimatedProgressBar(QWidget):
    """Progress bar kustom berbentuk pill putih dengan teks persentase biru tua."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self.setFixedHeight(34)  # Tinggi ideal sesuai proporsi komponen figma

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

        # 1. Gambar Track Latar Belakang (Transparan/Mengikuti warna kapsul induk)
        track_path = QPainterPath()
        track_path.addRoundedRect(rect, radius, radius)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(33, 72, 137, 50)) # Biru tua dengan opasitas rendah
        painter.drawPath(track_path)

        # 2. Gambar Isi Progress Bar (Warna Putih Solid sesuai Figma)
        full_width = rect.width()
        chunk_width = full_width * (self._value / 100.0)

        if chunk_width > rect.height(): # Pastikan path proporsional saat membulat
            chunk_rect = QRectF(rect.x(), rect.y(), chunk_width, rect.height())
            chunk_path = QPainterPath()
            chunk_path.addRoundedRect(chunk_rect, radius, radius)

            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#FFFFFF")) # Isi Putih
            painter.drawPath(chunk_path)
            painter.restore()
        elif chunk_width > 0:
            # Fallback bar awal saat progress masih sangat kecil
            painter.setBrush(QColor("#FFFFFF"))
            painter.drawEllipse(QRectF(rect.x(), rect.y(), rect.height(), rect.height()))

        # 3. Gambar Teks Persentase di Tengah (Warna Biru Tua #214889)
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
        # Set background utama hijau muda (#F6FFEC) global untuk page ini
        self.setStyleSheet("background-color: #F6FFEC;")
        
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setContentsMargins(50, 50, 50, 50)

        # 1. BAGIAN LOGO TRIAGO (Diletakkan di Atas Kapsul Kontainer)
        self.lbl_logo = QLabel()
        self.lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_logo.setStyleSheet("background: transparent; margin-bottom: 10px;")
        
        # Mengambil asset logo sesuai path lokal project TriaGO
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(current_dir, r"C:\Users\Adyty\Documents\Farid ITS\TriaGo\asset\logo.png") 
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            self.lbl_logo.setPixmap(pixmap.scaledToWidth(420, Qt.TransformationMode.SmoothTransformation))
        else:
            self.lbl_logo.setText("TriaGO")
            self.lbl_logo.setStyleSheet("font-size: 48px; font-weight: 900; color: #214889; background: transparent;")
        main_layout.addWidget(self.lbl_logo)

        # 2. KAPSUL KONTAINER UTAMA (Warna Biru Tua #214889 dengan Sudut Membulat Lebar)
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

        # 3. KELAS PROGRESS BAR KUSTOM
        self.progress_bar = AnimatedProgressBar()
        card_layout.addWidget(self.progress_bar)

        # 4. TEKS STATUS (Warna Putih, Italic, Center Aligned sesuai Figma)
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

        # Efek opasitas untuk animasi transisi teks halus
        self._status_effect = QGraphicsOpacityEffect(self.lbl_status)
        self._status_effect.setOpacity(1.0)
        self.lbl_status.setGraphicsEffect(self._status_effect)

        main_layout.addWidget(card_container)

    def start_stabilization(self):
        """Memicu fase pertama: Menunggu 2 detik untuk proses stabilisasi sensor."""
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Menstabilkan sensor....")

        self.worker = StabilizationWorker()
        self.worker.status_updated.connect(self.update_ui_state)
        self.worker.stabilization_finished.connect(self.handle_stabilization_completion)
        self.worker.start()

    def update_ui_state(self, text, progress_value):
        """Metode sinkronisasi nilai progress bar dan kelancaran transisi teks status."""
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

    def handle_stabilization_completion(self):
        """Fase stabilisasi selesai. Siap dialihkan ke halaman Live Data atau Loading Kuantisasi Berikutnya."""
        print("[LOG] Stabilisasi sensor selesai. Pindah ke halaman berikutnya.")
        self.lbl_status.setText("Sensor Siap! Membuka sistem rekaman...")
        
        # Hubungkan ke main_gui jika sudah diintegrasikan penuh nanti
        if hasattr(self, 'parent_main_win'):
            self.parent_main_win.go_to_live_data_page()
        else:
            pass

    def close_threads(self):
        """Memastikan thread mati jika aplikasi ditutup paksa"""
        if hasattr(self, 'worker'):
            self.worker.stop()


# =====================================================================
# BLOK INDEPENDEN TEST RUN 
# =====================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Jalankan pengetesan visual loading page secara standalone
    test_loader = LoadingPage()
    test_loader.resize(1920, 1080)
    test_loader.setWindowTitle("Pratinjau Desain Kios TriaGO - Loading & Stabilisasi Sensor")
    test_loader.show()
    
    # Otomatis langsung start simulasi stabilisasi sensor 2 detik saat aplikasi terbuka
    test_loader.start_stabilization()
    
    sys.exit(app.exec())