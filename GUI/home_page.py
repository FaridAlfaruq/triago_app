import os
import sys
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QApplication, QLabel, QPushButton
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, pyqtSignal

class HomePage(QWidget):
    # Sinyal untuk memberi tahu Main Window bahwa user menekan tombol MULAI
    start_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Background warna hijau muda (#F6FFEC) khusus untuk halaman ini
        self.setStyleSheet("background-color: #F6FFEC;")

        # Layout utama vertikal
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setSpacing(30) # Jarak antara logo dan tombol

        # 1. Komponen Logo
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Path ke file logo Anda (sesuaikan nama filenya jika berbeda)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(current_dir, r"C:\Users\Adyty\Documents\Farid ITS\TriaGo\asset\logo.png") 
        
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            # Menyesuaikan ukuran logo secara proporsional (lebar diatur kisaran 450px)
            scaled_pixmap = pixmap.scaledToWidth(450, Qt.TransformationMode.SmoothTransformation)
            self.logo_label.setPixmap(scaled_pixmap)
        else:
            self.logo_label.setText("TriaGO Logo Missing")
            self.logo_label.setStyleSheet("color: #121212; font-size: 32px; font-weight: bold;")

        # 2. Komponen Button "MULAI"
        self.btn_start = QPushButton("MULAI")
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.setFixedSize(200, 50) # Ukuran tombol proporsional
        
        # Style tombol: background merah (#F65C5C), teks hijau muda (#F6FFEC), dan rounded border
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #F65C5C;
                color: #F6FFEC;
                font-size: 20px;
                font-weight: bold;
                border-radius: 25px; 
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton:hover {
                background-color: #E05252; /* Sedikit lebih gelap saat di-hover */
            }
            QPushButton:pressed {
                background-color: #C84848;
            }
        """)
        
        # Menghubungkan klik tombol ke pengirim sinyal kustom
        self.btn_start.clicked.connect(self.start_requested.emit)

        # Menyusun komponen ke layout utama
        main_layout.addWidget(self.logo_label)
        main_layout.addWidget(self.btn_start, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(main_layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Inisialisasi halaman secara mandiri
    tampilan_home = HomePage()
    
    # Mengatur ukuran window dummy saat test run agar mirip ukuran main_gui
    tampilan_home.resize(1920, 1080)
    tampilan_home.setWindowTitle("Test Run - Home Page")
    
    # Opsional: Cek apakah sinyal tombol klik berfungsi saat ditest
    tampilan_home.start_requested.connect(lambda: print("[TEST LOG] Tombol MULAI berhasil diklik!"))
    
    tampilan_home.show()
    sys.exit(app.exec())