import sys
import csv  # Tambahan untuk handle CSV
from datetime import datetime  # Tambahan untuk penamaan file otomatis
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import QThread, pyqtSignal
from get_stm32 import stream_stm32_data

class DataWorker(QThread):
    data_received = pyqtSignal(dict)
    finished_session = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.running = True
        self.sample_limit = 24000  # 400Hz * 60 detik
        self.count = 0

    def run(self):
        for packet in stream_stm32_data():
            if not self.running or self.count >= self.sample_limit:
                break
            
            if packet["status"] == "OK":
                self.data_received.emit(packet)
                self.count += 1
                
        self.finished_session.emit()

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STM32 Live Dual-Sensor Plotter & Auto-Save")
        self.resize(800, 650)
        
        # UI Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Label Informasi Suhu Real-time
        self.status_label = QLabel("Menghubungkan ke STM32... | Amb: --°C | Obj: --°C")
        self.status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #ffffff; padding: 5px;")
        layout.addWidget(self.status_label)
        
        # --- PLOT 1: GRAFIK ECG ---
        self.plot_ecg = pg.PlotWidget()
        self.plot_ecg.setBackground('k')
        self.plot_ecg.showGrid(x=True, y=True)
        self.plot_ecg.setTitle("Grafik ECG (Analog ADC)")
        self.plot_ecg.setLabel('left', 'Amplitudo ADC')
        layout.addWidget(self.plot_ecg)
        
        # --- PLOT 2: GRAFIK PPG (IR) ---
        self.plot_ppg = pg.PlotWidget()
        self.plot_ppg.setBackground('k')
        self.plot_ppg.showGrid(x=True, y=True)
        self.plot_ppg.setTitle("Grafik PPG (Infrared Channel Only)")
        self.plot_ppg.setLabel('left', 'Intensitas Cahaya')
        self.plot_ppg.setLabel('bottom', 'Sampel Terkini')
        layout.addWidget(self.plot_ppg)
        
        self.plot_ppg.setXLink(self.plot_ecg)
        
        # Setup Kurva dan Buffer
        self.curve_ecg = self.plot_ecg.plot(pen=pg.mkPen(color='#00FF00', width=1.5))
        self.curve_ppg = self.plot_ppg.plot(pen=pg.mkPen(color='#FF3333', width=1.5))
        
        self.ecg_buffer = []
        self.ppg_ir_buffer = []
        
        # --- STORAGE HISTORY DATA (UNTUK CSV) ---
        # Menampung seluruh data mentah lengkap (3 kanal PPG, ECG, dan 2 Sensor Suhu)
        self.all_data_history = []

        # Threading Backend
        self.worker = DataWorker()
        self.worker.data_received.connect(self.update_plots)
        self.worker.finished_session.connect(self.on_finished)
        self.worker.start()

    def update_plots(self, packet):
        # 1. Ekstrak data
        ecg_value = packet["ecg"]
        ppg_ir_value = packet["ppg"]["ir"]
        t_amb = packet["temperature"]["ambient"]
        t_obj = packet["temperature"]["object"]
        
        # 2. Simpan rekaman lengkap ke history tanpa di-pop (untuk target CSV nanti)
        self.all_data_history.append({
            "timestamp": packet["timestamp"],
            "ppg_red": packet["ppg"]["red"],
            "ppg_ir": ppg_ir_value,
            "ppg_green": packet["ppg"]["green"],
            "ecg": ecg_value,
            "temp_ambient": t_amb,
            "temp_object": t_obj
        })
        
        # 3. Masukkan ke buffer visualisasi grafik (tetap di-pop agar GUI ringan)
        self.ecg_buffer.append(ecg_value)
        self.ppg_ir_buffer.append(ppg_ir_value)
        
        if len(self.ecg_buffer) > 300:
            self.ecg_buffer.pop(0)
        if len(self.ppg_ir_buffer) > 300:
            self.ppg_ir_buffer.pop(0)
            
        # 4. Refresh grafik live
        self.curve_ecg.setData(self.ecg_buffer)
        self.curve_ppg.setData(self.ppg_ir_buffer)
        
        self.status_label.setText(
            f"Streaming Active ({self.worker.count}/{self.worker.sample_limit}) | "
            f"Amb: {t_amb:.2f}°C | Obj: {t_obj:.2f}°C"
        )

    def on_finished(self):
        """
        Dijalankan otomatis begitu thread mendeteksi sampel ke-24000 (60 detik)
        """
        self.status_label.setText("Sesi Selesai. Sedang menyimpan data ke CSV...")
        QApplication.processEvents() # Paksa GUI refresh teks sebelum proses I/O file
        
        # Membuat nama file unik berdasarkan waktu saat ini (contoh: data_20260717_153022.csv)
        filename = f"data_test_{datetime.now()}.csv"
        
        try:
            # Mulai menulis data ke file CSV
            with open(filename, mode='w', newline='') as csv_file:
                fieldnames = ["timestamp", "ppg_red", "ppg_ir", "ppg_green", "ecg", "temp_ambient", "temp_object"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                
                writer.writeheader()  # Tulis baris judul kolom
                writer.writerows(self.all_data_history)  # Dump seluruh array history
                
            self.status_label.setText(f"Selesai! Data sukses disimpan ke: {filename}")
            print(f"[INFO] File CSV berhasil dibuat: {filename}")
            
        except Exception as e:
            self.status_label.setText(f"Error saat menyimpan CSV: {e}")
            print(f"[ERROR] Gagal menulis CSV: {e}")

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())