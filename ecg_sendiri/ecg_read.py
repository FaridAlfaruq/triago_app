import sys
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import serial

PORT = "COM16"  # Sesuaikan dengan port COM STM32 kamu
BAUDRATE = 115200


class LiveECGPlotter:

  def __init__(self):
    # 1. Inisialisasi Aplikasi GUI
    self.app = QtWidgets.QApplication(sys.argv)

    # 2. Setup Jendela Grafik
    self.win = pg.GraphicsLayoutWidget(
        show=True, title="Live ECG Monitor - STM32 CDC"
    )
    self.win.resize(1000, 450)

    self.plot = self.win.addPlot(title="Sinyal ECG Real-Time")
    self.plot.setYRange(0, 4095)  # Rentang ADC 12-bit STM32
    self.plot.setLabel("bottom", "Jumlah Sampel")
    self.plot.setLabel("left", "Nilai Amplitudo ADC")
    self.plot.showGrid(x=True, y=True)

    # Garis grafik warna merah khas sinyal medis
    self.curve = self.plot.plot(pen=pg.mkPen(color="#FF3333", width=2))

    # Array penampung data grafik (tampilkan 300 sampel terakhir di layar)
    self.window_size = 2000
    self.ydata = [0] * self.window_size

    # Akumulator buffer serial
    self.raw_accumulator = b""
    self.sample_count = 0
    self.start_time = time.time()

    # 3. Buka Port Serial & Kirim Perintah START
    try:
      self.ser = serial.Serial(PORT, BAUDRATE, timeout=1)
      self.ser.set_buffer_size(rx_size=1024 * 1024, tx_size=65536)

      self.ser.dtr = True
      self.ser.rts = True
      self.ser.reset_input_buffer()
      self.ser.reset_output_buffer()

      time.sleep(0.5)
      self.ser.write(b"START\n")
      print(f"[INFO] Port {PORT} terbuka, perintah 'START' terkirim.")

    except serial.SerialException as e:
      print(f"[FATAL] Gagal membuka port {PORT}: {e}")
      sys.exit(1)

    # 4. Timer GUI untuk membaca buffer serial & update kurva tanpa bikin freeze
    self.timer = QtCore.QTimer()
    self.timer.timeout.connect(self.update_loop)
    self.timer.start(5)  # Cek buffer tiap 5 milidetik

  def update_loop(self):
    try:
      bytes_to_read = self.ser.in_waiting
      if bytes_to_read > 0:
        self.raw_accumulator += self.ser.read(bytes_to_read)

        if b"\n" in self.raw_accumulator:
          lines = self.raw_accumulator.split(b"\n")
          self.raw_accumulator = lines.pop()  # Simpan sisa baris gantung

          new_point = False
          for raw_line in lines:
            clean_bytes = raw_line.replace(b"\x00", b"")
            line_str = clean_bytes.decode("utf-8", errors="ignore").strip()

            if not line_str:
              continue

            try:
              val = float(line_str)
              self.sample_count += 1

              # Geser window data
              self.ydata.pop(0)
              self.ydata.append(val)
              new_point = True

              # Cetak status ke terminal tiap 100 sampel
              if self.sample_count % 100 == 0:
                elapsed = time.time() - self.start_time
                hz = self.sample_count / elapsed
                print(
                    f"[{hz:.1f} Hz] Total Sampel: {self.sample_count} | Nilai"
                    f" ECG: {int(val)}"
                )

            except ValueError:
              pass

          # Update kurva grafik secara efisien
          if new_point:
            self.curve.setData(self.ydata)

    except Exception as e:
      print(f"[ERROR] {e}")

  def close_event(self):
    # Kirim perintah STOP dan lepas port serial saat jendela ditutup
    if hasattr(self, "ser") and self.ser.is_open:
      try:
        self.ser.write(b"STOP\n")
        print("[INFO] Perintah 'STOP' terkirim ke STM32.")
        self.ser.close()
        print("[INFO] Port serial ditutup dengan rapi.")
      except Exception:
        pass


if __name__ == "__main__":
  app_instance = LiveECGPlotter()

  # Tangkap tombol close jendela agar port serial selalu terlepas
  app_instance.app.aboutToQuit.connect(app_instance.close_event)

  sys.exit(app_instance.app.exec())