import matplotlib.pyplot as plt
import numpy as np
import serial

PORT = "COM16"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)

# Siapkan jendela grafik
plt.ion()
fig, ax = plt.subplots()

# Perpendek jumlah sampel penampung (misal 100 data terakhir agar lebih ringan)
window_size = 100
xdata = list(range(window_size))
ydata = [0] * window_size
(line,) = ax.plot(xdata, ydata)

ax.set_ylim(0, 4095)
ax.set_title("Live ECG Monitor - Stable Mode")
ax.set_xlabel("Sampel Waktu")
ax.set_ylabel("Amplitudo Sensor")

print("Mulai memantau ECG (Stabil)... Tekan Ctrl+C untuk berhenti.")

try:
  while True:
    line_data = ser.readline().decode("utf-8", errors="ignore").strip()
    if line_data.isdigit():
      val = int(line_data)

      ydata.append(val)
      ydata.pop(0)

      # Perbarui data garis
      line.set_ydata(ydata)

      # Gunakan fungsi pause(0.001) agar event loop GUI sempat bernapas dan tidak crash
      plt.pause(0.001)

except KeyboardInterrupt:
  ser.close()
  print("\nPerekaman dihentikan.")