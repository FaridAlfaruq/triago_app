import matplotlib.pyplot as plt
import numpy as np
import serial
import time

PORT = "COM16"  # Sesuaikan dengan port COM STM32 kamu
BAUDRATE = 115200


def live_ecg_plotter():
  print(f"--- STABLE ECG PLOTTER ACTIVE on {PORT} ---")
  try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    ser.set_buffer_size(rx_size=1024 * 1024, tx_size=65536)

    ser.dtr = True
    ser.rts = True
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # Siapkan jendela grafik real-time
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    window_size = 300  # Jumlah titik sampel yang tampil di layar
    xdata = list(range(window_size))
    ydata = [0] * window_size
    (line,) = ax.plot(xdata, ydata, color="red", linewidth=1.5)

    ax.set_ylim(0, 4095)  # Rentang ADC 12-bit STM32
    ax.set_xlim(0, window_size)
    ax.set_title("Live ECG Signal Monitor (Optimized)")
    ax.set_xlabel("Sampel")
    ax.set_ylabel("Amplitudo")
    ax.grid(True)

    raw_accumulator = b""
    sample_count = 0
    start_time = time.time()
    last_draw_time = time.time()

    while True:
      time.sleep(0.005)

      bytes_to_read = ser.in_waiting
      if bytes_to_read > 0:
        raw_accumulator += ser.read(bytes_to_read)

        if b"\n" in raw_accumulator:
          lines = raw_accumulator.split(b"\n")
          raw_accumulator = lines.pop()  # Simpan sisa baris gantung

          new_data_arrived = False

          for raw_line in lines:
            clean_bytes = raw_line.replace(b"\x00", b"")
            line_str = clean_bytes.decode("utf-8", errors="ignore").strip()

            if not line_str:
              continue

            try:
              val = float(line_str)
              sample_count += 1

              ydata.append(val)
              ydata.pop(0)
              new_data_arrived = True

              if sample_count % 100 == 0:
                elapsed = time.time() - start_time
                hz = sample_count / elapsed
                print(
                    f"[{hz:.1f} Hz] Total Sampel: {sample_count} | ECG Value:"
                    f" {int(val)}"
                )

            except ValueError:
              pass

          # Pengaman agar Matplotlib tidak crash saat merender gambar
          if new_data_arrived and (time.time() - last_draw_time >= 0.03):
            line.set_ydata(ydata)
            fig.canvas.draw()
            fig.canvas.flush_events()
            last_draw_time = time.time()

  except serial.SerialException as e:
    print(f"[FATAL] Gagal membuka port: {e}")
  except KeyboardInterrupt:
    print("\n[INFO] Plotter dihentikan oleh pengguna.")
  finally:
    if "ser" in locals() and ser.is_open:
      ser.close()
      print("[INFO] Port serial ditutup.")


if __name__ == "__main__":
  live_ecg_plotter()