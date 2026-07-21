import serial
import time

PORT = 'COM16'
BAUDRATE = 115200  # Sesuaikan dengan linecoding Virtual COM STM32 Anda

def read_ecg():
    print(f"--- ECG READER ACTIVE on {PORT} ---")
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=1)
        ser.set_buffer_size(rx_size=1024*1024, tx_size=65536)

        ser.dtr = True
        ser.rts = True
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        # Trigger STM32
        ser.write(b"START\n")
        print("[INFO] Sent 'START' command to STM32...")

        raw_accumulator = b""
        sample_count = 0
        start_time = time.time()

        while True:
            time.sleep(0.005)  # kasih jeda biar buffer OS terisi

            bytes_to_read = ser.in_waiting
            if bytes_to_read > 0:
                raw_accumulator += ser.read(bytes_to_read)

                if b'\n' in raw_accumulator:
                    lines = raw_accumulator.split(b'\n')
                    raw_accumulator = lines.pop()  # simpan sisa baris gantung

                    for raw_line in lines:
                        clean_bytes = raw_line.replace(b'\x00', b'')
                        line = clean_bytes.decode('utf-8', errors='ignore').strip()

                        if not line or "HEARTBEAT" in line:
                            continue

                        data = line.split(',')
                        if len(data) == 6:
                            try:
                                vals = list(map(float, data))
                                ecg_value = vals[3]
                                sample_count += 1

                                elapsed = time.time() - start_time
                                hz = sample_count / elapsed if elapsed > 0 else 0
                                print(f"[{hz:.1f} Hz] Sample #{sample_count} | ECG: {int(ecg_value)}")
                            except ValueError:
                                pass

    except serial.SerialException as e:
        print(f"[FATAL] Gagal buka port: {e}")
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh user.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("[INFO] Port ditutup.")

if __name__ == "__main__":
    read_ecg()