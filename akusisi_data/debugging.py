
import os
import sys
import serial
import time
from serial.tools import list_ports


BAUDRATE = 115200


def detect_serial_ports():
    """Mendeteksi seluruh port serial yang tersedia."""
    ports = list(list_ports.comports())

    if not ports:
        print("[ERROR] Tidak ditemukan perangkat serial.")
        return []

    print("\n--- AVAILABLE SERIAL PORTS ---")

    for i, port in enumerate(ports, start=1):
        print(f"[{i}] {port.device} | {port.description}")

    return ports


def select_serial_port():
    """Memilih port berdasarkan deteksi otomatis atau environment variable."""

    # Prioritas 1: Environment variable
    env_port = os.environ.get("TRIAGO_SERIAL_PORT")

    if env_port:
        print(f"[INFO] Using environment port: {env_port}")
        return env_port

    # Prioritas 2: Deteksi port
    ports = detect_serial_ports()

    if not ports:
        return None

    # Jika hanya ada satu port, langsung gunakan
    if len(ports) == 1:
        selected = ports[0].device
        print(f"[INFO] Automatically selected: {selected}")
        return selected

    # Jika lebih dari satu port, minta pilihan pengguna
    while True:
        try:
            choice = int(input("\nPilih nomor port serial: "))

            if 1 <= choice <= len(ports):
                selected = ports[choice - 1].device
                print(f"[INFO] Selected port: {selected}")
                return selected

            print("[WARN] Nomor port tidak valid.")

        except ValueError:
            print("[WARN] Masukkan angka yang valid.")


def debug_cdc():

    PORT = select_serial_port()

    if PORT is None:
        return

    print(f"\n--- OPTIMIZED SNIFFER ACTIVE on {PORT} ---")

    ser = None

    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=1)

        # Buffer configuration
        if sys.platform == "win32":
            ser.set_buffer_size(rx_size=1024*1024, tx_size=65536)

        ser.dtr = True
        ser.rts = True

        ser.reset_input_buffer()
        ser.reset_output_buffer()

        time.sleep(0.5)

        # Trigger STM32
        ser.write(b"START\n")
        ser.flush()

        print("[INFO] Sent 'START' command to STM32...")

        raw_accumulator = b""
        sample_count = 0
        status_count = 0

        start_time = time.time()
        last_start_time = start_time

        while True:

            time.sleep(0.005)

            bytes_to_read = ser.in_waiting

            if bytes_to_read > 0:

                raw_accumulator += ser.read(bytes_to_read)

                if b'\n' in raw_accumulator:

                    lines = raw_accumulator.split(b'\n')
                    raw_accumulator = lines.pop()

                    for raw_line in lines:

                        clean_bytes = raw_line.replace(b'\x00', b'')
                        line = clean_bytes.decode(
                            'utf-8', errors='ignore'
                        ).strip()

                        if not line or "HEARTBEAT" in line:
                            continue

                        if "SYS_STATUS" in line:
                            status_count += 1

                            if status_count == 1:
                                print(f"[CONNECTED] {line}")

                            continue

                        data = line.split(',')

                        if len(data) == 6:

                            try:
                                vals = list(map(float, data))
                                sample_count += 1

                                red_val = int(vals[0])
                                ir_val = int(vals[1])
                                green_val = int(vals[2])
                                ecg_val = int(vals[3])
                                amb_temp = vals[4]
                                obj_temp = vals[5]

                                if sample_count % 40 == 0:

                                    elapsed = time.time() - start_time
                                    hz = sample_count / elapsed

                                    print(
                                        f"[{hz:.1f} Hz] #{sample_count} | "
                                        f"RED: {red_val:<6} | "
                                        f"IR: {ir_val:<6} | "
                                        f"GREEN: {green_val:<6} | "
                                        f"ECG: {ecg_val:<4} | "
                                        f"Obj: {obj_temp:.2f}°C | "
                                        f"Amb: {amb_temp:.2f}°C"
                                    )

                            except ValueError:
                                print(f"[WARN] Non-numeric packet: {line}")

            now = time.time()

            if sample_count == 0 and now - last_start_time >= 1.0:
                ser.write(b"START\n")
                ser.flush()

                last_start_time = now

                print("[RETRY] Sent 'START' again; waiting for 6-column data...")

            if sample_count == 0 and now - start_time >= 10.0:

                print(
                    "[ERROR] STM32 terhubung, tetapi firmware "
                    "tidak memulai stream setelah menerima START."
                )

                print(f"[INFO] SYS_STATUS received: {status_count}")
                break

            # Batasan pengujian 60 detik
            if time.time() - start_time >= 60.0:

                elapsed_actual = time.time() - start_time

                print("\n--- 1-MINUTE TEST BENCH DONE ---")
                print(f"Actual Elapsed Time : {elapsed_actual:.2f} s")
                print(f"Total Samples Saved : {sample_count} samples")
                print(f"Success Percentage  : {(sample_count / 24000) * 100:.2f}%")

                break

    except serial.SerialException as e:
        print(f"[FATAL] Failed to open port: {e}")

    except KeyboardInterrupt:
        print("\n[INFO] Sniffer stopped by user.")

    finally:
        if ser is not None and ser.is_open:
            ser.close()
            print("[INFO] Serial port closed.")


if __name__ == "__main__":
    debug_cdc()
