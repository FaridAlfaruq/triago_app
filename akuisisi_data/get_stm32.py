import os
import time

import serial
from serial.tools import list_ports


DEFAULT_BAUDRATE = 921600
DEFAULT_DATA_TIMEOUT_SECONDS = 5.0
DEFAULT_STREAM_START_TIMEOUT_SECONDS = 10.0


def find_stm32_port():
    """Cari port STM32, dengan override melalui environment TRIAGO_SERIAL_PORT."""
    configured_port = os.environ.get("TRIAGO_SERIAL_PORT")
    if configured_port:
        return configured_port

    ports = list(list_ports.comports())
    preferred_markers = (
        "stm32",
        "stlink",
        "virtual com",
        "usb serial",
        "ttyacm",
        "ttyusb",
    )
    for port in ports:
        description = f"{port.device} {port.description} {port.manufacturer or ''}".lower()
        if any(marker in description for marker in preferred_markers):
            return port.device

    if len(ports) == 1:
        return ports[0].device

    return None


def stream_stm32_data(
    port=None,
    baudrate=DEFAULT_BAUDRATE,
    should_stop=None,
    data_timeout=DEFAULT_DATA_TIMEOUT_SECONDS,
    stream_start_timeout=DEFAULT_STREAM_START_TIMEOUT_SECONDS,
):
    """Alirkan paket STM32 sampai koneksi berhenti atau diminta berhenti."""
    should_stop = should_stop or (lambda: False)
    try:
        port = port or find_stm32_port()
        if not port:
            raise serial.SerialException(
                "Port STM32 tidak ditemukan. Sambungkan sensor atau set "
                "TRIAGO_SERIAL_PORT."
            )

        ser = serial.Serial(port, baudrate, timeout=1)
        # set_buffer_size tidak tersedia pada seluruh platform/driver serial.
        if hasattr(ser, "set_buffer_size"):
            ser.set_buffer_size(rx_size=1024 * 1024, tx_size=65536)
        
        ser.dtr = True
        ser.rts = True
        
        # Bersihkan sisa data lama di buffer saat pertama kali konek
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        # Beri waktu endpoint USB CDC siap setelah DTR/RTS diaktifkan.
        time.sleep(0.5)

        # Trigger STM32 untuk mulai mengirim data
        ser.write(b"START\n")
        ser.flush()

        raw_accumulator = b""
        last_sensor_activity_at = time.monotonic()
        last_start_sent_at = last_sensor_activity_at
        stream_started_at = last_sensor_activity_at
        has_valid_packet = False
        
        while not should_stop():
            # === TAMBAH: NAPAS KOMPUTASI (5 ms) ===
            # Memberikan jeda agar CPU laptop tidak overload dan buffer OS terisi penuh
            time.sleep(0.005) 
            
            bytes_to_read = ser.in_waiting
            if bytes_to_read > 0:
                last_sensor_activity_at = time.monotonic()
                # OPTIMASI: Baca seluruh bongkahan data yang sudah mengantre sekaligus
                raw_accumulator += ser.read(bytes_to_read)
                
                if b'\n' in raw_accumulator:
                    lines = raw_accumulator.split(b'\n')
                    raw_accumulator = lines.pop() # Simpan baris gantung yang belum utuh
                    
                    for raw_line in lines:
                        clean_bytes = raw_line.replace(b'\x00', b'')
                        line = clean_bytes.decode('utf-8', errors='ignore').strip()
                        
                        if not line or "HEARTBEAT" in line or "SYS_STATUS" in line:
                            continue
                        
                        # Parsing pembatas koma
                        data = line.split(',')
                        
                        # Validasi 6 kolom data: [RED, IR, GREEN, ECG, T_AMB, T_OBJ]
                        if len(data) == 6:
                            try:
                                vals = list(map(float, data))
                                
                                # Pack ke dalam dictionary terstruktur untuk GUI
                                yield {
                                    "status": "OK",
                                    "timestamp": time.time(),
                                    "ppg": {
                                        "red": int(vals[0]),
                                        "ir": int(vals[1]),
                                        "green": int(vals[2])
                                    },
                                    "ecg": int(vals[3]),
                                    "temperature": {
                                        "ambient": vals[4],
                                        "object": vals[5]
                                    }
                                }
                                has_valid_packet = True
                            except ValueError:
                                yield {"status": "ERROR", "message": f"Non-numeric data detected: {line}"}
                        else:
                            yield {"status": "WARNING", "message": f"Incomplete columns ({len(data)}/6): {line}"}

            now = time.monotonic()
            if not has_valid_packet and now - last_start_sent_at >= 1.0:
                ser.write(b"START\n")
                ser.flush()
                last_start_sent_at = now

            if (
                not has_valid_packet
                and stream_start_timeout is not None
                and now - stream_started_at >= stream_start_timeout
            ):
                raise serial.SerialTimeoutException(
                    "STM32 terhubung dan sensor terdeteksi sehat, tetapi firmware "
                    "tidak memulai stream data setelah menerima START."
                )

            if (
                data_timeout is not None
                and now - last_sensor_activity_at >= data_timeout
            ):
                raise serial.SerialTimeoutException(
                    f"Tidak ada respons serial dari sensor selama {data_timeout:g} detik."
                )
                                
    except serial.SerialException as e:
        print(f"[FATAL] Gagal mengakses port serial: {e}")
        raise e
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
