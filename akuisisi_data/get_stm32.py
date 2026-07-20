import serial
import serial.tools.list_ports
import time

def find_stm32_port(fallback='COM7'):
    for p in serial.tools.list_ports.comports():
        # VID:PID for STM32 Virtual COM Port (0483:5740)
        if "0483:5740" in p.hwid:
            return p.device
        # Alternative: look for USB Serial Device that is not Bluetooth
        if "usb" in p.description.lower() and "bluetooth" not in p.description.lower():
            return p.device
    return fallback

# UBAH KONFIGURASI DEFAULT SESUAI HASIL SUKSES
DEFAULT_PORT = find_stm32_port('COM7')
DEFAULT_BAUDRATE = 921600 # Diubah ke 921600 sesuai konfigurasi optimal

def stream_stm32_data(port=DEFAULT_PORT, baudrate=DEFAULT_BAUDRATE):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        # TAMBAH: Alokasikan memori buffer internal Windows sebesar 1MB agar data tidak meluap
        ser.set_buffer_size(rx_size=1024*1024, tx_size=65536)
        
        ser.dtr = True
        ser.rts = True
        
        # Bersihkan sisa data lama di buffer saat pertama kali konek
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        # Trigger STM32 untuk mulai mengirim data
        ser.write(b"START\n")
        
        raw_accumulator = b""
        
        while True:
            # === TAMBAH: NAPAS KOMPUTASI (5 ms) ===
            # Memberikan jeda agar CPU laptop tidak overload dan buffer OS terisi penuh
            time.sleep(0.005) 
            
            bytes_to_read = ser.in_waiting
            if bytes_to_read > 0:
                # OPTIMASI: Baca seluruh bongkahan data yang sudah mengantre sekaligus
                raw_accumulator += ser.read(bytes_to_read)
                
                if b'\n' in raw_accumulator:
                    lines = raw_accumulator.split(b'\n')
                    raw_accumulator = lines.pop() # Simpan baris gantung yang belum utuh
                    
                    for raw_line in lines:
                        clean_bytes = raw_line.replace(b'\x00', b'')
                        line = clean_bytes.decode('utf-8', errors='ignore').strip()
                        
                        if not line or "HEARTBEAT" in line:
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
                            except ValueError:
                                yield {"status": "ERROR", "message": f"Non-numeric data detected: {line}"}
                        else:
                            yield {"status": "WARNING", "message": f"Incomplete columns ({len(data)}/6): {line}"}
                                
    except serial.SerialException as e:
        print(f"[FATAL] Gagal mengakses port serial: {e}")
        raise e
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()