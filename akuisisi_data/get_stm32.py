import serial
import time

# KONFIGURASI DEFAULT
DEFAULT_PORT = 'COM7'
DEFAULT_BAUDRATE = 115200

def stream_stm32_data(port=DEFAULT_PORT, baudrate=DEFAULT_BAUDRATE):
    """
    Generator function untuk membaca, membersihkan, dan memparsing data 
    dari STM32 secara real-time.
    Yields:
        dict: Data sensor terstruktur jika valid, atau None jika terjadi error parsing.
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        ser.dtr = True
        ser.rts = True
        
        # Trigger STM32 untuk mulai mengirim data
        ser.write(b"START\n")
        
        while True:
            if ser.in_waiting > 0:
                # 1. Baca byte mentah & eliminasi byte sampah (\x00)
                raw_bytes = ser.readline()
                clean_bytes = raw_bytes.replace(b'\x00', b'')
                line = clean_bytes.decode('utf-8', errors='ignore').strip()
                
                if not line:
                    continue
                
                # 2. Parsing pembatas koma
                data = line.split(',')
                
                # 3. Validasi 6 kolom data: [RED, IR, GREEN, ECG, T_AMB, T_OBJ]
                if len(data) == 6:
                    try:
                        vals = list(map(float, data))
                        
                        # Pack ke dalam dictionary terstruktur (Keduanya sudah masuk di sini)
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
                    # Lewati log heartbeat bawaan STM32 agar tidak mengotori warning
                    if "HEARTBEAT" not in line:
                        yield {"status": "WARNING", "message": f"Incomplete columns ({len(data)}/6): {line}"}
                        
    except serial.SerialException as e:
        print(f"[FATAL] Gagal mengakses port serial: {e}")
        raise e
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    print(f"=== Memulai Konsumsi Data dari STM32 pada {DEFAULT_PORT} ===")
    
    try:
        for packet in stream_stm32_data():
            if packet["status"] == "OK":
                # FIX: Sekarang mencetak Ambient Temp DAN Object Temp sekaligus
                print(f"Time: {packet['timestamp']:.2f} | "
                      f"PPG: {list(packet['ppg'].values())} | "
                      f"ECG: {packet['ecg']} | "
                      f"Amb: {packet['temperature']['ambient']}°C | "
                      f"Obj: {packet['temperature']['object']}°C")
            else:
                print(f"[{packet['status']}] {packet.get('message')}")
                
    except KeyboardInterrupt:
        print("\n[INFO] Proses pembacaan dihentikan oleh pengguna.")