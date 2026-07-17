import serial

# KONFIGURASI
PORT = 'COM7' 
BAUDRATE = 115200

def debug_cdc():
    print(f"--- SNIFFER AKTIF pada {PORT} ---")
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=1)
        ser.dtr = True; ser.rts = True
        
        # Kirim perintah START ke STM32
        ser.write(b"START\n")
        print("[INFO] Mengirim perintah 'START' ke STM32...")
        
        while True:
            if ser.in_waiting > 0:
                # 1. Baca byte mentah
                raw_bytes = ser.readline()
                
                # 2. Bersihkan byte sampah (0x00) dan decode
                # Ini akan menghilangkan semua \x00 yang Anda lihat di HEX output
                clean_bytes = raw_bytes.replace(b'\x00', b'')
                line = clean_bytes.decode('utf-8', errors='ignore').strip()
                
                if not line: continue
                
                # 3. Parsing
                data = line.split(',')
                
                # 4. Validasi data
                if len(data) == 6:
                    try:
                        # Konversi ke float untuk memastikan data valid
                        vals = list(map(float, data))
                        print(f"DATA OK: PPG:[{int(vals[0])},{int(vals[1])},{int(vals[2])}] ECG:{int(vals[3])} Temp:[{vals[4]:.2f}, {vals[5]:.2f}]")
                    except ValueError:
                        print(f"[ERROR] Data berisi non-angka: {line}")
                else:
                    print(f"[WARNING] Data tidak lengkap (Kolom != 6): {line}")
                    
    except serial.SerialException as e:
        print(f"[FATAL] Gagal membuka port: {e}")
    except KeyboardInterrupt:
        print("\n[INFO] Sniffer dihentikan.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    debug_cdc()