import serial
import time

PORT = 'COM7' 
BAUDRATE = 115200  # Sesuaikan dengan linecoding Virtual COM STM32 Anda

def debug_cdc():
    print(f"--- OPTIMIZED SNIFFER ACTIVE on {PORT} ---")
    try:
        # Buka port tanpa menggunakan rx_buf_size di init
        ser = serial.Serial(PORT, BAUDRATE, timeout=1)
        ser.set_buffer_size(rx_size=1024*1024, tx_size=65536)
        
        ser.dtr = True; ser.rts = True
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        # Trigger STM32
        ser.write(b"START\n")
        print("[INFO] Sent 'START' command to STM32...")
        
        raw_accumulator = b""
        sample_count = 0
        start_time = time.time()
        
        while True:
            # OPTIMASI UTAMA: Berikan jeda 5ms agar buffer OS terisi penuh oleh data USB CDC
            time.sleep(0.005) 
            
            bytes_to_read = ser.in_waiting
            if bytes_to_read > 0:
                # Baca bongkahan data besar sekaligus dari buffer
                raw_accumulator += ser.read(bytes_to_read)
                
                if b'\n' in raw_accumulator:
                    lines = raw_accumulator.split(b'\n')
                    raw_accumulator = lines.pop() # Simpan sisa baris gantung
                    
                    for raw_line in lines:
                        clean_bytes = raw_line.replace(b'\x00', b'')
                        line = clean_bytes.decode('utf-8', errors='ignore').strip()
                        
                        if not line or "HEARTBEAT" in line: 
                            continue
                            
                        data = line.split(',')
                        if len(data) == 6:
                            try:
                                vals = list(map(float, data))
                                sample_count += 1
                                
                                # Print setiap 40 sampel (sekitar 0.1 detik sekali pada 400Hz)
                                if sample_count % 40 == 0:
                                    elapsed = time.time() - start_time
                                    hz = sample_count / elapsed
                                    print(f"[{hz:.1f} Hz] Total Samples: {sample_count} | ECG: {int(vals[3])} | Obj: {vals[5]:.2f}°C")
                            except ValueError:
                                pass
                                
            # Batasan uji coba 60 detik
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
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    debug_cdc()