import time
from gpiozero import DigitalInputDevice

# Inisialisasi pin switch dengan pull-up internal
sw_start = DigitalInputDevice(16, pull_up=True)
sw_end   = DigitalInputDevice(26, pull_up=True)

print("=== PENGUJIAN LIMIT SWITCH (VERSI AWAL) ===")
print("Tekan masing-masing saklar untuk melihat perubahan angka.")
print("Tekan Ctrl+C untuk keluar.\n")

try:
    while True:
        # Baca nilai murni pin (0 atau 1)
        val_start = sw_start.value
        val_end   = sw_end.value

        print(f"GPIO 16 (START): {val_start}   |   GPIO 26 (END): {val_end}     ", end="\r", flush=True)
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nPengujian selesai.")