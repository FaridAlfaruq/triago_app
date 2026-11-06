from time import sleep
from gpiozero import DigitalOutputDevice

# Setup Pin
step = DigitalOutputDevice(21)
direction = DigitalOutputDevice(20)

# PENGATURAN MANUAL
ARAH = "CCW"  # Pilih "CW" (maju) atau "CCW" (mundur)
JUMLAH_STEP = 100 # Jumlah langkah per eksekusi (misal: 50, 100, 200)
DELAY = 0.001  # Kecepatan gerak (makin besar nilainya makin pelan)

try:
    # Tentukan arah
    if ARAH.upper() == "CW":
        direction.on()
    else:
        direction.off()

    print(f"Menggerakkan motor {JUMLAH_STEP} step ke arah {ARAH}...")

    # Perulangan sederhana
    for _ in range(JUMLAH_STEP):
        step.on()
        sleep(DELAY)
        step.off()
        sleep(DELAY)

    print("Selesai.")

finally:
    step.close()
    direction.close()