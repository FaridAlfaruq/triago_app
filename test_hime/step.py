from gpiozero import DigitalOutputDevice
from time import sleep

# Setup Pin
step = DigitalOutputDevice(21)
direction = DigitalOutputDevice(20)

# Konfigurasi
STEPS = 200      # 1 putaran penuh (360°)
DELAY = 0.0025   # Semakin kecil angka, semakin cepat putaran

try:
    # Set arah berlawanan jarum jam (CCW)
    direction.off()

    # Jalankan 1 putaran
    for _ in range(STEPS):
        step.on()
        sleep(DELAY)
        step.off()
        sleep(DELAY)

    print("Motor selesai berputar 1x CCW.")

finally:
    # Pastikan pin mati saat program selesai atau dihentikan
    step.close()
    direction.close()