from gpiozero import DigitalOutputDevice
from time import sleep

# Setup Pin
step = DigitalOutputDevice(20)
direction = DigitalOutputDevice(21)

# Konfigurasi
STEPS_PER_REVOLUTION = 200
REVOLUTIONS = 5
DELAY = 0.0025

try:
    # Set arah CCW
    direction.off()

    # Jalankan 5 putaran
    total_steps = STEPS_PER_REVOLUTION * REVOLUTIONS

    for _ in range(total_steps):
        step.on()
        sleep(DELAY)
        step.off()
        sleep(DELAY)

    print(f"Motor selesai berputar {REVOLUTIONS}x CCW.")

finally:
    # Pastikan pin mati
    step.close()
    direction.close()