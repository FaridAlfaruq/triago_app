from gpiozero import DigitalOutputDevice
from time import sleep

DIR_PIN = 21
STEP_PIN = 20

direction = DigitalOutputDevice(DIR_PIN)
step = DigitalOutputDevice(STEP_PIN)

direction.on()

def move_smooth(total_steps=200, min_delay=0.002, start_delay=0.008, ramp_ratio=0.25):
    """
    total_steps: Jumlah langkah motor
    min_delay  : Jeda saat mencapai kecepatan penuh (kecepatan puncak)
    start_delay: Jeda saat mulai/berhenti (kecepatan lambat)
    ramp_ratio : Porsi langkah yang digunakan untuk akselerasi/deselerasi
    """
    ramp_steps = int(total_steps * ramp_ratio)
    
    for i in range(total_steps):
        # Fase Akselerasi (awal)
        if i < ramp_steps:
            progress = i / ramp_steps
            delay = start_delay - (start_delay - min_delay) * progress
        # Fase Deselerasi (akhir)
        elif i >= (total_steps - ramp_steps):
            progress = (total_steps - 1 - i) / ramp_steps
            delay = start_delay - (start_delay - min_delay) * progress
        # Fase Kecepatan Konstan
        else:
            delay = min_delay
            
        step.on()
        sleep(delay)
        step.off()
        sleep(delay)

# Jalankan 1 putaran dengan akselerasi halus
move_smooth(total_steps=200, min_delay=0.002, start_delay=0.006)