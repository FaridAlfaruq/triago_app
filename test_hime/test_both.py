import time
from gpiozero import DigitalOutputDevice, DigitalInputDevice

# Pin Stepper
DIR_PIN = DigitalOutputDevice(20)
STEP_PIN = DigitalOutputDevice(21)

# Pin Limit Switch
SW_START = DigitalInputDevice(16, pull_up=True)
SW_END = DigitalInputDevice(26, pull_up=True)

# Nilai saat saklar tertabrak:
# 1 = jika pakai COM + NC
# 0 = jika pakai COM + NO
HIT_VALUE = 0  

# Arah gerakan (1 = maju/atas, 0 = mundur/bawah)
DIR_PIN.value = 1 # CW (menjauhi stepper motor) # CCW/0 (mendekati stepper motor)

print("Motor berjalan... Tekan limit switch untuk STOP.")

try:
    while True:
        # Cek saklar sebelum melangkah
        if SW_START.value == HIT_VALUE:
            print("\n[STOP] Switch START (GPIO 16) tertabrak!")
            break

        if SW_END.value == HIT_VALUE:
            print("\n[STOP] Switch END (GPIO 26) tertabrak!")
            break

        # 1 Langkah (Step pulse)
        STEP_PIN.on()
        time.sleep(0.1)  # Kecepatan (makin kecil makin cepat)
        STEP_PIN.off()
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nDihentikan manual.")

finally:
    STEP_PIN.off()
    DIR_PIN.off()
    print("Selesai.")