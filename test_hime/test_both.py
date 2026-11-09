
import time
from gpiozero import DigitalOutputDevice, DigitalInputDevice

DIR = DigitalOutputDevice(20)
STEP = DigitalOutputDevice(21)
START = DigitalInputDevice(16, pull_up=True)
END = DigitalInputDevice(26, pull_up=True)

SPEED = 0.001
HIT = 0

direction = "CCW"

try:
    while True:
        DIR.value = 1 if direction == "CW" else 0

        if direction == "CCW" and START.value == HIT:
            direction = "CW"
            print("START tersentuh -> CW")
            time.sleep(0.2)
            continue

        if direction == "CW" and END.value == HIT:
            direction = "CCW"
            print("END tersentuh -> CCW")
            time.sleep(0.2)
            continue

        STEP.on()
        time.sleep(SPEED)
        STEP.off()
        time.sleep(SPEED)

except KeyboardInterrupt:
    print("Dihentikan manual.")

finally:
    STEP.off()
    DIR.off()
    STEP.close()
    DIR.close()
    START.close()
    END.close()
    print("Pengujian selesai.")
