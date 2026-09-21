from gpiozero import DigitalOutputDevice
from time import sleep

DIR_PIN = 21
STEP_PIN = 20

direction = DigitalOutputDevice(DIR_PIN)
step = DigitalOutputDevice(STEP_PIN)

# arah
direction.on()

# 200 step
for i in range(200):
    step.on()
    sleep(0.001)
    step.off()
    sleep(0.001)