import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

DIR_PIN = 21
STEP_PIN = 20

GPIO.setup(DIR_PIN, GPIO.OUT)
GPIO.setup(STEP_PIN, GPIO.OUT)

# Atur arah
GPIO.output(DIR_PIN, GPIO.HIGH)

# Gerakkan motor
for i in range(200):
    GPIO.output(STEP_PIN, GPIO.HIGH)
    time.sleep(0.001)
    GPIO.output(STEP_PIN, GPIO.LOW)
    time.sleep(0.001)

GPIO.cleanup()