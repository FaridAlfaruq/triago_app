import time
from gpiozero import DigitalInputDevice

# pull_up=True mengaktifkan resistor pull-up internal
sw_start = DigitalInputDevice(16, pull_up=True)
sw_end = DigitalInputDevice(26, pull_up=True)

print("=== Monitor Limit Switch (gpiozero) ===")
print("Tekan Ctrl+C untuk berhenti.\n")

try:
    while True:
        # .value membaca langsung nilai logic: 0 (GND) atau 1 (3.3V)
        val_start = sw_start.value
        val_end = sw_end.value

        print(f"GPIO 16: {val_start}  |  GPIO 26: {val_end}   ", end="\r", flush=True)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nPengujian selesai.")