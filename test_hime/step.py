from gpiozero import DigitalOutputDevice
from time import sleep

# =========================
# PIN CONFIGURATION
# =========================
STEP_PIN = 21
DIR_PIN = 20

# =========================
# MOTOR CONFIGURATION
# =========================
STEPS_PER_REV = 200
MICROSTEP = 1

# Kecepatan
START_DELAY = 0.008
MIN_DELAY = 0.0025

# Acceleration
ACCEL_STEPS = 300

# Jumlah putaran
TOTAL_REVOLUTIONS = 5


# =========================
# GPIO
# =========================
step = DigitalOutputDevice(
    STEP_PIN,
    initial_value=False
)

direction = DigitalOutputDevice(
    DIR_PIN,
    initial_value=False
)


def move_motor_cw(steps, revolution_number):

    total = abs(steps)

    # Set arah CW
    direction.on()

    direction_name = "CW (Clockwise / searah jarum jam)"

    degrees = total * 360 / (STEPS_PER_REV * MICROSTEP)

    print()
    print("=" * 55)
    print(f"[MOTOR] PUTARAN {revolution_number}/{TOTAL_REVOLUTIONS}")
    print(f"[MOTOR] Arah      : {direction_name}")
    print(f"[MOTOR] Step      : {total}")
    print(f"[MOTOR] Sudut     : {degrees:.2f}°")
    print(f"[MOTOR] STEP GPIO : {STEP_PIN}")
    print(f"[MOTOR] DIR GPIO  : {DIR_PIN}")
    print("=" * 55)

    # Acceleration
    accel = min(ACCEL_STEPS, total // 2)

    for i in range(total):

        # =========================
        # ACCELERATION
        # =========================
        if accel > 0 and i < accel:

            progress = i / accel

            delay = START_DELAY - (
                (START_DELAY - MIN_DELAY) * progress
            )

        # =========================
        # DECELERATION
        # =========================
        elif accel > 0 and i >= total - accel:

            progress = (total - i) / accel

            delay = START_DELAY - (
                (START_DELAY - MIN_DELAY) * progress
            )

        # =========================
        # CONSTANT SPEED
        # =========================
        else:
            delay = MIN_DELAY

        # STEP pulse
        step.on()
        sleep(delay)

        step.off()
        sleep(delay)

        # =========================
        # LOG PROGRESS
        # =========================
        if (i + 1) % 20 == 0 or i == total - 1:

            current_degree = (
                (i + 1)
                * 360
                / (STEPS_PER_REV * MICROSTEP)
            )

            percent = ((i + 1) / total) * 100

            print(
                f"[MOTOR] CW | "
                f"Putaran: {revolution_number}/{TOTAL_REVOLUTIONS} | "
                f"Step: {i + 1}/{total} | "
                f"Angle: {current_degree:.1f}° | "
                f"Progress: {percent:.0f}%"
            )

    print(
        f"[MOTOR] Putaran {revolution_number} "
        f"SELESAI - CW 360°"
    )


try:

    print()
    print("=" * 55)
    print("       STEPPER MOTOR - CW 5X")
    print("=" * 55)

    total_steps = STEPS_PER_REV * MICROSTEP

    # =========================
    # CW 360° × 5
    # =========================
    for revolution in range(1, TOTAL_REVOLUTIONS + 1):

        move_motor_cw(
            total_steps,
            revolution
        )

        # Jeda antar putaran
        if revolution < TOTAL_REVOLUTIONS:
            print(
                f"[MOTOR] Menunggu 1 detik "
                f"sebelum putaran berikutnya..."
            )
            sleep(1)

    print()
    print("=" * 55)
    print("[MOTOR] SEMUA GERAKAN SELESAI")
    print("[MOTOR] Total: 5x CW 360°")
    print("=" * 55)


except KeyboardInterrupt:

    print()
    print("[MOTOR] Dihentikan oleh user.")


finally:

    step.off()
    direction.off()

    print("[MOTOR] GPIO dimatikan.")