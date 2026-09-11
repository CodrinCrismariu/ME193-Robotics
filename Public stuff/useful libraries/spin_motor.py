"""
Spin a single LEGO Education motor.

Setup:
    pip install legoeducation
    (keep lelib.py next to this script)
"""

import time

import legoeducation as le
from lelib import singleMotor

# --- Bluetooth card info for your motor -----------------------------------
# Fill these in with the color/serial printed on your LEGO connection card.
# Valid colors: le.LEGO_COLOR_RED, _YELLOW, _BLUE, _GREEN, _PURPLE,
# _MAGENTA, _AZURE, _ORANGE.
MOTOR_CARD_COLOR = le.LEGO_COLOR_GREEN
MOTOR_CARD_SERIAL = 994


def main():
    motor = singleMotor()
    motor.connect(card_serial=MOTOR_CARD_SERIAL, card_color=MOTOR_CARD_COLOR)
    print("connected")

    # Spin exactly 2 full rotations, then stop.
    motor.spin(rotations=2)

    # Run continuously at half speed for 3 seconds, then stop.
    motor.run(speed=50)
    time.sleep(3)
    motor.stop()

    # Negative speed runs the other way.
    motor.run(speed=-50)
    time.sleep(3)
    motor.stop()

    print("done")


if __name__ == "__main__":
    main()
