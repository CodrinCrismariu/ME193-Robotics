"""
Drive a LEGO Education double motor.

Setup:
    pip install legoeducation
    (keep lelib.py next to this script)
"""

import time

import legoeducation as le
from lelib import doubleMotor

# --- Bluetooth card info for your double motor ----------------------------
# The color/serial printed on your LEGO connection card.
# Valid colors: le.LEGO_COLOR_RED, _YELLOW, _BLUE, _GREEN, _PURPLE,
# _MAGENTA, _AZURE, _ORANGE.
MOTOR_CARD_COLOR = le.LEGO_COLOR_GREEN
MOTOR_CARD_SERIAL = 994


def main():
    dm = doubleMotor()
    dm.connect(card_serial=MOTOR_CARD_SERIAL, card_color=MOTOR_CARD_COLOR)
    print("connected")

    dm.set_speed(40)
    dm.reset_heading()

    # Drive both wheels forward for 2 seconds.
    print("forward...")
    dm.run_time(2000)

    # Turn in place.
    print("turning right 90 degrees...")
    dm.turn_right(90)
    print(f"yaw after turn: {dm.yaw():.1f} degrees")

    # Run one wheel at a time.
    print("left wheel one rotation...")
    dm.run_left(360)
    print("right wheel one rotation...")
    dm.run_right(360)

    # Continuous run, then stop. Negative speed goes backward.
    print("backward for 2 seconds...")
    dm.run(speed=-40)
    time.sleep(2)
    dm.stop()

    print(f"final yaw: {dm.yaw():.1f} degrees")
    dm.disconnect()
    print("done")


if __name__ == "__main__":
    main()
