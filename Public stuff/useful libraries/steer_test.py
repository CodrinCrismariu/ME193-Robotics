"""Steering-only check using ABSOLUTE positioning. No driving.

Commands a series of angles and reads the encoder back, so the commanded and
actual angle can be compared directly. Returns to the same place each time --
that is the point of absolute positioning.
"""
import time
import legoeducation as le
from trike import Trike, STEER_CENTER_ABS

t = Trike()
t.steer.connect(card_color=le.LEGO_COLOR_GREEN, card_serial=994)
t.steer.motor_set_speed(40)
t.steer.motor_set_end_state(le.MOTOR_END_STATE_HOLD)
print(f"connected -- straight ahead is absolute position {STEER_CENTER_ABS}\n")
print(f"{'commanded':>10} {'actual':>8} {'error':>7}  {'raw abs':>8}")

try:
    for angle in (0, 30, -30, 60, -60, 0):
        t.set_steer(angle, wait=True)
        time.sleep(0.6)
        actual = t.steer_position()
        raw = t.steer.motor.absolutePosition
        print(f"{angle:+10.0f} {actual:+8.1f} {actual-angle:+7.1f} {raw:8d}")
finally:
    t.set_steer(0, wait=True)
    t.steer.disconnect()
    print("\nrecentred")
