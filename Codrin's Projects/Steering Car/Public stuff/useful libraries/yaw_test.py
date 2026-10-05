"""Calibrate doubleMotor.yaw() units: three successive 90-degree right turns."""
import time
import legoeducation as le
from lelib import doubleMotor

dm = doubleMotor()
dm.connect(card_serial=994, card_color=le.LEGO_COLOR_GREEN)
print("connected")

dm.set_speed(40)
dm.reset_heading()
time.sleep(0.5)
print(f"yaw at rest after reset: {dm.yaw()}")

total = 0
for i in range(1, 4):
    dm.turn_right(90)
    time.sleep(0.5)
    total += 90
    y = dm.yaw()
    print(f"after turn {i} (commanded {total} deg total): yaw={y}  ratio={y/total:.2f}")

dm.stop()
dm.disconnect()
