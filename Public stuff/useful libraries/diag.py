"""Accelerometer axes while tipping -- to build a singularity-free tilt angle."""
import math
import time
import legoeducation as le

dm = le.DoubleMotor()
dm.connect(card_color=le.LEGO_COLOR_GREEN, card_serial=994,
           device_notification_delay=15)
print("connected")
print("Hold UPRIGHT 4s, then tip slowly FORWARD, then BACK. 14 seconds.")
print(f"{'t':>5} {'accX':>7} {'accY':>7} {'accZ':>7} {'gyroY':>7} "
      f"{'atan2(X,Z)':>11} {'atan2(Y,Z)':>11} {'atan2(X,Y)':>11}")

t0 = time.perf_counter()
try:
    while time.perf_counter() - t0 < 14.0:
        t = time.perf_counter() - t0
        imu = dm.imu_device
        ax, ay, az = imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ
        d = math.degrees
        print(f"{t:5.1f} {ax:7d} {ay:7d} {az:7d} {imu.gyroscopeY:7d} "
              f"{d(math.atan2(ax, az)):11.1f} {d(math.atan2(ay, az)):11.1f} "
              f"{d(math.atan2(ax, ay)):11.1f}")
        time.sleep(0.4)
finally:
    dm.movement_stop()
    dm.disconnect()
