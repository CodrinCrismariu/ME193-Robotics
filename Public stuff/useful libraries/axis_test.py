"""Tip the robot slowly forward and back by hand; see which axis tracks it."""
import time
import legoeducation as le

dm = le.DoubleMotor()
dm.connect(card_color=le.LEGO_COLOR_GREEN, card_serial=994,
           device_notification_delay=15)
imu = dm.imu_device
print("tip the robot slowly FORWARD then BACK for 12 seconds...\n")
print(f"{'t':>5} {'pitch':>8} {'roll':>8} {'yaw':>8} {'gyroX':>8} {'gyroY':>8} {'gyroZ':>8}")

t0 = time.perf_counter()
try:
    while time.perf_counter() - t0 < 12.0:
        t = time.perf_counter() - t0
        print(f"{t:5.1f} {imu.pitch/10:8.1f} {imu.roll/10:8.1f} {imu.yaw/10:8.1f} "
              f"{imu.gyroscopeX:8d} {imu.gyroscopeY:8d} {imu.gyroscopeZ:8d}")
        time.sleep(0.4)
finally:
    dm.disconnect()
