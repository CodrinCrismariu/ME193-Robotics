"""Measure achievable control-loop rate for balancing: IMU update rate + motor write rate."""
import time
import legoeducation as le

NOTIFY_MS = 15  # fastest the library allows (~66 Hz)

dm = le.DoubleMotor()
dm.connect(card_color=le.LEGO_COLOR_GREEN, card_serial=994,
           device_notification_delay=NOTIFY_MS)
print(f"connected (requested notification delay {NOTIFY_MS} ms)")

imu = dm.imu_device
print("imu fields:", [a for a in ("pitch", "roll", "yaw", "gyroscopeX", "gyroscopeY",
                                  "gyroscopeZ", "accelerometerX", "accelerometerY",
                                  "accelerometerZ") if hasattr(imu, a)])
print("pitch/roll/yaw now:", imu.pitch, imu.roll, imu.yaw)

# --- How fast does the IMU actually refresh? ---
t0 = time.perf_counter()
updates, last = 0, None
while time.perf_counter() - t0 < 3.0:
    cur = (imu.pitch, imu.gyroscopeY, imu.gyroscopeX)
    if cur != last:
        updates += 1
        last = cur
    time.sleep(0.001)
print(f"IMU: {updates} distinct updates in 3.0 s -> {updates/3.0:.1f} Hz")

# --- How fast can we push motor commands? ---
for blocking in (False, True):
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        dm.movement_move_tank(0, 0, blocking=blocking)
        n += 1
    el = time.perf_counter() - t0
    print(f"tank(blocking={blocking}): {n} writes in {el:.2f} s -> {n/el:.1f} Hz")

dm.movement_stop()
dm.disconnect()
