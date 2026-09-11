"""Measure the gyroscope's raw-int16 -> deg/s scale factor.

Tilt angle comes from the accelerometer (absolute but noisy); the gyro gives
rate (smooth but unscaled). Integrating the gyro and least-squares fitting it
against the accel angle recovers the scale.
"""
import math
import time
import legoeducation as le

DURATION = 14.0


def tilt_deg(imu):
    """Tilt in the X-Z plane: 0 = upright, no singularity near balance."""
    return math.degrees(math.atan2(imu.accelerometerZ, -imu.accelerometerX))


dm = le.DoubleMotor()
dm.connect(card_color=le.LEGO_COLOR_GREEN, card_serial=994,
           device_notification_delay=15)
print("connected")
print(f"Tip the robot back and forth through as WIDE an angle as you can")
print(f"(roughly +/-30 deg), several times, for {DURATION:.0f} seconds. Go.")

samples = []
t0 = time.perf_counter()
last = t0
while time.perf_counter() - t0 < DURATION:
    now = time.perf_counter()
    dt = now - last
    if dt < 0.005:
        time.sleep(0.002)
        continue
    last = now
    imu = dm.imu_device
    samples.append((dt, tilt_deg(imu), float(imu.gyroscopeY)))

dm.movement_stop()
dm.disconnect()

# Cumulative gyro integral (raw units) vs accel angle change (degrees).
a0 = samples[0][1]
G = 0.0
num = den = 0.0
amin = amax = a0
for dt, a, g in samples:
    G += g * dt
    A = a - a0
    num += G * A
    den += G * G
    amin, amax = min(amin, a), max(amax, a)

print(f"\n{len(samples)} samples, tilt range {amin:.1f} to {amax:.1f} deg "
      f"(swing {amax-amin:.1f} deg)")
if den == 0 or amax - amin < 10:
    print("Not enough motion to fit -- tip it through a wider angle and retry.")
else:
    scale = num / den
    print(f"GYRO_SCALE = {scale:.5f}   (deg/s per raw unit)")
    print(f"  i.e. {1/scale:.1f} raw units per deg/s")
