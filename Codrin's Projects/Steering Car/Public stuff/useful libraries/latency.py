"""Measure command -> wheel-motion latency. Wheels must be free to spin."""
import time
import legoeducation as le

dm = le.DoubleMotor()
dm.connect(card_color=le.LEGO_COLOR_GREEN, card_serial=994,
           device_notification_delay=15)
print("connected -- hold the robot with wheels OFF the ground")
time.sleep(1.0)

def wheel_speed():
    m = dm.motor
    try:
        return float(m[0].speed)
    except (TypeError, IndexError, AttributeError):
        return float(m.speed)

delays = []
for trial in range(5):
    dm.movement_move_tank(0, 0, blocking=False)
    time.sleep(1.0)
    base = wheel_speed()

    t0 = time.perf_counter()
    dm.movement_move_tank(80, 80, blocking=False)
    moved = None
    while time.perf_counter() - t0 < 2.0:
        s = wheel_speed()
        if s == s and abs(s - base) > 5:      # NaN-safe
            moved = time.perf_counter() - t0
            break
        time.sleep(0.001)
    dm.movement_move_tank(0, 0, blocking=False)
    if moved is None:
        print(f"trial {trial+1}: no motion detected within 2 s (base={base})")
    else:
        delays.append(moved)
        print(f"trial {trial+1}: {moved*1000:6.1f} ms to first motion")
    time.sleep(0.5)

dm.movement_stop()
dm.disconnect()
if delays:
    print(f"\nmean latency {sum(delays)/len(delays)*1000:.0f} ms, "
          f"best {min(delays)*1000:.0f} ms")
