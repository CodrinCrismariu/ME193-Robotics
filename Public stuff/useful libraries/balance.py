"""
Self-balancing double motor (inverted pendulum) using a PID loop.

STATUS: does NOT balance on this hardware, and cannot without a change to
the control path. Measured with latency.py, a motor command takes ~175 ms to
produce wheel motion (BLE round trip + motor ramp). This robot falls from
upright to 35 deg in ~360 ms, so the wheels react when the fall is already
half over. That caps usable control bandwidth near 1 Hz; an inverted
pendulum this size needs several Hz. No choice of gains fixes dead time.

The sensing side IS solved and correct (see below), and the loop itself runs
at ~155 Hz. What is left is the actuation delay. To actually balance, the
control loop has to run ON the hub rather than over Bluetooth.

Kept as a working reference for the sensor handling, and because the trace
output makes the latency limit easy to demonstrate.

Why this does not use imu.pitch
-------------------------------
This robot balances at pitch ~= 90 deg, which is exactly the Euler-angle
singularity: pitch pins at 90.0, and roll/yaw jump by 180 as it passes
through. Useless at the setpoint. Instead the tilt comes from the
accelerometer via atan2 in the X-Z plane, which is smooth and continuous
through upright, fused with the gyro by a complementary filter.

Also note: every IMU notification REBINDS dm.imu_device to a new object, so
it must be re-read inside the loop. Caching it once gives a frozen snapshot.
"""

import argparse
import math
import time

import legoeducation as le

# --- Hardware ------------------------------------------------------------
CARD_COLOR = le.LEGO_COLOR_GREEN
CARD_SERIAL = 994
NOTIFY_MS = 15          # fastest IMU push the library allows (~66 Hz)

# Measured with calibrate_gyro.py -- deg/s per raw gyro unit.
# Negative: the gyro's sign is opposite the accel tilt convention.
GYRO_SCALE = -1.11661

# --- Controller gains ----------------------------------------------------
KP = 25.0               # proportional: speed per degree of tilt
KI = 0.0                # integral: corrects steady-state lean (start at 0)
KD = 0.8                # derivative: damping, from the gyro rate
DRIVE_SIGN = 1          # flip to -1 if the robot drives itself over

ALPHA = 0.98            # complementary filter: gyro weight per step
MAX_SPEED = 100         # motor command clamp (percent)
FALL_CUTOFF = 35.0      # degrees from upright -> give up and stop
INTEGRAL_CLAMP = 50.0   # anti-windup limit
RUN_SECONDS = 20.0      # hard stop, so the motors can never be left running


def tilt_deg(imu):
    """Tilt from the accelerometer: 0 = upright, no singularity near balance."""
    return math.degrees(math.atan2(imu.accelerometerZ, -imu.accelerometerX))


def parse_args():
    ap = argparse.ArgumentParser(description="Self-balancing PID. Tune from the command line.")
    ap.add_argument("--kp", type=float, default=KP, help=f"proportional gain (default {KP})")
    ap.add_argument("--ki", type=float, default=KI, help=f"integral gain (default {KI})")
    ap.add_argument("--kd", type=float, default=KD, help=f"derivative gain (default {KD})")
    ap.add_argument("--sign", type=int, default=DRIVE_SIGN, choices=(1, -1),
                    help="flip if the robot drives itself over")
    ap.add_argument("--alpha", type=float, default=ALPHA, help=f"complementary filter (default {ALPHA})")
    ap.add_argument("--max-speed", type=int, default=MAX_SPEED, help=f"speed clamp (default {MAX_SPEED})")
    ap.add_argument("--seconds", type=float, default=RUN_SECONDS, help=f"hard stop (default {RUN_SECONDS})")
    ap.add_argument("--cutoff", type=float, default=FALL_CUTOFF, help=f"fall cutoff deg (default {FALL_CUTOFF})")
    ap.add_argument("--cmd-hz", type=float, default=25.0,
                    help="motor command rate (default 25). BLE drops the link "
                         "if you write much faster than ~30 Hz.")
    ap.add_argument("--deadband", type=int, default=2,
                    help="skip writes that change speed by less than this (default 2)")
    ap.add_argument("--release-rate", type=float, default=40.0,
                    help="deg/s of tilt rate that counts as 'released from the "
                         "support/hand' -- the survival clock starts there (default 40)")
    ap.add_argument("--quiet", action="store_true", help="skip the per-sample trace")
    return ap.parse_args()


def main():
    a = parse_args()
    kp, ki, kd = a.kp, a.ki, a.kd
    sign, alpha = a.sign, a.alpha
    max_speed, run_seconds, cutoff = a.max_speed, a.seconds, a.cutoff
    cmd_interval = 1.0 / a.cmd_hz if a.cmd_hz > 0 else 0.0
    print(f"gains: KP={kp} KI={ki} KD={kd} sign={sign} alpha={alpha} max={max_speed}")
    print(f"command rate {a.cmd_hz:.0f} Hz, deadband {a.deadband}")

    dm = le.DoubleMotor()
    dm.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL,
               device_notification_delay=NOTIFY_MS)
    print("connected")

    # --- Calibrate the upright setpoint --------------------------------
    print("hold the robot upright and STILL ...")
    samples = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        samples.append(tilt_deg(dm.imu_device))
        time.sleep(0.01)
    target = sum(samples) / len(samples)
    spread = max(samples) - min(samples)
    print(f"level tilt = {target:.2f} deg (noise {spread:.1f} deg) -- let go now")

    angle = target          # complementary filter state
    integral = 0.0
    last_cmd_t = 0.0
    last_speed = None
    sent = 0
    release_t = None        # when the robot actually became free-standing
    start = last_t = time.perf_counter()
    loops = 0
    worst = 0.0
    trace = []

    try:
        print("balancing (Ctrl+C to stop)")
        while True:
            now = time.perf_counter()
            if now - start > run_seconds:
                print(f"reached {run_seconds:.0f}s time limit -- stopping")
                break
            dt = now - last_t
            if dt < 0.005:        # don't spin faster than the IMU updates
                time.sleep(0.002)
                continue
            last_t = now

            imu = dm.imu_device   # re-read: each notification rebinds this
            accel_angle = tilt_deg(imu)
            rate = float(imu.gyroscopeY) * GYRO_SCALE

            # Complementary filter: trust the gyro short-term, the
            # accelerometer long-term.
            angle = alpha * (angle + rate * dt) + (1.0 - alpha) * accel_angle

            error = angle - target
            # Only count tilt AFTER release: while propped on a support or held,
            # error sits near zero and means nothing.
            if release_t is None:
                if abs(rate) > a.release_rate:
                    release_t = now
                    print(f"released at t={now - start:.2f}s -- clock started")
            else:
                worst = max(worst, abs(error))

            if release_t is not None and abs(error) > cutoff:
                dm.movement_move_tank(0, 0, blocking=False)
                print(f"fell over ({error:+.1f} deg) -- stopping")
                break

            integral += error * dt
            integral = max(-INTEGRAL_CLAMP, min(INTEGRAL_CLAMP, integral))

            output = sign * (kp * error + ki * integral + kd * rate)
            speed = int(max(-max_speed, min(max_speed, output)))

            # Throttle: writing every loop floods the BLE link and the hub
            # drops the connection.
            due = (now - last_cmd_t) >= cmd_interval
            changed = last_speed is None or abs(speed - last_speed) >= a.deadband
            if due and changed:
                if not dm.connected:
                    print("device disconnected -- stopping")
                    break
                try:
                    dm.movement_move_tank(speed, speed, blocking=False)
                except Exception as exc:
                    print(f"write failed ({exc}) -- stopping")
                    break
                last_cmd_t = now
                last_speed = speed
                sent += 1

            trace.append((now - start, error, rate, speed))
            loops += 1

    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        elapsed = time.perf_counter() - start
        if loops:
            print(f"loop rate {loops/elapsed:.0f} Hz, {sent} writes "
                  f"({sent/elapsed:.0f} Hz), worst tilt {worst:.1f} deg")
        if release_t is None:
            print("NEVER RELEASED -- it stayed on the support/hand the whole run, "
                  "so this tells us nothing about balancing.")
        else:
            print(f"SURVIVED {time.perf_counter() - release_t:.2f} s after release "
                  f"<-- the number to compare between runs")
        if trace and not a.quiet:
            print("")
            print("    t    error     rate  speed")
            step = max(1, len(trace) // 25)
            for i in range(0, len(trace), step):
                t, e, r, sp = trace[i]
                print(f"{t:5.2f} {e:8.1f} {r:8.1f} {sp:6d}")
            print("")
            print("If 'speed' has the SAME sign as 'error', the robot drives toward")
            print("its fall (correct -- raise KP). If OPPOSITE, set DRIVE_SIGN = -1.")
        for stop_call in (lambda: dm.movement_move_tank(0, 0, blocking=False),
                          dm.movement_stop, dm.disconnect):
            try:
                stop_call()
            except Exception:
                pass


if __name__ == "__main__":
    main()


# --- TUNING ---------------------------------------------------------------
# Catch the robot by hand while tuning. Work one gain at a time:
#
# 1. First check DRIVE_SIGN. If the robot accelerates the way it is already
#    falling (it "helps" itself over), set DRIVE_SIGN = -1.
# 2. KI = 0, KD = 0. Raise KP until it reacts briskly to a push and starts
#    oscillating, then back off ~20%.
# 3. Raise KD until the oscillation damps out. Too much makes the motors
#    chatter -- back off if so.
# 4. Only if it balances but drifts steadily one way, add a little KI (~0.5).
#
# A taller robot with mass up high falls more slowly and is much easier to
# balance than a short one.
