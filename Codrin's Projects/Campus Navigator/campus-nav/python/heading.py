"""Heading, 90-degree turns and per-stride features from the ankle IMU
(no compass, no GPS).

Heading. The MPU6050 has no magnetometer, so heading comes from the gyro
alone. On the ankle the shank swings through ~60 deg and twists by +/-10-20
deg every stride, at the same time; rotations about tilted axes don't simply
add, so summing a "yaw rate" picks up a few degrees of false turning per
stride (tested: hundreds of degrees over a minute). Instead the sensor's full
attitude is integrated as a quaternion (exact for any combination of
rotations), its tilt gently pulled toward gravity by the accelerometer, and
heading is the compass direction of a sensor axis across the shank. That
direction twists back and forth each stride, so turns are found on heading
averaged over about a stride. On simulated walks: ~20 deg of drift a minute.

Turns. A turn is the averaged heading leaving the current straight's heading
by more than TURN_MIN deg and then settling. It is snapped to the nearest
multiple of 90 deg (+ = left, counterclockwise seen from above; a U-turn is
+/-180). While walking straight, the reference heading slowly follows the
measured one, which soaks up gyro drift and gentle curves; a bend that
settles between DEPART and TURN_MIN deg is absorbed too.

Strides. Each stride runs between the planted moments (least rotation) just
after consecutive foot contacts of this leg, which the step counter finds and
passes in with add_contact(). For each stride a few features are kept for the
stairs classifier (stairs.py): height change (vertical acceleration in the
world frame integrated twice, velocity zero at both planted ends; noisy on
the ankle, +/-0.1-0.3 m on flat ground, so it's one feature among several),
stride time, how far the shin tilts forward and back, and the rotation and
impact peaks.

The gyro bias (heading drifts by bias x time) is measured whenever the leg is
still, so stand still for a couple of seconds now and then (you do at every
labelled location anyway).

numpy only; plain Python floats in the per-sample loop (100 Hz).
"""
import math
from collections import deque

import numpy as np

FS = 100.0
DT = 1.0 / FS
DEG = math.pi / 180.0
G = 9.81

KP_TILT = 0.5              # rad/s per unit error: tilt pulled toward gravity ...
TILT_ACC = 0.3             # ... while | |a| - 1 g | is below this
STILL_WINDOW = 200         # gyro bias (as in gait.py)
STILL_DPS = 4.0
# Turns
SMOOTH_S = 1.2             # heading averaged over about a stride
SETTLE_S = 1.2             # settled: averaged heading moved less than SETTLE_DEG over this long
SETTLE_DEG = 15.0
DEPART = 20.0              # leaving the straight's heading
TURN_MIN = 55.0            # a turn (snapped to 90 / 180)
BEND_SETTLE_S = 4.0        # a smaller change that stays this long is a bend: absorbed
TRACK_TAU_S = 8.0          # reference follows drift and gentle curves while walking straight
TICK = 10                  # turn detector runs every TICK samples (10 Hz)
# Strides
ANCHOR_WINDOW_S = 0.6      # the planted moment: calmest rotation this soon after a contact
STRIDE_MAX_S = 2.5
FEATURES = ("dz", "seconds", "tilt_fwd", "tilt_back", "tilt_mean", "w_peak", "acc_peak", "az_range")


def _q_from_up(a):
    """Quaternion (sensor -> world) turning the measured up vector a onto +z."""
    n = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2) or 1.0
    ux, uy, uz = a[0] / n, a[1] / n, a[2] / n
    w = 1.0 + uz                                        # axis u x z = (uy, -ux, 0), angle acos(uz)
    if w < 1e-6:
        return (0.0, 1.0, 0.0, 0.0)
    m = math.sqrt(w * w + uy * uy + ux * ux)
    return (w / m, uy / m, -ux / m, 0.0)


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


class Heading:
    """Feed raw samples with process(); tell it the step counter's contacts
    with add_contact(); pop() the turns and strides found since the last pop."""

    def __init__(self):
        self.bias = np.zeros(3)
        self._bias_set = False
        self.last_still_n = None                # when the bias was last measured
        self._still = deque(maxlen=STILL_WINDOW)
        self.n = 0                              # samples processed
        self.q = None                           # attitude, sensor -> world (w, x, y, z)
        self.axis = None                        # sensor axis across the shank: its compass direction is the heading
        self.shin = None                        # sensor axis along the shin (gravity when last still)
        self._az = None
        self.yaw_deg = 0.0                      # unwrapped, + = counterclockwise (left)
        self._yaw_raw = deque(maxlen=int(SMOOTH_S * FS))
        self._yaw_hist = deque(maxlen=int(10 * FS / TICK))    # (sample, smoothed yaw) at 10 Hz
        self.ref = None                         # heading of the current straight
        self._depart_n = None
        self.moving = False
        self._buf = deque(maxlen=int(2 * STRIDE_MAX_S * FS))  # (n, |a|, |w|, vertical accel, shin tilt)
        self._contacts = []
        self._last_anchor = None
        self.turns = []                         # {n, n0, n1, angle, measured}
        self.strides = []                       # {n, n0, <FEATURES>}

    # --- public ----------------------------------------------------------------------

    def process(self, accel, gyro):
        """(k, 3) g and (k, 3) deg/s, sensor axes."""
        for a, g in zip(np.asarray(accel, float).tolist(), np.asarray(gyro, float).tolist()):
            self._sample(a, g)

    def add_contact(self, n):
        """A foot contact of this leg (sample index, counted like self.n)."""
        self._contacts.append(int(n))

    def reset_turns(self):
        """Start a new straight from the current heading (e.g. at a new walk)."""
        self.ref = self._smoothed()
        self._depart_n = None

    def relative_heading(self) -> float:
        """Averaged heading minus the current straight's, deg (+ = left)."""
        return 0.0 if self.ref is None else self._smoothed() - self.ref

    @property
    def calibrated(self) -> bool:
        """The gyro bias has been measured (2 s standing still). Until then,
        heading drifts by the raw bias: a few deg/s on a real MPU6050."""
        return self._bias_set

    def seconds_since_still(self):
        return None if self.last_still_n is None else (self.n - self.last_still_n) / FS

    def pop(self):
        out = (self.turns, self.strides)
        self.turns, self.strides = [], []
        return out

    # --- per sample --------------------------------------------------------------------

    def _sample(self, a, g):
        self.n += 1
        self._still.append(a + g)
        if self.n % 50 == 0:
            self._update_bias()
        if self.q is None:
            self._init_attitude(a)
        wx, wy, wz = g[0] - self.bias[0], g[1] - self.bias[1], g[2] - self.bias[2]
        self._integrate(a, wx * DEG, wy * DEG, wz * DEG)
        az = self._azimuth()
        d = az - self._az
        d -= 2 * math.pi * round(d / (2 * math.pi))
        self._az = az
        self.yaw_deg += d / DEG
        self._yaw_raw.append(self.yaw_deg)

        qw, qx, qy, qz = self.q
        vx, vy, vz = 2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz
        an = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
        wn = math.sqrt(wx * wx + wy * wy + wz * wz)
        vert = (vx * a[0] + vy * a[1] + vz * a[2] - 1.0) * G
        s = self.shin
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, vx * s[0] + vy * s[1] + vz * s[2]))))
        self._buf.append((self.n, an, wn, vert, tilt))
        if self.n % TICK == 0:
            self._turn_tick()
            self._contact_tick()

    def _init_attitude(self, a):
        self.q = _q_from_up(a)
        up = _unit(a)
        self.shin = tuple(up)
        e = np.eye(3)[int(np.argmin(np.abs(up)))]
        self.axis = tuple(_unit(e - (e @ up) * up))
        self._az = self._azimuth()

    def _integrate(self, a, wx, wy, wz):
        """One gyro step of the attitude, tilt nudged toward gravity."""
        qw, qx, qy, qz = self.q
        an = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
        if abs(an - 1.0) < TILT_ACC:
            vx = 2.0 * (qx * qz - qw * qy)                 # world up in sensor axes
            vy = 2.0 * (qy * qz + qw * qx)
            vz = qw * qw - qx * qx - qy * qy + qz * qz
            ax, ay, az = a[0] / an, a[1] / an, a[2] / an
            wx += KP_TILT * (ay * vz - az * vy)
            wy += KP_TILT * (az * vx - ax * vz)
            wz += KP_TILT * (ax * vy - ay * vx)
        h = 0.5 * DT
        q = (qw - h * (qx * wx + qy * wy + qz * wz),
             qx + h * (qw * wx + qy * wz - qz * wy),
             qy + h * (qw * wy - qx * wz + qz * wx),
             qz + h * (qw * wz + qx * wy - qy * wx))
        m = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        self.q = (q[0] / m, q[1] / m, q[2] / m, q[3] / m)

    def _azimuth(self):
        """Compass angle (rad, counterclockwise) of self.axis in the world."""
        qw, qx, qy, qz = self.q
        x, y, z = self.axis
        wxv = (1 - 2 * (qy * qy + qz * qz)) * x + 2 * (qx * qy - qw * qz) * y + 2 * (qx * qz + qw * qy) * z
        wyv = 2 * (qx * qy + qw * qz) * x + (1 - 2 * (qx * qx + qz * qz)) * y + 2 * (qy * qz - qw * qx) * z
        return math.atan2(wyv, wxv)

    def _update_bias(self):
        if len(self._still) < STILL_WINDOW:
            return
        arr = np.array(self._still)
        if arr[:, 3:].std(axis=0).max() < STILL_DPS and np.linalg.norm(arr[:, :3], axis=1).std() < 0.03:
            m = arr[:, 3:].mean(axis=0)
            self.bias = m if not self._bias_set else 0.8 * self.bias + 0.2 * m
            self._bias_set = True
            self.last_still_n = self.n
            up = tuple(arr[-50:, :3].mean(axis=0))
            self.shin = tuple(_unit(up))
            if self.q is not None:                         # still: tilt straight from gravity
                self.q = _q_from_up(up)
                self._az = self._azimuth()                 # (a new compass zero, not a turn)
            self.moving = False

    # --- turns ------------------------------------------------------------------------

    def _smoothed(self):
        return sum(self._yaw_raw) / len(self._yaw_raw) if self._yaw_raw else self.yaw_deg

    def _turn_tick(self):
        h = self._smoothed()
        self._yaw_hist.append((self.n, h))
        if self.ref is None:
            self.ref = h
            return
        off = h - self.ref
        if abs(off) <= DEPART:
            self._depart_n = None
            if self.moving:
                self.ref += off * (TICK / FS) / TRACK_TAU_S
            return
        if self._depart_n is None:
            self._depart_n = self.n
        recent = [y for n, y in self._yaw_hist if n > self.n - SETTLE_S * FS]
        settled = len(recent) >= SETTLE_S * FS / TICK - 1 and max(recent) - min(recent) < SETTLE_DEG
        if abs(off) >= TURN_MIN and settled:
            k = max(-2, min(2, round(off / 90.0))) or (1 if off > 0 else -1)
            half = self.ref + off / 2.0                 # the turn's moment: halfway through it
            n_mid = next((n for n, y in self._yaw_hist
                          if n >= self._depart_n - SETTLE_S * FS and (y - half) * off >= 0), self.n)
            self.turns.append({"n": int(n_mid), "n0": int(self._depart_n), "n1": int(self.n),
                               "angle": int(90 * k), "measured": round(off, 1)})
            self.ref = h
            self._depart_n = None
        elif abs(off) < TURN_MIN and self.n - self._depart_n >= BEND_SETTLE_S * FS:
            stable = [y for n, y in self._yaw_hist if n > self.n - BEND_SETTLE_S * FS]
            if max(stable) - min(stable) < SETTLE_DEG:
                self.ref = h                           # a bend in the path, not a turn
                self._depart_n = None

    # --- strides ------------------------------------------------------------------------

    def _contact_tick(self):
        while self._contacts and self.n >= self._contacts[0] + ANCHOR_WINDOW_S * FS:
            c = self._contacts.pop(0)
            window = [s for s in self._buf if c <= s[0] <= c + ANCHOR_WINDOW_S * FS]
            if len(window) < 5:
                self._last_anchor = None
                continue
            w = np.convolve([s[2] for s in window], np.ones(5) / 5, mode="same")
            self._stride_to(window[int(np.argmin(w[2:-2])) + 2][0])

    def _stride_to(self, anchor):
        prev, self._last_anchor = self._last_anchor, anchor
        self.moving = True
        if prev is None or anchor - prev > STRIDE_MAX_S * FS:
            return
        rows = np.array([s[1:] for s in self._buf if prev <= s[0] <= anchor])
        if len(rows) < 20:
            return
        an, wn, vert, tilt = rows.T
        v = np.cumsum(vert) * DT
        v -= np.linspace(0.0, v[-1], len(v))           # planted at both ends: zero velocity
        stand = float(np.median(tilt[:5]))
        self.strides.append({
            "n": int(anchor), "n0": int(prev),
            "dz": round(float(np.sum(v) * DT), 3),
            "seconds": round((anchor - prev) / FS, 2),
            "tilt_fwd": round(float(tilt.max() - stand), 1),
            "tilt_back": round(float(stand - tilt.min()), 1),
            "tilt_mean": round(float(tilt.mean()), 1),
            "w_peak": round(float(wn.max()), 0),
            "acc_peak": round(float(an.max()), 2),
            "az_range": round(float(vert.max() - vert.min()), 1),
        })
