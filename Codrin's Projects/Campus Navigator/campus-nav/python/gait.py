"""Shank frame: the sensor's raw axes turned into the shank's own axes.

The band can sit anywhere around the ankle, at any angle, on either leg. So
before the network sees anything, every sample is rotated into a frame that
follows the shank:

  ML    side to side: the main rotation axis across the shank over the last
        5 s of motion (twist about the shank itself is left out first). Its
        sign says which way is forward (see below).
  long  along the shank: the mean acceleration (gravity) over the last 5 s,
        made perpendicular to ML.
  AP    front to back: ML x long.

Which way is forward is decided once, after 6 s of continuous motion.
Knees only bend forward, so in any gait the shank leans with the knee ahead
of the ankle on average (~12 deg walking, ~30 deg running) compared with
standing: that tells forward from backward whichever way you walk (right in
~97 % of simulated sessions, starting forwards or backwards). It needs a
still moment first; without one, the forward swing being the fastest and
most lopsided rotation decides (which assumes the first walk is forward).
The decision stays until reset() or 60 s of stillness (the band may have
been re-strapped). Walking backwards then swings negative.

The frame updates every 0.5 s and is frozen while still. The gyro bias is
measured whenever the leg is still. One more channel is the shank angle: the
ML gyro integrated with a 2 s leak, so it can't drift.

This exact code runs on the board (stream.py) and on the laptop to prepare
the simulated training data (step-counter-training/dataset.py), so both see
identical inputs. Results don't depend on how the samples are chunked.
numpy only; works with numpy 1.26 and 2.x.
"""
import numpy as np

FS = 100.0
UPDATE = 50                 # samples between frame updates (0.5 s)
WINDOW = 500                # samples the frame is estimated from (5 s)
MOVING_DPS = 30.0           # gyro RMS over the window above this: moving
STILL_DPS = 4.0             # gyro spread over 2 s below this: still (bias is measured)
LOCK_AFTER_S = 6.0          # continuous motion before forward is decided
RELOCK_S = 60.0
ANGLE_TAU_S = 2.0
CHANNELS = ("acc_ap", "acc_ml", "acc_long", "gyro_ap", "gyro_ml", "gyro_long", "angle_ml")


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def forward_score(acc_dir, still_dir, gyro_ml, ap):
    """> 0 if the ML axis (whose AP = ML x long is `ap`) points the forward-
    swing-positive way. acc_dir: mean acceleration direction while moving;
    still_dir: gravity direction from the last still moment, or None."""
    p99, p1 = np.percentile(gyro_ml, 99), np.percentile(gyro_ml, 1)
    swing = 2.0 * np.log(max(p99, 1e-3) / max(-p1, 1e-3))
    sd = gyro_ml.std()
    if sd > 1e-6:
        swing += np.mean((gyro_ml - gyro_ml.mean()) ** 3) / sd ** 3
    if still_dir is None:
        return float(swing)
    # The knee's forward lean decides; the swing cues (which flip when walking
    # backwards) only break a near tie.
    lean = np.degrees(np.arcsin(np.clip((acc_dir - still_dir) @ ap, -1.0, 1.0)))
    return float(lean + 0.1 * swing)


class ShankFrame:
    def __init__(self):
        self.reset()

    def reset(self):
        self._buf = np.zeros((0, 6))
        self._count = 0
        self.long = None
        self.ml = None
        self.locked = False
        self._still = 0
        self._moving_for = 0
        self.still_dir = None
        self.bias = np.zeros(3)
        self._bias_set = False
        self.angle = 0.0
        self.moving = False

    def unlock(self):
        """Decide forward again from the next walk (e.g. after a reset)."""
        self.locked = False
        self._moving_for = 0

    @property
    def matrix(self) -> np.ndarray:
        """Rows: AP, ML, long, in sensor coordinates."""
        return np.stack([np.cross(self.ml, self.long), self.ml, self.long])

    def process(self, accel: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """(n, 3) g and (n, 3) deg/s in sensor axes -> (n, 7) in CHANNELS order."""
        raw = np.hstack([np.asarray(accel, float), np.asarray(gyro, float)])
        if self.long is None and len(raw):
            self._init_frame(raw[0])
        out = np.empty((len(raw), len(CHANNELS)))
        i = 0
        while i < len(raw):
            # Samples up to the next update boundary share one frame.
            j = min(len(raw), i + (UPDATE - self._count % UPDATE))
            block = raw[i:j]
            out[i:j] = self._rotate(block)
            self._buf = np.vstack([self._buf, block])[-WINDOW:]
            self._count += j - i
            if self._count % UPDATE == 0:
                self._update()
            i = j
        return out

    # --- internals -------------------------------------------------------------------

    def _init_frame(self, first):
        self.long = _unit(first[:3])
        axis = np.eye(3)[int(np.argmin(np.abs(self.long)))]       # any axis across the shank
        self.ml = _unit(axis - axis @ self.long * self.long)

    def _rotate(self, block):
        M = self.matrix
        acc = block[:, :3] @ M.T
        gyr = (block[:, 3:] - self.bias) @ M.T
        # Leaky integral of the ML rate: y_k = a y_k-1 + w_k dt, vectorized over the block.
        dt = 1.0 / FS
        a = np.exp(-dt / ANGLE_TAU_S)
        powers = a ** np.arange(1, len(block) + 1)
        angle = powers * self.angle + dt * powers * np.cumsum(gyr[:, 1] / powers)
        if len(block):
            self.angle = float(angle[-1])
        return np.hstack([acc, gyr, angle[:, None]])

    def _update(self):
        win = self._buf
        acc, gyr_raw = win[:, :3], win[:, 3:]
        last = win[-200:]
        still_now = (len(last) == 200 and last[:, 3:].std(axis=0).max() < STILL_DPS and
                     np.linalg.norm(last[:, :3], axis=1).std() < 0.03)
        if still_now:
            # First estimate straight away, then slowly, so one wobbly 2 s window
            # (a weight shift) can't drag it far.
            m = last[:, 3:].mean(axis=0)
            self.bias = m if not self._bias_set else 0.8 * self.bias + 0.2 * m
            self._bias_set = True
            d = _unit(last[:, :3].mean(axis=0))
            self.still_dir = d if self.still_dir is None else _unit(0.8 * self.still_dir + 0.2 * d)
        gyr = gyr_raw - self.bias
        acc_dir = _unit(acc.mean(axis=0))
        self.long = _unit(0.7 * self.long + 0.3 * acc_dir)
        rms = float(np.sqrt((gyr ** 2).sum(axis=1).mean()))
        self.moving = rms > MOVING_DPS
        if self.moving:
            self._still = 0
            self._moving_for += UPDATE
            # Main rotation axis *across* the shank: the twist about the shank
            # itself (the band rotating around the leg, turns) is removed first.
            # On a real ankle it can be as strong as the swing and would otherwise
            # pull the axis toward the shank.
            across = gyr - np.outer(gyr @ self.long, self.long)
            _, vecs = np.linalg.eigh(across.T @ across / len(across))
            ml_new = _unit(vecs[:, -1] - vecs[:, -1] @ self.long * self.long)
            if ml_new @ self.ml < 0:                  # follow the axis without flipping it
                ml_new = -ml_new
            if not self.locked:
                ml_new = _unit(0.5 * self.ml + 0.5 * ml_new) if self._moving_for > UPDATE else ml_new
                ap = np.cross(ml_new, self.long)
                if forward_score(acc_dir, self.still_dir, gyr @ ml_new, ap) < 0:
                    ml_new = -ml_new                  # provisional until locked
                if self._moving_for >= LOCK_AFTER_S * FS:
                    self.locked = True
                self.ml = ml_new
            else:
                self.ml = _unit(0.7 * self.ml + 0.3 * ml_new)
        else:
            self._moving_for = 0
            self._still += UPDATE
            if self._still >= RELOCK_S * FS:
                self.locked = False
        self.ml = _unit(self.ml - self.ml @ self.long * self.long)
