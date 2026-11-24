# Copy of step-counter/python/stream.py (campus-nav adds contact_log). Keep the two in step.
"""Live step counting: raw IMU in, steps and distance out.

    shank frame (gait.py) -> network (nn.py) -> contacts, activity, speed
    -> steps, distance, cadence (this file)

The network runs on fixed 0.25 s blocks of new samples, with 2.5 s of context
before them, and each block's outputs are appended to one continuous stream,
so nothing is scored twice and results don't depend on how samples arrive.
Its outputs describe the moment 0.4 s back (model.LOOKAHEAD), so a counted
step shows up about half a second after it happens.

Steps: each contact of the instrumented foot is one stride = 2 steps: one
counted at the contact, the other foot's half a stride later, so the count
ticks up one step at a time. A "bout" only counts once 2 contacts in a row
have plausible, similar stride times while the network says the person is
moving; until then they're held, and a bout that never gets there (a few
taps, a fidget) is dropped.
Distance: the network's speed, integrated while it says the person is
moving; |speed| counts both directions. Walking backwards only counts (and
shows) once most of the last 3 s looked backward, so turning around isn't
mistaken for it.

The laptop evaluation (step-counter-training/evaluate.py) runs this same
class over simulated sessions.
"""
from collections import deque

import numpy as np

import gait
from nn import MODEL, Net

FS = gait.FS
CLASSES = ("standing", "walking", "running", "backward")
BLOCK = 25                   # samples per network run: results update 4 times a second
THRESHOLD = 0.5              # contact probability peak
MIN_GAP = 35                 # samples between contacts of one foot (0.35 s)
MAX_GAP = 260                # a longer pause ends the bout (2.6 s)
BOUT_MIN = 2                 # contacts before a bout counts (simulated: no more false steps than 3)
RATIO = (0.6, 1.6)           # stride time vs the previous one, while confirming a bout
KEEP_S = 20.0                # history kept for the page's plots
BACK_S = 3.0                 # "walking backwards" only after this long looking backward...
BACK_SHARE = 0.6             # ...in at least this share of it: a turn (1-2 s) isn't walking backwards


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class Pipeline:
    def __init__(self, height_m=1.75, model_path=MODEL, bout_min=BOUT_MIN):
        self.bout_min = bout_min
        self.net = Net(model_path)
        self.selftest_err = self.net.selftest()
        self.height = float(height_m)
        self.frame = gait.ShankFrame()
        self.ctx = self.net.receptive_field - 1
        self.D = self.net.lookahead
        keep = int(KEEP_S * FS)
        self._chan = deque(maxlen=keep)       # shank-frame channels by input sample (plots)
        self._prob = deque(maxlen=keep)       # contact probability by label sample
        self._cls = deque(maxlen=keep)
        self._n_in = 0                        # input samples seen
        self._n_label = 0                     # label samples produced
        self.reset_stream()
        self.reset_counts()

    # --- control ------------------------------------------------------------------------

    def reset_stream(self):
        """After lost samples: refill the network's context, end the bout."""
        self._inputs = np.zeros((0, 8))
        self._pending = np.zeros((0, 8))
        self._filled = 0
        self._peak_from = None
        self._end_bout()

    def reset_counts(self):
        self.steps = 0
        self.distance = 0.0
        self.backward = 0.0
        self._bout = []                       # contacts (label sample index) of the current bout
        self._confirmed = False
        self._last_contact = None
        self._stride_dist = deque(maxlen=8)
        self._other_foot_due = None           # label sample when the other foot's step is counted
        self._dist_at_contact = 0.0
        self._recent = deque(maxlen=int(FS))  # (class, |speed|) for the last second
        self._back = deque(maxlen=int(BACK_S * FS))   # 1 where the network says backward
        self._shown = deque(maxlen=60)        # committed contacts for the plot
        self.contact_log = []
        self.frame.unlock()                   # decide forward again from the next walk

    def break_stream(self):
        self.reset_stream()

    def set_height(self, height_m):
        self.height = float(height_m)

    # --- data ---------------------------------------------------------------------------

    def feed(self, accel, gyro):
        """New raw samples: (n, 3) g and (n, 3) deg/s, sensor axes."""
        ch = self.frame.process(accel, gyro)
        for row in ch:
            self._chan.append(row)
        x = np.hstack([ch, np.full((len(ch), 1), self.height)])
        self._pending = np.vstack([self._pending, x])
        while len(self._pending) >= BLOCK:
            block, self._pending = self._pending[:BLOCK], self._pending[BLOCK:]
            self._inputs = np.vstack([self._inputs, block])[-(self.ctx + BLOCK):]
            self._filled += BLOCK
            self._n_in += BLOCK
            if self._filled < self.ctx + BLOCK:
                continue                       # not enough context yet: outputs would be guesses
            out = self.net.forward(self._inputs)[-BLOCK:]
            self._consume(out)

    def _consume(self, out):
        prob = _sigmoid(out[:, 0])
        cls = _softmax(out[:, 1:5]).argmax(axis=1)
        speed = out[:, 5]
        start = self._n_in - BLOCK - self.D        # label sample of the first output
        if self._peak_from is None:
            self._peak_from = start
        for i in range(BLOCK):
            moving = cls[i] != 0
            v = abs(float(speed[i])) if moving else 0.0
            self.distance += v / FS
            self._back.append(1 if cls[i] == 3 else 0)
            if moving and self.backward_now:
                self.backward += v / FS
            self._recent.append((int(cls[i]), v))
            self._prob.append(float(prob[i]))
            self._cls.append(int(cls[i]))
        self._n_label = start + BLOCK
        self._find_contacts()
        if self._other_foot_due is not None and self._n_label > self._other_foot_due:
            self.steps += 1
            self._other_foot_due = None

    def _find_contacts(self):
        """Local maxima of the contact probability above THRESHOLD, at least
        MIN_GAP apart; a peak is final once MIN_GAP/2 samples after it are in."""
        half = MIN_GAP // 2
        base = self._n_label - len(self._prob)
        p = np.array(self._prob)
        last_final = self._n_label - 1 - half
        i = max(self._peak_from, base + half)
        while i <= last_final:
            k = i - base
            if p[k] >= THRESHOLD and p[k] == p[k - half:k + half + 1].max():
                if self._last_contact is None or i - self._last_contact >= MIN_GAP:
                    self._contact(i)
            i += 1
        self._peak_from = last_final + 1

    def _contact(self, i):
        base = self._n_label - len(self._cls)
        moving = self._cls[i - base] != 0 if 0 <= i - base < len(self._cls) else False
        if self._last_contact is not None and i - self._last_contact > MAX_GAP:
            self._end_bout()
        self._last_contact = i
        self._bout.append(i)
        if self._confirmed:
            self._commit([i])
            return
        if not moving:
            self._bout = [i]
            return
        if len(self._bout) >= self.bout_min:
            gaps = np.diff(self._bout[-self.bout_min:])
            ok = all(MIN_GAP <= g <= MAX_GAP for g in gaps) and all(
                RATIO[0] <= b / a <= RATIO[1] for a, b in zip(gaps[:-1], gaps[1:]))
            if ok:
                self._confirmed = True
                self._commit(self._bout[-self.bout_min:])
            else:
                self._bout = self._bout[-(self.bout_min - 1):]

    def _commit(self, contacts):
        """Count contacts of this foot: 1 step each now, and the other foot's
        step half a stride later (paid at once if the next contact comes first)."""
        for c in contacts:
            if self._other_foot_due is not None:
                self.steps += 1
            prev = self._shown[-1] if self._shown else None
            gap = c - prev if prev is not None and c - prev <= MAX_GAP else 100
            self.steps += 1
            self._other_foot_due = c + gap // 2
            self._stride_dist.append(self.distance - self._dist_at_contact)
            self._dist_at_contact = self.distance
            self._shown.append(c)
            self.contact_log.append(c)     # campus-nav: read and cleared by main.py (heading strides)

    def _end_bout(self):
        self._bout = []
        self._confirmed = False

    # --- for the page -------------------------------------------------------------------

    @property
    def backward_now(self) -> bool:
        """Walking backwards for real, not just a turn: most of the last 3 s."""
        return len(self._back) == self._back.maxlen and sum(self._back) >= BACK_SHARE * len(self._back)

    def counts(self) -> dict:
        recent = list(self._recent)
        if recent:
            classes = [c for c, _ in recent]
            activity = CLASSES[max(set(classes), key=classes.count)]
            moving = [v for c, v in recent if c != 0]
            speed = float(np.mean(moving)) if moving else 0.0
        else:
            activity, speed = "standing", 0.0
        if self.backward_now and activity != "standing":
            activity = "backward"
        elif activity == "backward":
            activity = "walking"                     # a brief backward look: turning
        if activity == "backward":
            speed = -speed
        cadence = None
        if self._confirmed and len(self._bout) >= 2:
            stride = np.median(np.diff(self._bout[-5:])) / FS
            cadence = 120.0 / stride                       # steps per minute
        last = self._stride_dist[-1] if self._confirmed and len(self._stride_dist) > 1 else None
        return {"steps": int(self.steps), "distance_m": round(self.distance, 2),
                "backward_m": round(self.backward, 2), "activity": activity,
                "cadence_spm": None if cadence is None else round(cadence, 1),
                "speed_mps": round(speed, 2), "last_stride_m": None if last is None else round(last, 2)}

    def live(self, n: int) -> dict:
        """The last n input samples in the shank frame, the contact probability
        lined up with them (None where it isn't known yet), and counted contacts."""
        chan = np.array(list(self._chan)[-n:]) if self._chan else np.zeros((0, 7))
        m = len(chan)
        end = self._n_in + len(self._pending)                # input sample just after the window
        start = end - m
        base = self._n_label - len(self._prob)
        prob = [None] * m
        p = list(self._prob)
        for k in range(m):
            j = start + k - base
            if 0 <= j < len(p):
                prob[k] = round(p[j], 3)
        contacts = [c - start for c in self._shown if start <= c < end]
        return {"frame": {"accel": np.round(chan[:, 0:3].T, 3).tolist(),
                          "gyro": np.round(chan[:, 3:6].T, 1).tolist(),
                          "axes": ["AP", "ML", "long"]},
                "prob": prob, "contacts": contacts}

    def info(self) -> dict:
        m = self.net.meta
        return {"ready": True, "selftest_err": self.selftest_err,
                "describe": m.get("describe", "Neural network trained on simulated legs"), **m}
