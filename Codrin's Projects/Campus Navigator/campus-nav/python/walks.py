"""Labelled walks: recording, storage, edits, and turning a walk into legs.

A walk starts at one of the waypoints ("I'm at ..."), and every time you get
to another one you tap "Reached ...". Nothing else is labelled by hand: the
turns come from heading.py and the distances from the step counter. Each
stretch between two waypoints is a leg:

    In front of class -> Down stairs:  forward 23.4 m, turn left, forward 8.1 m

Stored per walk in data/walks/:
    <id>.json   walker, height, labels [{t, loc}], ticks (t, steps, distance)
                at 4 Hz, turns [{t, angle, measured, auto}], strides (stairs
                features, see heading.FEATURES), edits are applied in place
    <id>.csv    the raw IMU (t_s, ax_g ... gz_dps), for re-processing later
Times t are seconds since the walk started (sample count / FS). On the board
data/ is /home/arduino/ArduinoApps/campus-nav/data; deploying never touches it.
"""
import io
import json
import re
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WALKS = DATA_DIR / "walks"
WAYPOINTS = ["In front of class", "Up stairs", "Down stairs", "In front of Nolop", "In Nolop"]
MAX_S = 30 * 60
_ID = re.compile(r"^walk_[\w-]+$")
_HEADER = "t_s,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps"


class WalkRecorder:
    """The walk in progress (one at a time) and the saved ones."""

    def __init__(self, fs: float):
        self.fs = fs
        self._lock = threading.Lock()
        self.active = None                    # the walk being recorded (dict) or None
        self._raw = []
        self._samples = 0
        self._n0 = 0
        self.saved = {}                       # id -> summary
        self._scan()

    # --- the walk in progress ----------------------------------------------------------

    def start(self, walker: str, height_m: float, loc: str, n_now: int) -> str | None:
        if loc not in WAYPOINTS:
            return "Pick a waypoint"
        with self._lock:
            if self.active is not None:
                return "Already recording a walk"
            stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.active = {"id": f"walk_{stamp}", "created": time.time(), "walker": walker[:40] or "?",
                           "height_m": height_m, "labels": [{"t": 0.0, "loc": loc}],
                           "ticks": {"t": [], "steps": [], "distance": []},
                           "turns": [], "strides": [], "lost_samples": 0}
            self._raw = []
            self._samples = 0
            self._n0 = n_now
            self._steps0 = None
            self._dist0 = None
        return None

    def t(self, n: int) -> float:
        return round((n - self._n0) / self.fs, 2)

    def label(self, loc: str, n: int) -> str | None:
        with self._lock:
            if self.active is None:
                return "No walk in progress"
            if loc not in WAYPOINTS:
                return "Unknown waypoint"
            if loc == self.active["labels"][-1]["loc"]:
                return f"You're already at {loc}"
            self.active["labels"].append({"t": self.t(n), "loc": loc})
        return None

    def undo_label(self) -> str | None:
        with self._lock:
            if self.active is None or len(self.active["labels"]) < 2:
                return "Nothing to undo"
            self.active["labels"].pop()
        return None

    def add_raw(self, raw: np.ndarray):
        with self._lock:
            if self.active is not None and self._samples < MAX_S * self.fs:
                self._raw.append(raw.copy())
                self._samples += len(raw)

    def lost(self, n: int):
        with self._lock:
            if self.active is not None:
                self.active["lost_samples"] += n

    def tick(self, n_label: int, steps: int, distance: float):
        """Counts from the step counter; n_label is the sample they describe."""
        with self._lock:
            w = self.active
            if w is None or n_label < self._n0:
                return
            if self._steps0 is None:
                self._steps0, self._dist0 = steps, distance
            w["ticks"]["t"].append(self.t(n_label))
            w["ticks"]["steps"].append(int(steps - self._steps0))
            w["ticks"]["distance"].append(round(distance - self._dist0, 2))

    def turn(self, ev: dict):
        with self._lock:
            if self.active is not None and ev["n"] >= self._n0:
                self.active["turns"].append({"t": self.t(ev["n"]), "angle": ev["angle"],
                                             "measured": ev["measured"], "auto": ev["angle"]})

    def stride(self, st: dict):
        with self._lock:
            if self.active is not None and st["n0"] >= self._n0:
                self.active["strides"].append({**{k: v for k, v in st.items() if k not in ("n", "n0")},
                                               "t0": self.t(st["n0"]), "t1": self.t(st["n"])})

    def stop(self, save: bool = True) -> str | None:
        with self._lock:
            w, raw, self.active, self._raw = self.active, self._raw, None, []
        if w is None:
            return "No walk in progress"
        if not save:
            return None
        if len(w["labels"]) < 2:
            return "Not saved: tap \"Reached\" at least once"
        WALKS.mkdir(parents=True, exist_ok=True)
        data = np.vstack(raw) if raw else np.zeros((0, 6))
        t = np.arange(len(data)) / self.fs
        rows = np.column_stack([t, data[:, :3] / 1000.0, data[:, 3:] / 10.0])
        buf = io.StringIO()
        np.savetxt(buf, rows, fmt=["%.2f", "%.3f", "%.3f", "%.3f", "%.1f", "%.1f", "%.1f"],
                   delimiter=",", header=_HEADER, comments="")
        self._write(WALKS / f"{w['id']}.csv", buf.getvalue())
        self._save(w)
        return None

    # --- saved walks ---------------------------------------------------------------------

    def load(self, walk_id: str) -> dict | None:
        if not isinstance(walk_id, str) or not _ID.match(walk_id):
            return None
        try:
            return json.loads((WALKS / f"{walk_id}.json").read_text())
        except (OSError, ValueError):
            return None

    def edit_turn(self, walk_id: str, index: int, op: str) -> str | None:
        """op: flip (left <-> right), delete, left, right, around."""
        w = self.load(walk_id)
        if w is None:
            return "No such walk"
        if not isinstance(index, int) or not 0 <= index < len(w["turns"]):
            return "No such turn"
        turn = w["turns"][index]
        if op == "delete":
            w["turns"].pop(index)
        elif op == "flip":
            turn["angle"] = -turn["angle"]
        elif op in ("left", "right", "around"):
            turn["angle"] = {"left": 90, "right": -90}.get(op, 180 if turn["angle"] >= 0 else -180)
        else:
            return "Unknown edit"
        self._save(w)
        return None

    def edit_label(self, walk_id: str, index: int, op: str, loc: str | None = None) -> str | None:
        """op: rename (to loc) or delete (merges the two legs around it)."""
        w = self.load(walk_id)
        if w is None:
            return "No such walk"
        if not isinstance(index, int) or not 0 <= index < len(w["labels"]):
            return "No such label"
        if op == "rename" and loc in WAYPOINTS:
            w["labels"][index]["loc"] = loc
        elif op == "delete" and index > 0 and len(w["labels"]) > 2:
            w["labels"].pop(index)
        else:
            return "Can't do that"
        self._save(w)
        return None

    def delete(self, walk_id: str) -> str | None:
        if self.load(walk_id) is None:
            return "No such walk"
        for suffix in (".json", ".csv"):
            (WALKS / f"{walk_id}{suffix}").unlink(missing_ok=True)
        with self._lock:
            self.saved.pop(walk_id, None)
        return None

    def zip_bytes(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(WALKS.glob("walk_*")):
                z.write(path, f"campus-nav-walks/{path.name}")
        return buf.getvalue()

    def snapshot(self) -> dict:
        with self._lock:
            w = self.active
            active = None if w is None else {
                "id": w["id"], "walker": w["walker"], "labels": w["labels"],
                "turns": w["turns"], "elapsed": round(self._samples / self.fs, 1),
                "distance": w["ticks"]["distance"][-1] if w["ticks"]["distance"] else 0.0,
                "steps": w["ticks"]["steps"][-1] if w["ticks"]["steps"] else 0,
                "legs": legs(w)}
            saved = sorted(self.saved.values(), key=lambda s: s["created"], reverse=True)
        return {"waypoints": WAYPOINTS, "active": active, "walks": saved}

    # --- internals ------------------------------------------------------------------------

    def _save(self, w: dict):
        WALKS.mkdir(parents=True, exist_ok=True)
        self._write(WALKS / f"{w['id']}.json", json.dumps(w))
        with self._lock:
            self.saved[w["id"]] = _summary(w)

    @staticmethod
    def _write(path: Path, text: str):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(path)

    def _scan(self):
        for path in WALKS.glob("walk_*.json"):
            try:
                w = json.loads(path.read_text())
                self.saved[w["id"]] = _summary(w)
            except (OSError, ValueError, KeyError) as exc:
                print(f"[walks] skipping {path.name}: {exc!r}")


# --- legs -------------------------------------------------------------------------------

def _interp(ticks: dict, key: str, t: float) -> float:
    ts = ticks["t"]
    if not ts:
        return 0.0
    return float(np.interp(t, ts, ticks[key]))


def legs(w: dict) -> list:
    """The walk cut at its waypoint labels; each leg a list of directions:
    {"do": "forward", "m": metres} / {"do": "left"} / {"do": "right"}.
    A U-turn (+/-180) is two turns the same way."""
    out = []
    labels = w["labels"]
    turns = sorted(w["turns"], key=lambda e: e["t"])
    ticks = w["ticks"]
    for i in range(len(labels) - 1):
        a, b = labels[i], labels[i + 1]
        items, t_prev = [], a["t"]
        for k, turn in enumerate(turns):
            if not a["t"] < turn["t"] < b["t"]:
                continue
            m = _interp(ticks, "distance", turn["t"]) - _interp(ticks, "distance", t_prev)
            items.append({"do": "forward", "m": round(m, 1)})
            side = "left" if turn["angle"] > 0 else "right"
            for _ in range(abs(turn["angle"]) // 90):
                items.append({"do": side, "turn": k})
            t_prev = turn["t"]
        m = _interp(ticks, "distance", b["t"]) - _interp(ticks, "distance", t_prev)
        items.append({"do": "forward", "m": round(m, 1)})
        items = [it for it in items if it["do"] != "forward" or it["m"] >= 0.5 or it is items[-1]]
        out.append({"from": a["loc"], "to": b["loc"], "t0": a["t"], "t1": b["t"],
                    "distance": round(_interp(ticks, "distance", b["t"]) - _interp(ticks, "distance", a["t"]), 1),
                    "steps": int(round(_interp(ticks, "steps", b["t"]) - _interp(ticks, "steps", a["t"]))),
                    "items": items})
    return out


def _summary(w: dict) -> dict:
    return {"id": w["id"], "created": w["created"], "walker": w["walker"], "height_m": w["height_m"],
            "seconds": w["ticks"]["t"][-1] if w["ticks"]["t"] else 0.0,
            "labels": w["labels"], "turns": w["turns"], "legs": legs(w),
            "lost_samples": w.get("lost_samples", 0)}
