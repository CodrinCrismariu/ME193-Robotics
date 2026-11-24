import json
import threading
import time

import numpy as np
from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import *
from fastapi.responses import Response

import heading
import nav
from walks import DATA_DIR, WAYPOINTS, WalkRecorder

# Campus navigator: labelled walks, and navigation by behaviour cloning on them (nav.py).
#
# Built on the step counter (stream.py, the same network and code as the
# step-counter app): the sketch streams the ankle IMU at 100 Hz, a worker
# thread wakes every 0.25 s and feeds new samples to
#   stream.Pipeline   steps, distance, activity, foot contacts
#   heading.Heading   heading, 90-degree turns, per-stride features
# The contacts the step counter commits are passed on to the heading tracker,
# which cuts strides at them. While a walk is being recorded, its waypoint
# labels, counts, turns and strides go to walks.WalkRecorder.
#
# GET /walks.zip   every saved walk (data/walks/)

FS = 100.0
TICK_S = 0.25
STATS_INTERVAL = 5.0
SETTINGS = DATA_DIR / "settings.json"
DEFAULTS = {"walker": "", "height_m": 1.75}

web_ui = WebUI()
walks = WalkRecorder(FS)
tracker = heading.Heading()
policy = nav.Policy()
navigator = nav.Navigator(policy)
_distance = 0.0                          # the step counter's distance at the last update

_lock = threading.Lock()
_pending = []                            # batches not yet processed (None = stream broke)
_expected_seq = None
_received = 0
_gaps = 0
_mcu_status = "waiting for the MCU"
_work_ms = None
_recent_turns = []                       # last few turns for the page, any time


def _load_settings() -> dict:
    try:
        return {**DEFAULTS, **json.loads(SETTINGS.read_text())}
    except (OSError, ValueError):
        return dict(DEFAULTS)


_settings = _load_settings()


def _save_settings():
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS.with_suffix(".tmp")
    tmp.write_text(json.dumps(_settings))
    tmp.replace(SETTINGS)


def _load_pipeline():
    try:
        import stream
        return stream.Pipeline(height_m=_settings["height_m"])
    except Exception as exc:
        print(f"[nav] couldn't load the step counter: {exc!r}")
    return None


pipeline = _load_pipeline()


def _send_walks(error: str | None = None, room: str | None = None):
    snap = walks.snapshot()
    policy.fit(snap["walks"])                     # cheap: refit on every change to the walks
    snap["settings"] = _settings
    snap["nav_options"] = {w: policy.reachable(w) for w in WAYPOINTS}
    if error:
        snap["message"] = error
    web_ui.send_message("walks", snap, room=room)


def imu_batch(seq: int, data: str):
    """A run of consecutive samples from the sketch: 6 int16 each, mg and 0.1 deg/s."""
    global _expected_seq, _received, _gaps
    raw = np.frombuffer(bytes.fromhex(data), dtype="<i2").reshape(-1, 6)
    with _lock:
        if _expected_seq is not None and seq != _expected_seq:
            if seq > _expected_seq:
                _gaps += 1
                walks.lost(seq - _expected_seq)
                print(f"[nav] {seq - _expected_seq} samples lost on the link")
            else:
                print("[nav] MCU restarted")
            _pending.append(None)
        _expected_seq = seq + len(raw)
        _received += len(raw)
        _pending.append(raw)
    walks.add_raw(raw)


def imu_status(message: str):
    global _mcu_status
    _mcu_status = message
    print(f"[mcu] {message}")


def _process(batches):
    global _distance
    for batch in batches:
        if batch is None:
            if pipeline is not None:
                pipeline.break_stream()
            continue
        accel, gyro = batch[:, :3] / 1000.0, batch[:, 3:] / 10.0
        tracker.process(accel, gyro)
        if pipeline is not None:
            pipeline.feed(accel, gyro)
    if pipeline is not None:
        for c in pipeline.contact_log:
            tracker.add_contact(c + 1)          # label sample c is tracker sample c + 1
        pipeline.contact_log.clear()
    counts = pipeline.counts() if pipeline is not None else None
    if counts is not None:
        _distance = counts["distance_m"]
    turns, strides = tracker.pop()
    for t in turns:
        walks.turn(t)
        navigator.on_turn(t["angle"], _distance)
        _recent_turns.append({"angle": t["angle"], "at": time.time()})
        del _recent_turns[:-5]
    for s in strides:
        walks.stride(s)
    if counts is not None:
        walks.tick(pipeline._n_label, counts["steps"], counts["distance_m"])
    return counts


def _worker():
    """Every TICK_S: process new samples, then update the page. Nothing here
    may end the thread, or the page would freeze."""
    global _work_ms
    while True:
        time.sleep(TICK_S)
        try:
            with _lock:
                batches = _pending[:]
                _pending.clear()
            t0 = time.perf_counter()
            counts = _process(batches)
            _work_ms = (time.perf_counter() - t0) * 1000.0
            web_ui.send_message("live", {
                "counts": counts, "heading": round(tracker.relative_heading(), 1),
                "moving": tracker.moving, "turns": _recent_turns,
                "calibrated": tracker.calibrated, "since_still": tracker.seconds_since_still(),
                "walk": walks.snapshot()["active"], "nav": navigator.update(_distance)})
        except Exception as exc:
            print(f"[nav] worker: {exc!r}")


def _stats():
    global _received
    while True:
        time.sleep(STATS_INTERVAL)
        with _lock:
            rate = _received / STATS_INTERVAL
            _received = 0
            gaps = _gaps
        web_ui.send_message("status", {"rate": rate, "gaps": gaps, "mcu": _mcu_status, "work_ms": _work_ms,
                                       "model": pipeline is not None})


# --- page commands ----------------------------------------------------------------------

def _arg(data, key, default=None):
    return data.get(key, default) if isinstance(data, dict) else default


def on_connect(sid: str):
    _send_walks(room=sid)


def on_settings(sid: str, data):
    try:
        height = float(_arg(data, "height_m"))
    except (TypeError, ValueError):
        height = None
    if height is None or not 1.2 <= height <= 2.2:
        _send_walks("Height must be between 1.20 and 2.20 m", room=sid)
        return
    _settings["walker"] = str(_arg(data, "walker", ""))[:40]
    _settings["height_m"] = round(height, 2)
    _save_settings()
    if pipeline is not None:
        pipeline.set_height(height)
    _send_walks()


def on_reset(sid: str, data):
    """Zero steps, distance and turns (not while a walk is being recorded)."""
    if walks.active is not None:
        _send_walks("Stop the recording first", room=sid)
        return
    if pipeline is not None:
        pipeline.reset_counts()
    tracker.reset_turns()
    _recent_turns.clear()
    _send_walks()


def on_walk_start(sid: str, data):
    if navigator.active:
        _send_walks("Stop navigating first", room=sid)
        return
    if not tracker.calibrated:
        _send_walks("Stand still for a few seconds first (measuring the gyro's drift)", room=sid)
        return
    with _lock:                                    # nothing waiting: the walk starts at the current sample
        n_now = tracker.n + sum(len(b) for b in _pending if b is not None)
    error = walks.start(_settings["walker"], _settings["height_m"], _arg(data, "loc"), n_now)
    if error is None:
        tracker.reset_turns()
    _send_walks(error)


def on_nav_start(sid: str, data):
    if walks.active is not None:
        _send_walks("Stop the recording first", room=sid)
        return
    if not tracker.calibrated:
        _send_walks("Stand still for a few seconds first (measuring the gyro's drift)", room=sid)
        return
    tracker.reset_turns()
    _send_walks(navigator.start(_arg(data, "here"), _arg(data, "goal"), _distance))


def on_nav_at(sid: str, data):
    """You told it where you are: plan again from there."""
    loc = _arg(data, "loc")
    if loc not in WAYPOINTS:
        return
    tracker.reset_turns()
    _send_walks(navigator.at(loc, _distance))


def on_nav_stop(sid: str, data):
    navigator.stop()
    _send_walks()


def on_walk_label(sid: str, data):
    with _lock:
        n_now = tracker.n + sum(len(b) for b in _pending if b is not None)
    error = walks.label(_arg(data, "loc"), n_now)
    if error is None:
        tracker.reset_turns()                      # you stop at a waypoint: a new straight from here
    _send_walks(error)


def on_walk_undo(sid: str, data):
    _send_walks(walks.undo_label())


def on_walk_stop(sid: str, data):
    _send_walks(walks.stop(save=bool(_arg(data, "save", True))))


def on_turn_edit(sid: str, data):
    _send_walks(walks.edit_turn(_arg(data, "id"), _arg(data, "index"), _arg(data, "op")))


def on_label_edit(sid: str, data):
    _send_walks(walks.edit_label(_arg(data, "id"), _arg(data, "index"), _arg(data, "op"), _arg(data, "loc")))


def on_walk_delete(sid: str, data):
    _send_walks(walks.delete(_arg(data, "id")))


def get_walks_zip():
    stamp = time.strftime("%Y-%m-%d_%H-%M")
    return Response(walks.zip_bytes(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="campus-nav-walks_{stamp}.zip"'})


web_ui.on_connect(on_connect)
web_ui.on_message("settings", on_settings)
web_ui.on_message("reset", on_reset)
web_ui.on_message("walk_start", on_walk_start)
web_ui.on_message("walk_label", on_walk_label)
web_ui.on_message("nav_start", on_nav_start)
web_ui.on_message("nav_at", on_nav_at)
web_ui.on_message("nav_stop", on_nav_stop)
web_ui.on_message("walk_undo", on_walk_undo)
web_ui.on_message("walk_stop", on_walk_stop)
web_ui.on_message("turn_edit", on_turn_edit)
web_ui.on_message("label_edit", on_label_edit)
web_ui.on_message("walk_delete", on_walk_delete)
web_ui.expose_api("GET", "/walks.zip", get_walks_zip)

Bridge.provide("imu_batch", imu_batch)
Bridge.provide("imu_status", imu_status)

threading.Thread(target=_worker, name="worker", daemon=True).start()
threading.Thread(target=_stats, name="stats", daemon=True).start()

App.run()
