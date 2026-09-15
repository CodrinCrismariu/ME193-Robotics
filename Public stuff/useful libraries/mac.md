# Running the tricycle robot on macOS

Everything in [TRIKE.md](TRIKE.md) applies — the kinematics, the calibration
values, the controls. This file covers only the places where macOS differs from
the Windows setup that document was written against.

Verified on macOS 15 (Apple Silicon), Python 3.12.7, with both motors connected
over Bluetooth.

---

## Camera handling is built into `hand_control.py`

`hand_control.py` used to open the camera with `cv2.CAP_DSHOW` — DirectShow,
which is Windows-only. On macOS it could never open, and the script died with:

```
cannot open camera 0
Future exception was never retrieved
ConnectionError: Device disconnected unexpectedly
```

The second error was a knock-on effect: the `raise SystemExit` for the camera
happened *before* the `try`/`finally` that disconnects the motors, so the robot
was left connected and live when the script bailed.

Those three Windows assumptions are now fixed in `hand_control.py` itself,
guarded per platform, so one script runs everywhere:

| Assumption | What it does now |
| --- | --- |
| `CAP_DSHOW` backend | picks the backend per platform (`CAP_ANY` off Windows) |
| camera index `0` is built-in | selects the built-in camera by *device type* |
| camera failure exits uncleanly | disconnects the motors on that path |

`hand_control_mac.py` is kept as a one-line shim that calls into
`hand_control.py`, so the commands below still work either way. Prefer
`hand_control.py` in anything new.

---

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate

pip install legoeducation
pip install opencv-python "mediapipe==0.10.35"
pip install pyobjc-framework-AVFoundation   # optional, see below

curl -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Check it:

```bash
python -c "import legoeducation, cv2, mediapipe; print('ok')"
```

### Pin MediaPipe to 0.10.x

**MediaPipe 1.x aborts on Apple Silicon.** The moment `HandLandmarker` is
created — in every running mode, with either delegate — it dies with:

```
F graph_service.h:139] Check failed: service_ Service is unavailable.
    @ -[DrishtiMetalHelper initWithCalculatorContext:]
    @ mediapipe::api2::TensorsToDetectionsCalculator::Open()
```

That is a fatal abort, not a Python exception, so it cannot be caught or worked
around in code. `0.10.35` exposes the same Tasks API the scripts use, so nothing
else has to change.

### `pyobjc-framework-AVFoundation` is optional

It is only used to find the built-in camera by name. Without it the code falls
back to index 0, which is usually right — just not guaranteed (see below).

### The virtual environment is not portable

A `.venv` committed from Windows (`Scripts/`, `Lib/`, `home = C:\Python312`)
cannot run here. Build a fresh one with the commands above; `.gitignore` already
excludes it.

---

## Camera

### A black preview window is a permissions problem

macOS does not raise an error when camera access is denied — it hands the process
a stream of **all-zero frames**. The window opens, the loop runs, everything
looks fine, and the picture is black.

Grant the camera to the app that *owns the terminal* — Visual Studio Code,
Terminal, iTerm — **not** Python, which never appears in the list. It is under
System Settings → Privacy & Security → Camera. Then **fully quit and reopen that
app**: the permission is only re-read at launch, which is the step most people
miss.

To tell a permissions problem from a merely dark room:

```bash
python -c "
import cv2; cap = cv2.VideoCapture(0, cv2.CAP_ANY)
for _ in range(30): ok, f = cap.read()
print('mean:', f.mean(), 'max:', f.max())"
```

- `mean` and `max` both exactly `0` → permission denied.
- small but nonzero, creeping up over successive frames → real sensor data in a
  dark scene. The camera works; add light. MediaPipe needs contrast to find a
  hand.

### Continuity Camera shifts the indices

A nearby iPhone appears as extra capture devices:

```
index 0: FaceTime HD Camera            (built-in)
index 1: <name>'s iPhone Camera        (Continuity)
index 2: <name>'s iPhone Desk View Camera
```

The order is not guaranteed and changes as the phone connects and disconnects,
so a hardcoded `0` is not dependably the Mac's own camera. `hand_control.py`
picks the built-in one by device type instead. Override with `--camera N`.

---

## Bluetooth

No macOS-specific setup. `legoeducation` pulls the CoreBluetooth backend
automatically, and the first run triggers the standard Bluetooth permission
prompt. Confirm both motors are visible before driving:

```bash
python scan_tmp.py
```

```
2 LEGO device(s) in range:
  '🟩 0994 Single Motor'   product_id=0x0200  card_serial=0994
  '🟩 0994 Double Motor'   product_id=0x0201  card_serial=0994
```

---

## Versions

`legoeducation` 1.1.1 · `bleak` 3.0.2 · `opencv-python` 5.0.0.93 ·
`mediapipe` 0.10.35 · `numpy` 2.5.3 · Python 3.12.7
