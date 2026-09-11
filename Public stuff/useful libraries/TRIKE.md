# Tricycle robot — setup, theory, and usage

A LEGO Education robot driven by two fixed traction wheels (a **Double Motor**)
plus a third free-spinning wheel whose angle is set by a **Single Motor**. It can
be driven from a keyboard panel or by waving your hand at a webcam.

Everything here is calibrated to one specific robot. The measured values are
listed in [Calibration](#calibration-values) — if you rebuild the robot, those
are the numbers to redo.

---

## Contents

| File | What it is |
| --- | --- |
| `trike.py` | The core library: kinematics, steering, motor connection |
| `robot_ui.py` | Pop-up control panel (keyboard + mouse) |
| `hand_control.py` | Webcam hand control |
| `hand_control_mac.py` | Webcam hand control, macOS variant |
| `hand_landmarker.task` | MediaPipe hand model — **not in git**, [download it](#install) |
| `lelib.py` | Class-wide wrapper around `legoeducation` |
| `scan_tmp.py` | Lists every LEGO device in Bluetooth range with its card |
| `read_abs.py` | Reads the steering motor's absolute encoder (for recalibration) |
| `steer_test.py` | Steering accuracy check, no driving |
| `latency.py` | Measures command-to-motion delay |

Leftovers from an abandoned self-balancing attempt — `balance.py`,
`calibrate_gyro.py`, `diag.py`, `axis_test.py`, `bench_loop.py`, `yaw_test.py` —
are kept because they document why balancing did not work. See
[Why there is no self-balancing](#why-there-is-no-self-balancing).

---

## Install

Requires **Python 3.12**. Neither the virtual environment nor the hand model is
stored in git (both are large binaries, and `.venv` contains a 107 MB file that
GitHub refuses outright), so a fresh clone needs all three steps below.

**1. Create a virtual environment** — from this folder:

```powershell
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
```

**2. Install the packages:**

```powershell
pip install legoeducation            # required -- talks to the motors
pip install opencv-python mediapipe  # only for hand_control.py
```

**On macOS (Apple Silicon), pin MediaPipe to the 0.10 line instead:**

```bash
pip install opencv-python "mediapipe==0.10.35"
```

MediaPipe 1.x aborts on arm64 Macs the moment `HandLandmarker` is created, in
every running mode and with either delegate:

```
F graph_service.h:139] Check failed: service_ Service is unavailable.
    @ -[DrishtiMetalHelper initWithCalculatorContext:]
    @ mediapipe::api2::TensorsToDetectionsCalculator::Open()
```

It is a fatal abort, not a Python exception, so it cannot be caught. 0.10.35
exposes the same Tasks API `hand_control.py` uses, so no code change is needed.

**3. Download the hand model** — required by `hand_control.py`, which will not
start without it. It must sit in this folder, next to the script:

```powershell
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

That file is ~7.8 MB. If it is missing you will get an error from
`HandLandmarker.create_from_options` about the model asset path.

Check the install worked:

```powershell
python -c "import legoeducation, cv2, mediapipe; print('ok')"
```

Versions this was built against: `legoeducation` 1.1.1, `opencv-python` 5.0.0.93,
`mediapipe` 1.0.1 on Windows, `mediapipe` 0.10.35 on macOS (see above).

> `robot_ui.py` and `trike.py` need only step 1 and `legoeducation`. OpenCV,
> MediaPipe and the model are exclusively for `hand_control.py`.

> **MediaPipe 1.x removed `mp.solutions`.** Older tutorials use
> `mp.solutions.hands`, which no longer exists. `hand_control.py` uses the Tasks
> API (`HandLandmarker`), which is why the model file is a separate download.

### Before running anything

1. Power on **both** motor units.
2. Enable Bluetooth.
3. Check they are visible:

```powershell
python scan_tmp.py
```

Expected:

```
2 LEGO device(s) in range:
  🟩 0994 Single Motor    product_id=0x0200  card_color=5  card_serial=0994
  🟩 0994 Double Motor    product_id=0x0201  card_color=5  card_serial=0994
```

---

## Running it

### Keyboard panel

```powershell
python robot_ui.py
```

| Key | Action |
| --- | --- |
| `W` / `↑` | forward |
| `S` / `↓` | backward |
| `A` / `←` | steer left |
| `D` / `→` | steer right |
| `Space` | stop |
| `Esc` | quit |

Hold to drive, release to stop — a dead-man switch. Steering sweeps while held
and springs back to centre when released, like a car's wheel. The slider sets top
speed; the readout shows the real steering angle and both wheel speeds.

**Click the window first** so it has keyboard focus. The on-screen arrows work
with the mouse regardless.

### Hand control

```powershell
python hand_control.py             # drives the robot
python hand_control.py --no-robot  # camera only, nothing moves
python hand_control.py --camera 1  # a different webcam
```

On macOS use `hand_control_mac.py`, which carries the platform fixes described
below; `hand_control.py` opens the camera with a Windows-only backend and will
fail with "cannot open camera 0".

In the macOS variant, with no `--camera` the built-in camera is selected by
device type rather than by index. Continuity Camera makes a nearby iPhone appear
as an extra capture device, and the index order shifts as the phone connects and
disconnects, so a hardcoded `0` is not reliably the Mac's own camera. Pass
`--camera N` to override. The lookup needs `pyobjc-framework-AVFoundation`:

```bash
pip install pyobjc-framework-AVFoundation   # macOS only, optional
```

Without it the code falls back to index 0.

**If the preview window is black on macOS**, the camera is streaming but the
frames are all zeros -- that is how macOS reports a denied camera permission; it
does not raise an error. Grant the camera to the app that *owns* the terminal
(Visual Studio Code, Terminal, iTerm -- not Python) under System Settings ->
Privacy & Security -> Camera, then **fully quit and reopen that app**, since the
permission is only re-read at launch.

Hold one hand up. Relative to the centre of the frame:

- **up** → forward, **down** → backward
- **left** → steer left, **right** → steer right

Both axes are live at once, so up-and-left is a fast left turn. Output is
continuous — the further from centre, the more speed or steering.

`Space` arms/disarms, `Q` quits. **It starts disarmed**, so you can position your
hand before anything moves.

### Scripted driving

```python
from trike import Trike

t = Trike().connect()
t.forward(40)              # straight ahead
t.turn(40, 30)             # 30° left, wheel speeds matched to the arc
t.turn(40, -30)            # right
t.turn_radius(40, 400)     # trace a 400 mm radius
t.drive_at(60, 15)         # the general form: speed + steering angle
t.stop()
t.disconnect()
```

Positive steering angles are **left**. Speed is a percentage, -100 to 100.

---

## How it works

### The no-slip condition

This is the whole point of `trike.py`. Put the drive axle at the origin, `x`
forward, `y` left, track width `W`, steered wheel `L` ahead on the centreline:

```
      (0, +W/2)  left drive wheel
         |
         |------ (L, 0)  steered wheel
         |
      (0, -W/2)  right drive wheel
```

The two drive wheels are bolted to the frame, so they can only roll along `x`.
That forces the **instantaneous centre of rotation** (ICR) — the single point the
whole robot pivots about at any instant — to lie somewhere on their shared axle
line, at `(0, R)`.

Every wheel must circle that same ICR. If any wheel is pointed or driven
inconsistently with it, that wheel scrubs sideways across the floor.

The steered wheel sits at `(L, 0)`, so its radius vector from the ICR is
`(L, -R)`, and it must point perpendicular to that:

```
tan(δ) = L / R          →     R = L / tan(δ)
```

With the body rotating at ω about the ICR, the midpoint moves at `v = ω·R` and
each drive wheel travels at its own radius:

```
v_left  = ω·(R − W/2) = v·(1 − W·tan(δ) / 2L)
v_right = ω·(R + W/2) = v·(1 + W·tan(δ) / 2L)
```

Written that way, `R` never has to be computed and there is no singularity at
δ = 0. For this robot `W/2L = 70/170 = 0.412`:

| Steer | v_left | v_right | Turn radius |
| --- | --- | --- | --- |
| 0° | 50.0 | 50.0 | straight |
| 10° | 46.4 | 53.6 | 482 mm |
| 20° | 42.5 | 57.5 | 234 mm |
| 30° | 38.1 | 61.9 | 147 mm |
| 45° | 29.4 | 70.6 | 85 mm |
| 60° | 14.3 | 85.7 | 49 mm |

### Details that matter

**Speeds follow the *measured* steering angle.** The steering motor has ~3° of
deadband, so it rarely lands exactly on the commanded angle. `drive_at()` reads
the encoder and computes wheel speeds from where the wheel actually is. A small
heading error is much cheaper than slip.

**Clipping scales both wheels together.** At high speed and steer, the outer
wheel wants more than 100%. Clamping it alone would change the *ratio* between
the wheels and reintroduce scrub, so both are scaled by the same factor.

**Steering settles before driving.** Most scrubbing happens while the steered
wheel is still swinging. `drive_at(..., settle=True)` waits for it; the GUIs pass
`settle=False` and rely on the measured-angle feedback instead, so the UI never
blocks.

**Absolute positioning for steering.** Straight ahead is encoder position 98, a
fixed value that survives power cycles — no re-zeroing at startup.

**60° is geometrically valid but practically bad.** The turn radius there (49 mm)
is smaller than the drive wheels (62 mm), and the inner wheel barely creeps. Both
GUIs cap steering at 45°.

### Command rate and latency

Two hardware facts shape everything:

- **~175 ms** between issuing a motor command and the wheels moving, measured
  with `latency.py` and consistent to within 10 ms.
- **BLE drops the link above roughly 30 Hz** of writes. Faster is not better: the
  hub disconnects and floods the console with
  `Device disconnected unexpectedly`.

Hence both GUIs run their control loop at **25 Hz**. The robot also keeps moving
for about a fifth of a second after you let go — give it room.

### Hand control mapping

The palm centre is the average of the wrist and middle-finger-base landmarks,
which is steadier than any single point. Its offset from the frame centre maps to
speed and steering through a deadzone and a linear ramp:

| Offset from centre | Output |
| --- | --- |
| 0 – 0.07 | 0 (deadzone) |
| 0.10 | 11% |
| 0.20 | 48% |
| 0.34 and beyond | 100% (clamped) |

An exponential moving average (`SMOOTHING = 0.35`) removes jitter. The image is
mirrored so moving your hand right moves the on-screen dot right.

Safety: losing the hand for 5 frames stops the robot, as does disarming,
quitting, or any exception.

Tuning knobs at the top of `hand_control.py`:

| Constant | Effect |
| --- | --- |
| `DEADZONE` | size of the dead area at centre |
| `FULL_AT` | how far you must reach for full output |
| `SMOOTHING` | lower = smoother but laggier |
| `MAX_SPEED`, `MAX_STEER` | output limits |

---

## Calibration values

In `trike.py`. Redo these if the robot is rebuilt.

| Constant | Value | How it was found |
| --- | --- | --- |
| `TRACK_MM` | 70 | measured between drive wheel centres |
| `WHEELBASE_MM` | 85 | measured, drive axle to steered wheel |
| `DRIVE_WHEEL_DIA_MM` | 62 | measured (unused by the kinematics) |
| `STEER_CENTER_ABS` | 98 | `read_abs.py` with the wheel straight |
| `STEER_GEAR_RATIO` | 1.0 | `steer_test.py`; positive = left |
| `DRIVE_SIGN` | -1 | drive motor's positive direction is reversed |
| `SWAP_DRIVE_WHEELS` | True | because the reversal also mirrors left/right |

### Recalibrating the steering centre

1. Point the steered wheel straight ahead by hand.
2. `python read_abs.py`
3. Put the reported `absolutePosition` into `STEER_CENTER_ABS`.
4. `python steer_test.py` to confirm — errors should be within ~2°.

---

## Troubleshooting

**`Could not find device matching Card color 5, Card serial 0994`**

Both motors share card 0994; the library filters on *device type* as well, so
this usually means the wrong type is being requested, or the device is off.
Check with `scan_tmp.py`. Also note `le.SingleMotor` / `le.DoubleMotor` do **not**
raise when a device is missing — they print and return with `.connected == False`.
`Trike.connect()` checks this and raises a clear error.

**It says "not found" but the motor is definitely on**

A force-killed script leaves the hub holding the Bluetooth link until it times
out — wait ~15 s, or power-cycle the motor. Always exit with `Q`/`Esc`/the window
close button, which disconnects cleanly.

**Flood of `Device disconnected unexpectedly`**

Commands are being sent faster than BLE can carry them. Keep the loop at or below
25 Hz.

**UI stuck on "connecting ..."**

It now reports each stage (`connecting to drive motor ...` etc.), so the message
says which device is not answering. A GUI must pass `connect(center=False)` —
centring uses *blocking* motor moves that never return if the wheel cannot reach
its target.

**Forward and backward are swapped** → flip `DRIVE_SIGN`.

**A left turn swings the robot right** → toggle `SWAP_DRIVE_WHEELS`.

**Robot scrubs in turns** → check `WHEELBASE_MM` and `TRACK_MM`; their *ratio*
sets the wheel speed split. Then confirm steering accuracy with `steer_test.py`.

---

## Why there is no self-balancing

An earlier attempt to balance this robot on its two drive wheels was abandoned.
The sensing was solved; the actuation was not.

Three real bugs were fixed along the way, worth knowing about:

1. **Every IMU notification rebinds `dm.imu_device` to a new object.** Caching it
   once gives a permanently frozen snapshot. It must be re-read inside the loop.
2. **`imu.pitch` is unusable for this robot.** It balances at pitch ≈ 90°, exactly
   the Euler singularity: pitch pins at 90.0 while roll and yaw jump by 180°.
   Tilt must come from the accelerometer via `atan2(accZ, -accX)`, which is smooth
   through upright, fused with the gyro by a complementary filter.
3. **`lelib.yaw()` returned values 10× too large** — the hardware reports
   decidegrees. Fixed, and the docstring's sign convention was backwards too.

The blocker is the **~175 ms** command-to-motion latency. The robot falls from
upright to 35° in about 360 ms, so the wheels begin responding when the fall is
already half over. That caps usable control bandwidth near 1 Hz, while an
inverted pendulum this size needs several Hz. No choice of gains fixes dead time.

Balancing this robot would require running the control loop **on the hub** rather
than over Bluetooth.
