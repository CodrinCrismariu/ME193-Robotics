---
name: lego-education-python
description: Write, review, or debug Python that drives LEGO(R) Education Computer Science & AI kit hardware (Single Motor, Double Motor, Color Sensor, Controller) via the `legoeducation` PyPI package. Use whenever code imports `legoeducation`, mentions `le.SingleMotor`/`le.DoubleMotor`/`le.ColorSensor`/`le.Controller`, Connection Cards, `movement_*`/`motor_*`/`imu_*` calls, or the user talks about the LEGO Education robot, hub, levers, or BLE hardware in this repo. MANDATORY: consult the vendored upstream docs at reference/LEGOEducation before writing any LEGO API call.
---

# LEGO Education Python API

## Rule 0 — the upstream repo is the only source of truth

**You must read the vendored copy of <https://github.com/LEGO/LEGOEducation> before writing,
editing, or reviewing a single `legoeducation` API call. No exceptions, no "this one is obvious".**

This API is small, education-oriented, and deliberately breaks Python conventions. Signatures are
not guessable, method names do not follow other LEGO / Pybricks / ev3dev / SPIKE Prime APIs, and
plausible-looking calls invented from memory will silently be wrong. Model recall of this library
is unreliable — treat anything you "remember" about it as a hypothesis to verify against the docs,
never as a fact.

The docs live at:

```
reference/LEGOEducation/
```

If that directory is missing or stale, refresh it first:

```
python scripts/sync_lego_docs.py
```

That script clones or pulls `https://github.com/LEGO/LEGOEducation` into `reference/` (it is
gitignored — upstream stays upstream). If the machine is offline and the directory is absent,
say so and stop; do **not** fall back to guessing the API.

### Which file to read

| You need | Read |
|---|---|
| Exact signature, keyword args, defaults, per-arg semantics | `reference/LEGOEducation/function_description.md` (the full API reference — grep it) |
| Install, compatibility, firmware/BLE troubleshooting | `reference/LEGOEducation/README.md` |
| Connect / check / interact / disconnect lifecycle | `reference/LEGOEducation/connect.md` |
| Single Motor usage and readable data | `reference/LEGOEducation/singlemotor.md` |
| Double Motor, tank drive, IMU | `reference/LEGOEducation/doublemotor.md` |
| Color Sensor readings and callbacks | `reference/LEGOEducation/colorsensor.md` |
| Controller lever input | `reference/LEGOEducation/controller.md` |
| Multi-device programs | `reference/LEGOEducation/combine1.md`, `combine2.md` |
| Every `le.*` constant name | `reference/LEGOEducation/constants.md` |
| Runnable upstream samples | `reference/LEGOEducation/examples/*.py` |

Fast lookup of a call before you use it — grep the reference with a few lines of context:

```
grep -n -A 25 'movement_turn_for_degrees' reference/LEGOEducation/function_description.md
```

### Non-negotiables

- **Never invent a method, property, keyword argument, or constant.** If it is not in
  `function_description.md` or `constants.md`, it does not exist. Grep before you type it.
- **Never invent a constant value.** Always use the symbolic `le.*` name (e.g.
  `le.LEGO_COLOR_AZURE`, `le.MOTOR_END_STATE_BRAKE`), never a bare integer.
- **Cite what you relied on.** When you write or change LEGO API code, name the doc file (and
  ideally the section) you confirmed the signature against, so the user can check you.
- **Prefer upstream idiom over "better" Python.** The API favors simplicity over convention on
  purpose; matching the docs keeps student code readable and fits this repo's teaching context.
- If the docs are ambiguous, the escape hatch is the installed package itself:
  `python -c "import legoeducation as le; help(le.DoubleMotor)"`. Documented behavior still wins
  over your inference.

## Verified API essentials

Everything below is confirmed against the upstream docs. It is a *map*, not a substitute for
reading them — you still open the file for exact signatures and defaults.

### Import and the six-step lifecycle

`reference/LEGOEducation/connect.md` defines the shape every program follows:

```python
import legoeducation as le                                     # 1. import, always as `le`

singlemotor = le.SingleMotor()                                 # 2. define
singlemotor.connect(card_color=le.LEGO_COLOR_AZURE,            # 3. connect
                    card_serial="3683")

if not singlemotor.connected:                                  # 4. check
    print("Error connecting to Single Motor.")
    exit(1)

singlemotor.motor_run_for_degrees(360)                         # 5. interact

singlemotor.disconnect()                                       # 6. disconnect
exit(0)
```

Skipping step 4 or step 6 is a bug — an unclosed connection leaves the hardware unable to
re-enter broadcast mode for the next run. In this repo, use `src/mobile_robotics/hardware.py`
(`connect_device` / `session`) so disconnect happens even on exception.

### The four device classes

| Class | Purpose |
|---|---|
| `le.SingleMotor()` | One motor: direction, speed, absolute/relative position, power, accel/decel, end state |
| `le.DoubleMotor()` | Two motors (left/right), synchronized or independent, **plus a 6-axis IMU** |
| `le.ColorSensor()` | Detected LEGO color number, raw RGB, HSV, reflection |
| `le.Controller()` | Two levers (left/right), angle and percent |

### Command-family prefixes

- `motor_*` — single-motor-style commands. Work on `SingleMotor` and, via the `motor=` keyword
  (`le.MOTOR_LEFT` / `le.MOTOR_RIGHT`), on one side of a `DoubleMotor`.
  e.g. `motor_run`, `motor_run_for_time`, `motor_run_for_degrees`,
  `motor_run_to_relative_position`, `motor_run_to_absolute_position`, `motor_set_speed`,
  `motor_set_duty_cycle`, `motor_stop`, `motor_set_end_state`, `motor_set_acceleration`,
  `motor_reset_relative_position`.
- `movement_*` — **DoubleMotor only**, drives both sides as a vehicle.
  e.g. `movement_move`, `movement_move_for_time`, `movement_move_for_degrees`,
  `movement_move_tank`, `movement_move_tank_for_degrees`, `movement_turn_for_degrees`,
  `movement_stop`, `movement_set_speed`, `movement_set_end_state`,
  `movement_set_acceleration`, `movement_set_turn_steering`.
- `imu_*` — DoubleMotor IMU config: `imu_set_yaw_face`, `imu_reset_yaw_axis`.
- Shared: `search`, `connect`, `disconnect`, `info`, `device_uuid`, `done`,
  `set_notification_callback`, `device_notification_request`, `program_flow_notification`,
  `light_color`, `beep`, `stop_beep`, `begin_batch` / `end_batch` / `batch` / `cancel_batch`,
  and the module-level `le.device_notification_parser`.

Do not mix the families up: `movement_turn_for_degrees` on a `SingleMotor` is not a thing.

### Keyword-only conventions

Most optional parameters are keyword-only (after `*`). Two that matter constantly:

- `blocking=True` (default) waits for the command to finish. `blocking=False` returns
  immediately — pair it with `device.done()` when you need to wait later.
- `speed` defaults to "unchanged": omitting it reuses the previously set speed rather than
  overriding it. Pass an explicit `speed=` when the value matters.

  **Trap:** the docs render that default as `speed: int = UNCHANGED`, but `UNCHANGED` is an
  internal of `legoeducation.device` whose value is `None` — it is **not** exported, so
  `le.UNCHANGED` raises `AttributeError` and is absent from `constants.md`. To mean
  "unchanged", omit the argument (or pass `speed=None`); never write `le.UNCHANGED`.
  Verified against the installed package, v1.1.1.

Docstring defaults in `function_description.md` are written with internal constant *names*
that are not always importable. When a default's name is not listed in `constants.md`, check
the installed package before using it as a value:

```
python -c "import legoeducation as le, inspect; print(inspect.signature(le.SingleMotor.motor_run_for_degrees))"
```

Negative speeds reverse direction, and there are also explicit direction constants
(`le.MOTOR_MOVE_DIRECTION_*`, `le.MOVEMENT_DIRECTION_*`, `le.MOVEMENT_MOVE_DIRECTION_*`,
`le.MOVEMENT_TURN_DIRECTION_*`). Check `constants.md` for the right family — they are not
interchangeable.

### Reading sensor data

Two styles, both documented — inline polling and a notification callback:

```python
# Inline
print(colorsensor.sensor.color)          # compare against le.LEGO_COLOR_* constants
print(controller.sensor.leftPercent)
print(singlemotor.motor.position)
print(doublemotor.motor[le.MOTOR_LEFT].speed)
print(doublemotor.imu_device.yaw)

# Callback
def on_notification(data):
    for item in le.device_notification_parser(data):
        if isinstance(item, le.ColorSensorNotification):
            print(item.color)

colorsensor.set_notification_callback(on_notification)
```

Attribute names are **camelCase and inconsistent between devices** — `SingleMotor` exposes
`absolutePosition` while `DoubleMotor`'s per-motor data exposes `absolutePos`. This is exactly
the kind of trap that makes Rule 0 non-optional: read the device's own `## Available Data`
section every time.

Readable fields, per upstream:

- `singlemotor.motor`: `motorState`, `absolutePosition`, `power`, `speed`, `position`, `gesture`
- `doublemotor.motor[le.MOTOR_LEFT | le.MOTOR_RIGHT]`: `motorBitMask`, `motorState`,
  `absolutePos`, `power`, `speed`, `position`, `gesture`
- `doublemotor.imu_device`: `orientation`, `yawFace`, `yaw`, `pitch`, `roll`,
  `accelerometerX/Y/Z`, `gyroscopeX/Y/Z`; plus `doublemotor.imu_gesture`
- `colorsensor.sensor`: `color`, `reflection`, `rawRed`, `rawGreen`, `rawBlue`, `hue`,
  `saturation`, `value`
- `controller.sensor`: `leftPercent`, `rightPercent`, `leftAngle`, `rightAngle`

### Connection Cards

Every physical device carries a Connection Card with a color and a serial number. A bare
`connect()` grabs the first matching device it finds — fine with one device on the bench, a race
condition in a classroom. **Always filter by card in this repo**, and read the card values from
`hardware.json` (see `hardware.example.json`) rather than hardcoding them, so the same program
runs on another kit.

Card colors: `le.LEGO_COLOR_GREEN`, `_BLUE`, `_RED`, `_ORANGE`, `_YELLOW`, `_AZURE`,
`_PURPLE`, `_MAGENTA`.

`search(timeout=..., card_color=..., card_serial=...)` lists what is broadcasting — use it to
debug "which device is that?" before reaching for `connect()`. It returns `bleak` `BLEDevice`
objects whose `.name` carries the card as an emoji and serial, e.g.
`'🔴 1129 Single Motor'` — that is the fastest way to read a device's real card values.

Two behaviors verified against hardware (package v1.1.1):

- **`search()` returns `None`, not `[]`,** when nothing matches the filter. `len(...)` on the
  result raises `TypeError`. Always guard with `if not found:` before iterating.
- **The card filter is strict.** Wrong color, wrong serial, or either one alone all leave
  `connected == False` — there is no fall back to first-found. `connect()` does not raise;
  it logs `Could not find device matching Card color <n>, Card serial <s>` and returns, which
  is exactly why checking `.connected` is mandatory.

### Batching

`begin_batch()` / `end_batch()` (or the `batch()` helper) queue several configuration and start
commands so they take effect together — the documented way to start both sides of a `DoubleMotor`
simultaneously. `cancel_batch()` discards the queue. Read the `# Other Functions` section of
`function_description.md` before using these.

## Working in this repo

- Runnable programs go in `examples/`; reusable logic in `src/mobile_robotics/`.
- Connect through `src/mobile_robotics/hardware.py` rather than repeating boilerplate.
- Never commit `hardware.json` (gitignored) — it is per-kit. Update `hardware.example.json`
  when the schema changes.
- `reference/LEGOEducation/` is gitignored and refreshed by `python scripts/sync_lego_docs.py`.
  Do not edit anything inside it; it is upstream's file.

## Hardware realities to design around

Code that ignores these looks correct and fails on the bench:

- **BLE, not USB.** The host needs Bluetooth access from Python. On Chromebooks the Linux
  container cannot reach Bluetooth at all, so connecting is impossible there.
- **Python >= 3.14** is required by `legoeducation`. An import-time failure is usually an old
  interpreter, not bad code.
- **Hardware must be charged, powered on, and broadcasting** before `connect()`. A failed
  connect is the normal case to handle, not an edge case.
- **Firmware:** update hardware by connecting it once at <https://code.legoeducation.com>.
  Unexplained command failures are often stale firmware.
- **Many simultaneous connections degrade responsiveness.** Connect only what a program uses.
- **Every physical run needs floor space and a stop path.** Prefer bounded commands
  (`*_for_time`, `*_for_degrees`) over open-ended `motor_run` / `movement_move` in examples, and
  always stop and disconnect in a `finally`.

## Checklist before you hand back LEGO code

1. Did you open `reference/LEGOEducation/` this session? (Not "do you recall it" — did you open it.)
2. Does every method, property, keyword argument, and constant appear verbatim in
   `function_description.md` or `constants.md`?
3. Right command family for the device (`motor_*` vs `movement_*` vs `imu_*`)?
4. Is `connect()` filtered by Connection Card, sourced from config?
5. Is `connected` checked, with a real failure path?
6. Is `disconnect()` guaranteed on every exit path, including exceptions?
7. Is motion bounded, or an explicit stop guaranteed?
8. Did you tell the user which doc file you verified against?
