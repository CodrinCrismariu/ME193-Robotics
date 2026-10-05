# AI in Mobile Robotics

Coursework and robot programs for **AI in Mobile Robotics**, driving LEGO® Education
Computer Science & AI kit hardware — Single Motor, Double Motor, Color Sensor, and
Controller — from Python over Bluetooth Low Energy.

Built on the [LEGO® Education Python API](https://github.com/LEGO/LEGOEducation)
([`legoeducation`](https://pypi.org/project/legoeducation/) on PyPI).

## Requirements

- **Python 3.14** — what this project targets and tests on. Upstream's README specifies 3.14 as
  the minimum, though the published wheel's own metadata accepts `>=3.11`; we follow upstream's
  guidance rather than the looser metadata.
- A machine where Python can reach **Bluetooth**. Chromebooks will not work: the Linux
  container Python runs in has no Bluetooth access.
- LEGO Education hardware, charged, powered on, and broadcasting.
- Up-to-date hardware firmware — connect each device once at
  [code.legoeducation.com](https://code.legoeducation.com) to update it.

## Setup

```bash
git clone <this-repo>
cd "AI in Mobile Robotics"

py -3.14 -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux

pip install -r requirements.txt

# Vendor the LEGO API documentation into reference/ (read before writing any LEGO code)
python scripts/sync_lego_docs.py

# Tell the project about your hardware
copy hardware.example.json hardware.json     # cp on macOS/Linux
```

Then edit `hardware.json`: each device has a Connection Card with a color and a serial
number printed on it. Enter those, and give each device a short name you will use in code.

```json
{
  "devices": {
    "drive": { "kind": "double_motor", "card_color": "azure",  "card_serial": "3683" },
    "eye":   { "kind": "color_sensor", "card_color": "red",    "card_serial": "1427" }
  }
}
```

`hardware.json` is gitignored — those values belong to the physical kit in front of you,
not to the repo.

## Run something

```bash
python examples/check_connection.py drive    # smoke test: connect, blink, beep, report
python examples/move_single_motor.py 180     # Single Motor rotates out and back
python examples/drive_square.py              # Double Motor drives a square
python examples/color_reactive_drive.py      # Color Sensor gates the Double Motor
```

Start with `check_connection.py` whenever hardware "isn't working" — it separates the BLE
connection from your robot logic.

## Layout

| Path | What it holds |
|---|---|
| `src/mobile_robotics/` | Reusable helpers: config-driven connect, `session()` teardown |
| `examples/` | Runnable programs, one behavior each |
| `scripts/sync_lego_docs.py` | Vendors the upstream LEGO docs into `reference/` |
| `reference/LEGOEducation/` | Upstream LEGO docs (gitignored, read-only) |
| `hardware.example.json` | Template for your `hardware.json` |
| `.claude/skills/` | Skills for Claude Code, incl. the mandatory LEGO API rules |

## Writing code against the hardware

Use `session()` rather than raw `connect()`/`disconnect()` pairs — it looks the Connection
Card up in `hardware.json` and guarantees disconnect even if your program raises. Hardware
left connected cannot return to broadcast mode for the next run.

```python
import legoeducation as le
from mobile_robotics import session

with session("drive") as drive:
    drive.movement_move_for_time(1000, speed=30)
    drive.movement_turn_for_degrees(90, direction=le.MOVEMENT_TURN_DIRECTION_LEFT)
    drive.movement_stop()
```

The underlying `legoeducation` objects are handed straight back, so all upstream methods
are available — nothing is wrapped or renamed.

### The documentation rule

<https://github.com/LEGO/LEGOEducation> is the only authority on this API. **Read
`reference/LEGOEducation/` before writing any LEGO API call.** Method names, keyword
arguments, and constants are not guessable, and the API resembles neither idiomatic Python
nor other LEGO SDKs (Pybricks, ev3dev, SPIKE Prime). If a name is not in
`function_description.md` or `constants.md`, it does not exist.

| You need | Read |
|---|---|
| Signatures, keyword args, defaults | `reference/LEGOEducation/function_description.md` |
| Connect / check / interact / disconnect | `reference/LEGOEducation/connect.md` |
| Per-device usage and readable data | `singlemotor.md`, `doublemotor.md`, `colorsensor.md`, `controller.md` |
| Multi-device programs | `combine1.md`, `combine2.md` |
| Constant names | `reference/LEGOEducation/constants.md` |

That rule is encoded for Claude Code in
[`.claude/skills/lego-education-python/SKILL.md`](.claude/skills/lego-education-python/SKILL.md)
and [`CLAUDE.md`](CLAUDE.md), so AI-assisted edits check the docs instead of guessing.

## Troubleshooting

| Symptom | Try |
|---|---|
| `ImportError` on `import legoeducation` | Check `python --version` — 3.14+ is required |
| Connect fails | Charged? Powered on? Broadcasting? Right card color and serial? |
| Wrong device connects | Filter by Connection Card (`session()` does this); check `hardware.json` |
| Commands ignored or erratic | Update firmware at [code.legoeducation.com](https://code.legoeducation.com) |
| Sluggish response | Connect only the devices the program actually uses |
| Robot keeps moving after exit | Stop in a `finally`; prefer `*_for_time` / `*_for_degrees` |

More in [LEGO Education's FAQ](https://teach.legoeducation.com/en-us/computer-science/frequently-asked-questions).

## Attribution

LEGO® is a trademark of the LEGO Group, which does not sponsor, authorize, or endorse this
project. The LEGO® Education Python API is © the LEGO Group and distributed under its own
license; see the upstream [repository](https://github.com/LEGO/LEGOEducation). This repo
vendors those docs at build time and does not redistribute them.
