"""Connection helpers for the LEGO(R) Education hardware used in this course.

Every LEGO Education device carries a Connection Card with a color and a serial
number. Those values differ per kit, so they live in `hardware.json` (gitignored,
copied from `hardware.example.json`) instead of being hardcoded in each program.

This module wraps the six-step connect-and-run flow documented in
`reference/LEGOEducation/connect.md`, adding two things the raw API does not:

1. Card color/serial come from config, so programs are portable between kits.
2. `session()` guarantees `disconnect()` runs even if the program raises -- an
   undisconnected device cannot return to broadcast mode for the next run.

The API surface itself is not abstracted away: `session()` hands back the real
`legoeducation` device objects, so use the upstream method names directly and
verify them against `reference/LEGOEducation/function_description.md`.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, Iterator

import legoeducation as le

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "hardware.json"
EXAMPLE_CONFIG_PATH = REPO_ROOT / "hardware.example.json"

#: Device kinds in `hardware.json` -> the `legoeducation` class that drives them.
DEVICE_CLASSES: dict[str, type] = {
    "single_motor": le.SingleMotor,
    "double_motor": le.DoubleMotor,
    "color_sensor": le.ColorSensor,
    "controller": le.Controller,
}

#: Connection Card color names in `hardware.json` -> the `le.LEGO_COLOR_*` constant.
#: Card colors per `reference/LEGOEducation/connect.md`.
CARD_COLORS: dict[str, Any] = {
    "green": le.LEGO_COLOR_GREEN,
    "blue": le.LEGO_COLOR_BLUE,
    "red": le.LEGO_COLOR_RED,
    "orange": le.LEGO_COLOR_ORANGE,
    "yellow": le.LEGO_COLOR_YELLOW,
    "azure": le.LEGO_COLOR_AZURE,
    "purple": le.LEGO_COLOR_PURPLE,
    "magenta": le.LEGO_COLOR_MAGENTA,
}


class HardwareConfigError(RuntimeError):
    """`hardware.json` is missing, malformed, or lacks the requested device."""


class ConnectionFailed(RuntimeError):
    """A device did not connect -- check power, charge, and broadcasting mode."""


def load_config(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Read `hardware.json` and return the mapping of device name -> card info."""
    path = path or CONFIG_PATH
    if not path.is_file():
        raise HardwareConfigError(
            f"{path.name} not found. Copy {EXAMPLE_CONFIG_PATH.name} to {path.name} "
            "and fill in the Connection Card color and serial of your own hardware."
        )

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HardwareConfigError(f"{path.name} is not valid JSON: {exc}") from exc

    devices = raw.get("devices")
    if not isinstance(devices, dict) or not devices:
        raise HardwareConfigError(
            f"{path.name} must contain a non-empty 'devices' object; "
            f"see {EXAMPLE_CONFIG_PATH.name}."
        )
    return devices


def connect_device(name: str, *, config: dict[str, dict[str, str]] | None = None) -> Any:
    """Build and connect the device named `name` in `hardware.json`.

    Returns the connected `legoeducation` device object. Raises `ConnectionFailed`
    if the hardware did not answer, so callers never operate on a dead handle.
    """
    devices = config if config is not None else load_config()

    entry = devices.get(name)
    if entry is None:
        raise HardwareConfigError(
            f"No device named {name!r} in hardware.json. Known devices: "
            f"{', '.join(sorted(devices)) or '(none)'}"
        )

    kind = entry.get("kind")
    if kind not in DEVICE_CLASSES:
        raise HardwareConfigError(
            f"Device {name!r} has kind {kind!r}; expected one of "
            f"{', '.join(sorted(DEVICE_CLASSES))}."
        )

    color_name = str(entry.get("card_color", "")).lower()
    if color_name not in CARD_COLORS:
        raise HardwareConfigError(
            f"Device {name!r} has card_color {entry.get('card_color')!r}; expected one of "
            f"{', '.join(sorted(CARD_COLORS))}."
        )

    serial = entry.get("card_serial")
    if not serial:
        raise HardwareConfigError(f"Device {name!r} is missing 'card_serial'.")

    device = DEVICE_CLASSES[kind]()
    # Always filter by Connection Card: a bare connect() takes the first device it
    # finds, which is a race when several are broadcasting in the same room.
    device.connect(card_color=CARD_COLORS[color_name], card_serial=str(serial))

    if not device.connected:
        raise ConnectionFailed(
            f"Could not connect to {name!r} ({kind}, {color_name} card {serial}). "
            "Check that it is charged, powered on, and broadcasting."
        )

    print(f"connected: {name} ({kind}, {color_name} card {serial})")
    return device


@contextlib.contextmanager
def session(*names: str) -> Iterator[Any]:
    """Connect the named devices, yield them, and always disconnect afterwards.

    Yields a single device when one name is given, otherwise a tuple in the order
    the names were passed:

        with session("drive", "eye") as (drive, eye):
            drive.movement_move_for_time(1000, speed=30)

    Devices already connected are disconnected even if the body raises, and a
    partial failure tears down whatever had connected so far.
    """
    if not names:
        raise ValueError("session() needs at least one device name")

    config = load_config()
    connected: list[Any] = []
    try:
        for name in names:
            connected.append(connect_device(name, config=config))
        yield connected[0] if len(connected) == 1 else tuple(connected)
    finally:
        for device in reversed(connected):
            try:
                device.disconnect()
            except Exception as exc:  # a failed teardown must not mask the real error
                print(f"warning: disconnect failed: {exc}")
