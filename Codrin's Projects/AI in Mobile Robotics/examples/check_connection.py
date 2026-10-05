"""Smoke test: connect to one configured device, blink it, and disconnect.

Run this first whenever the hardware "isn't working" -- it isolates the BLE
connection from whatever robot logic you are debugging.

    python examples/check_connection.py drive

Device names come from hardware.json (see hardware.example.json).

API verified against reference/LEGOEducation/connect.md and
reference/LEGOEducation/function_description.md (light_color, beep).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import legoeducation as le  # noqa: E402

from mobile_robotics import ConnectionFailed, HardwareConfigError, load_config, session  # noqa: E402


def main() -> int:
    try:
        devices = load_config()
    except HardwareConfigError as exc:
        print(f"error: {exc}")
        return 1

    if len(sys.argv) > 2:
        print(f"usage: python {Path(__file__).name} [device-name]")
        return 2

    if len(sys.argv) == 2:
        name = sys.argv[1]
    else:
        name = next(iter(devices))
        print(f"no device given, using the first in hardware.json: {name}")

    try:
        with session(name) as device:
            # Announce ourselves on the hardware itself, so you can tell which
            # physical brick answered.
            device.light_color(le.LEGO_COLOR_GREEN, pattern=le.LIGHT_PATTERN_BREATHE)
            device.beep(pattern=le.SOUND_PATTERN_BEEP_SINGLE, frequency=880)

            # info() returns an object, not a dict -- iterate it with vars(),
            # per the function_description.md entry for info().
            print(f"{name} technical info:")
            for key, value in vars(device.info()).items():
                print(f"  {key}: {value}")
    except (HardwareConfigError, ConnectionFailed) as exc:
        print(f"error: {exc}")
        return 1

    print("connection OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
