"""Two devices in one program: a Color Sensor gates the Double Motor.

Green -> drive forward. Red -> stop. Anything else -> hold the last decision.
Runs for a fixed 15 seconds, then stops and disconnects.

    python examples/color_reactive_drive.py

Needs a "drive" (double_motor) and an "eye" (color_sensor) entry in hardware.json.

API verified against reference/LEGOEducation/colorsensor.md (sensor.color),
reference/LEGOEducation/combine1.md (multi-device pattern), and the movement_*
section of reference/LEGOEducation/function_description.md.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import legoeducation as le  # noqa: E402

from mobile_robotics import ConnectionFailed, HardwareConfigError, session  # noqa: E402

RUN_SECONDS = 15
POLL_INTERVAL = 0.1
SPEED = 25


def main() -> int:
    try:
        with session("drive", "eye") as (drive, eye):
            print("green = go, red = stop. Running for 15 seconds; Ctrl+C to quit early.")
            moving = False
            deadline = time.monotonic() + RUN_SECONDS
            try:
                while time.monotonic() < deadline:
                    color = eye.sensor.color

                    if color == le.LEGO_COLOR_GREEN and not moving:
                        print("green -> go")
                        # blocking=False so the loop keeps polling the sensor
                        # while the robot drives.
                        drive.movement_move(
                            direction=le.MOVEMENT_DIRECTION_FORWARD,
                            speed=SPEED,
                            blocking=False,
                        )
                        moving = True
                    elif color == le.LEGO_COLOR_RED and moving:
                        print("red -> stop")
                        drive.movement_stop()
                        moving = False

                    time.sleep(POLL_INTERVAL)
            finally:
                drive.movement_stop()
    except (HardwareConfigError, ConnectionFailed) as exc:
        print(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("interrupted")
        return 130

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
