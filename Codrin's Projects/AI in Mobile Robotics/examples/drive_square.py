"""Drive the Double Motor in a square: forward, turn 90 degrees, four times.

    python examples/drive_square.py

Needs a "drive" entry of kind double_motor in hardware.json, and about a metre
of clear floor. Motion is bounded (for_time / for_degrees) and the session
context manager stops and disconnects even if you interrupt the program.

API verified against reference/LEGOEducation/doublemotor.md and the
movement_* section of reference/LEGOEducation/function_description.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import legoeducation as le  # noqa: E402

from mobile_robotics import ConnectionFailed, HardwareConfigError, session  # noqa: E402

SIDES = 4
SIDE_TIME_MS = 1000
TURN_DEGREES = 90
SPEED = 30


def main() -> int:
    try:
        with session("drive") as drive:
            try:
                for side in range(SIDES):
                    print(f"side {side + 1}/{SIDES}")
                    drive.movement_move_for_time(SIDE_TIME_MS, speed=SPEED)
                    drive.movement_turn_for_degrees(
                        TURN_DEGREES,
                        direction=le.MOVEMENT_TURN_DIRECTION_LEFT,
                        speed=SPEED,
                    )
            finally:
                # Stop before the session disconnects, so the robot never keeps
                # rolling after the program ends.
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
