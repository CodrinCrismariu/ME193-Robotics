"""Move a Single Motor: rotate a set number of degrees, then rotate back.

    python examples/move_single_motor.py                  # 180 deg out and back at 40%
    python examples/move_single_motor.py 90               # 90 deg out and back
    python examples/move_single_motor.py 360 --speed 25   # slower, one full turn
    python examples/move_single_motor.py --name arm

Needs an entry of kind single_motor in hardware.json (the template calls it "arm").

Motion is bounded by degrees and the motor is stopped in a `finally`, so the
motor cannot keep running if the program is interrupted.

API verified against reference/LEGOEducation/singlemotor.md and the motor_*
section of reference/LEGOEducation/function_description.md:
  motor_run_for_degrees(degrees, *, direction=..., motor=..., speed=..., blocking=...)
  motor_reset_relative_position(*, motor=..., position=0, blocking=...)
  motor_set_end_state(end_state, *, motor=..., blocking=...)
Readable motor data (singlemotor.motor): position, absolutePosition, speed, power.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import legoeducation as le  # noqa: E402

from mobile_robotics import ConnectionFailed, HardwareConfigError, session  # noqa: E402

DEFAULT_DEGREES = 180
DEFAULT_SPEED = 40
PAUSE_SECONDS = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rotate a Single Motor out and back.")
    parser.add_argument(
        "degrees",
        nargs="?",
        type=int,
        default=DEFAULT_DEGREES,
        help=f"angle to rotate, in degrees (default: {DEFAULT_DEGREES})",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=DEFAULT_SPEED,
        help=f"speed as a percentage, 1-100 (default: {DEFAULT_SPEED})",
    )
    parser.add_argument(
        "--name",
        default="arm",
        help="device name in hardware.json (default: arm)",
    )
    parser.add_argument(
        "--no-return",
        action="store_true",
        help="rotate out only, do not come back",
    )
    args = parser.parse_args()

    if args.degrees <= 0:
        parser.error("degrees must be positive; use --no-return or reverse the direction")
    if not 1 <= args.speed <= 100:
        parser.error("speed must be between 1 and 100")
    return args


def main() -> int:
    args = parse_args()

    try:
        with session(args.name) as motor:
            try:
                # Hold position when a command finishes, so the arm does not sag
                # under load between moves.
                motor.motor_set_end_state(le.MOTOR_END_STATE_HOLD)

                # Zero the relative position so the printed values read as
                # "degrees from where we started".
                motor.motor_reset_relative_position()
                print(f"start position: {motor.motor.position} deg")

                print(f"rotating {args.degrees} deg clockwise at {args.speed}%")
                motor.motor_run_for_degrees(
                    args.degrees,
                    direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE,
                    speed=args.speed,
                )
                print(f"  now at: {motor.motor.position} deg")

                if not args.no_return:
                    time.sleep(PAUSE_SECONDS)
                    print(f"rotating {args.degrees} deg counter-clockwise back")
                    motor.motor_run_for_degrees(
                        args.degrees,
                        direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE,
                        speed=args.speed,
                    )
                    print(f"  now at: {motor.motor.position} deg")
            finally:
                # Stop before the session disconnects: a motor left running keeps
                # running after the program exits.
                motor.motor_stop()
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
