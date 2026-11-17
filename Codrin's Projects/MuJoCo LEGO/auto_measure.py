"""Short timed motor probes; save raw data and estimated speed conversion."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

from lego_robot import LegoRobot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--percent', type=int, default=20, choices=(20, 30))
    parser.add_argument('--duration-ms', type=int, default=200, choices=(200, 250))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root / "hardware.json").read_text())
    trials = []
    with LegoRobot(config) as robot:
        le = robot.le
        initial = robot.read()
        print("Initial pose:", json.dumps(initial), flush=True)
        for name, motor in (("left", le.MOTOR_LEFT), ("right", le.MOTOR_RIGHT)):
            for direction, label in ((le.MOTOR_MOVE_DIRECTION_CLOCKWISE, "clockwise"),
                                     (le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, "counterclockwise")):
                robot.read()
                print("Probing", name, label, args.percent, args.duration_ms, flush=True)
                samples = []
                before = robot.read()
                robot.device.motor_run_for_time(args.duration_ms, motor=motor, speed=args.percent,
                                               direction=direction, blocking=False)
                started = time.monotonic()
                try:
                    while time.monotonic() - started < 0.7:
                        samples.append(robot.read())
                        time.sleep(0.01)
                finally:
                    robot.stop()
                trials.append({"motor": name, "direction": label, "percent": args.percent,
                               "duration_ms": args.duration_ms,
                               "before": before, "samples": samples})
                time.sleep(0.5)
        final = robot.read()
    folder = root / "calibration"
    folder.mkdir(exist_ok=True)
    result = {"initial": initial, "trials": trials, "final": final}
    (folder / "motor_probes.json").write_text(json.dumps(result, indent=2) + "\n")
    for t in trials:
        key = t["motor"]
        samples = [t["before"], *t["samples"]]
        unique = {s["received_at"][key]: s for s in samples}
        seq = list(sorted(unique.items()))
        velocities = [(b[1][key]["position"] - a[1][key]["position"]) / (b[0] - a[0])
                      for a, b in zip(seq, seq[1:]) if b[0] > a[0]]
        moving = [v for v in velocities if abs(v) > 50]
        print(key, t["direction"], "delta_deg", seq[-1][1][key]["position"] - seq[0][1][key]["position"],
              "moving_deg_s", statistics.median(moving) if moving else 0,
              "gyro_peak", [max((s["imu"]["gyroscope"+a] for s in samples), key=abs) for a in "XYZ"], flush=True)


if __name__ == "__main__":
    main()
