"""Record raw calibration measurements without guessing conversion constants."""
import argparse
import json
from pathlib import Path
import time

from lego_robot import LegoRobot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--seconds", type=float, default=3)
    args = parser.parse_args()
    if not 0 < args.seconds <= 60:
        parser.error("seconds must be 0..60")
    config = json.loads(Path(__file__).with_name("hardware.json").read_text())
    records = []
    with LegoRobot(config) as robot:
        start = time.monotonic()
        print("RECORDING", args.name, flush=True)
        while time.monotonic() - start < args.seconds:
            records.append(robot.read())
            time.sleep(0.02)
    folder = Path(__file__).with_name("calibration")
    folder.mkdir(exist_ok=True)
    filename = folder / (args.name + ".json")
    filename.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Saved {len(records)} samples to {filename}")
    if records:
        print(json.dumps(records[-1], indent=2))


if __name__ == "__main__":
    main()
