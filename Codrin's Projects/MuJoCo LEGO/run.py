"""Run on Windows for Bluetooth access; train/export in WSL or on HPC."""
import argparse
import json
from pathlib import Path
import time

from lego_robot import CONTROL_DT, LegoRobot, validate_calibration


def positive_seconds(value):
    value = float(value)
    if not 0 < value <= 3600:
        raise argparse.ArgumentTypeError("seconds must be in (0, 3600]")
    return value


def load_policy(path):
    import numpy as np
    import onnxruntime as ort
    manifest = json.loads(path.with_suffix(".json").read_text())
    expected = {"task": "Beam-Drive-Straight", "history": 10, "observation_size": 160,
                "action_scale": 21.0, "control_dt": CONTROL_DT,
                "terms": ["gyro", "accel", "heading", "encoder_angles", "encoder_speeds", "last_action"],
                "history_order": "term-major-oldest-first"}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Policy contract mismatch: {key}")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 160] or len(outputs) != 1 or outputs[0].shape != [1, 2]:
        raise ValueError("Expected feed-forward ONNX model [1,160] -> [1,2]")

    def policy(observation):
        out = session.run(None, {inputs[0].name: np.asarray([observation], dtype=np.float32)})[0]
        if out.shape != (1, 2) or not np.isfinite(out).all():
            raise ValueError("Invalid policy output")
        return out[0].tolist()

    policy([0.0] * 160)  # warm up before connecting to hardware
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "monitor", "policy"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("hardware.json"))
    parser.add_argument("--seconds", type=positive_seconds, default=10.0)
    parser.add_argument("--model", type=Path, help="Exported policy.onnx (not an rsl_rl checkpoint)")
    parser.add_argument("--move", action="store_true", help="Enable motor commands in policy mode")
    parser.add_argument("--allow-estimated", action="store_true", help="Use saved automatic calibration estimates")
    args = parser.parse_args()
    if args.move and args.mode != "policy":
        parser.error("--move is only supported in policy mode")
    config = json.loads(args.config.read_text())
    policy = None
    if args.mode == "policy":
        if args.model is None:
            parser.error("policy mode requires --model policy.onnx")
        validate_calibration(config["calibration"], args.allow_estimated)
        policy = load_policy(args.model)
    with LegoRobot(config, allow_estimated=args.allow_estimated) as robot:
        print("Connected to Double Motor. " + ("Motion enabled." if args.move else "Reading sensors only."))
        if args.mode == "check":
            print(json.dumps(robot.read(), indent=2))
            return
        start = deadline = time.monotonic()
        print_at = start
        while time.monotonic() - start < args.seconds:
            now = time.monotonic()
            if now - deadline > 0.1:
                raise RuntimeError("Control loop missed its deadline by over 100 ms")
            if policy:
                action = policy(robot.observation())
                if args.move:
                    robot.command(action)
                if now >= print_at:
                    print(json.dumps({"seconds": round(now - start, 2), "policy": action,
                                      "applied": robot.last_action}))
            elif now >= print_at:
                print(json.dumps(robot.read()))
            if now >= print_at:
                print_at = now + 0.2
            deadline += CONTROL_DT
            time.sleep(max(0.0, deadline - time.monotonic()))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.")
