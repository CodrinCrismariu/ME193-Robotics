"""BLE adapter for one LEGO Education Double Motor, with no other devices."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
import threading
import time


TERMS = {"gyro": 3, "accel": 3, "heading": 2, "encoder_angles": 4,
         "encoder_speeds": 2, "last_action": 2}
HISTORY = 10
ACTION_SCALE = 21.0
CONTROL_DT = 0.02


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite sensor, calibration, or action value")
    return value


def validate_calibration(c, allow_estimated=False):
    estimated = allow_estimated and c.get("status") == "automatic_estimate"
    if c["verified"] is not True and not estimated:
        raise ValueError("Verify hardware.json calibration before using policy observations")
    axes = c["body_axes"]
    if sorted(abs(a) for a in axes) != [1, 2, 3]:
        raise ValueError("body_axes must be a signed permutation of [1, 2, 3]")
    # A physical rotation must preserve handedness.
    inversions = sum(abs(axes[i]) > abs(axes[j]) for i in range(3) for j in range(i + 1, 3))
    if (-1) ** inversions * math.prod(1 if a > 0 else -1 for a in axes) != 1:
        raise ValueError("body_axes must describe a right-handed rotation")
    for key in ("motor_sign", "encoder_zero_deg", "rad_s_at_100_percent"):
        if len(c[key]) != 2:
            raise ValueError(f"{key} must have two entries")
    if any(s not in (-1, 1) for s in c["motor_sign"]):
        raise ValueError("motor_sign entries must be +1 or -1")
    for key in ("gyro_rad_s_per_raw", "accel_g_per_raw"):
        if finite(c[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if finite(c["heading_rad_per_raw"]) == 0:
        raise ValueError("heading_rad_per_raw must be nonzero (signed)")
    if any(finite(s) <= 0 for s in c["rad_s_at_100_percent"]):
        raise ValueError("Measure rad_s_at_100_percent for each output")
    for v in c["encoder_zero_deg"]:
        finite(v)


class LegoRobot:
    def __init__(self, config, sdk=None, clock=time.monotonic, allow_estimated=False):
        if sdk is None:
            import legoeducation as sdk
        self.le, self.config, self.clock = sdk, config, clock
        self.allow_estimated = allow_estimated
        self.device = sdk.DoubleMotor()
        self.lock = threading.Lock()
        self.samples = {}
        self.timestamps = {}
        self.heading_zero = None
        self.last_action = [0.0, 0.0]
        self.previous_encoders = None
        self.encoder_velocity = [0.0, 0.0]
        self.history = {k: deque(maxlen=HISTORY) for k in TERMS}
        self._validate_config()

    def _validate_config(self):
        c = self.config
        if not isinstance(c["card_serial"], str) or not c["card_serial"]:
            raise ValueError("card_serial must be a nonempty string")
        if not 15 <= c["notification_ms"] <= 1000:
            raise ValueError("notification_ms must be 15..1000")
        if not 0 < finite(c["max_percent"]) <= 100:
            raise ValueError("max_percent must be 1..100")
        if not 20 <= c["command_ms"] <= 250:
            raise ValueError("command_ms must be 20..250")
        if not 0 < finite(c["stale_seconds"]) <= 1:
            raise ValueError("stale_seconds must be in (0, 1]")

    def _notification(self, notification):
        # SDK callback runs on its BLE thread. Never issue SDK commands here.
        now = self.clock()
        with self.lock:
            for item in self.le.device_notification_parser(notification):
                key = None
                if isinstance(item, self.le.ImuDeviceNotification):
                    key = "imu"
                elif isinstance(item, self.le.MotorNotification):
                    if item.motorBitMask == self.le.MOTOR_BITS_LEFT:
                        key = "left"
                    elif item.motorBitMask == self.le.MOTOR_BITS_RIGHT:
                        key = "right"
                if key:
                    # RPC objects include a Header instance; telemetry should
                    # contain only the public numeric data, suitable for JSON.
                    self.samples[key] = {k: deepcopy(v) for k, v in vars(item).items()
                                         if k != "header" and not k.startswith("_")}
                    self.timestamps[key] = now

    def __enter__(self):
        try:
            self.device.set_notification_callback(self._notification)
            color = getattr(self.le, "LEGO_COLOR_" + self.config["card_color"].upper())
            self.device.connect(card_color=color, card_serial=self.config["card_serial"],
                                device_notification_delay=self.config["notification_ms"])
            if not self.device.connected:
                raise ConnectionError("Double Motor connection failed; check power, card, and Bluetooth")
            deadline = self.clock() + 5.0
            while True:
                try:
                    self.read()
                    return self
                except RuntimeError:
                    if self.clock() >= deadline:
                        raise TimeoutError("No fresh IMU and both encoder notifications")
                    time.sleep(0.02)
        except BaseException:
            self.close()
            raise

    def read(self):
        if not self.device.connected:
            raise ConnectionError("Double Motor disconnected")
        with self.lock:
            now = self.clock()
            for key in ("imu", "left", "right"):
                if key not in self.timestamps or now - self.timestamps[key] > self.config["stale_seconds"]:
                    raise RuntimeError(f"Missing/stale {key} telemetry")
            return {"received_at": dict(self.timestamps), **deepcopy(self.samples)}

    def observation(self):
        """160 raw actor inputs; ONNX includes the trained observation normalizer."""
        c = self.config["calibration"]
        validate_calibration(c, self.allow_estimated)
        s = self.read()
        imu = s["imu"]
        def vector(prefix, scale):
            raw = [finite(imu[prefix + a]) * scale for a in "XYZ"]
            return [raw[abs(a) - 1] * (1 if a > 0 else -1) for a in c["body_axes"]]
        yaw = finite(imu["yaw"]) * c["heading_rad_per_raw"]
        if self.heading_zero is None:
            self.heading_zero = yaw
        heading = yaw - self.heading_zero
        q = [math.radians((finite(s[k]["position"]) - c["encoder_zero_deg"][i]) * c["motor_sign"][i])
             for i, k in enumerate(("left", "right"))]
        times = [s["received_at"][k] for k in ("left", "right")]
        # SDK speed is a signed byte/percentage, NOT degrees/s. Differentiate
        # encoder degrees on receipt timestamps instead of treating it as rad/s.
        if self.previous_encoders is not None:
            old_q, old_times = self.previous_encoders
            for i in range(2):
                dt = times[i] - old_times[i]
                if dt > 0:
                    self.encoder_velocity[i] = (q[i] - old_q[i]) / dt
        self.previous_encoders = q, times
        values = {
            "gyro": vector("gyroscope", c["gyro_rad_s_per_raw"]),
            "accel": [max(-8.0, min(8.0, v)) for v in vector("accelerometer", c["accel_g_per_raw"])],
            "heading": [math.sin(heading), math.cos(heading)],
            "encoder_angles": [math.sin(v) for v in q] + [math.cos(v) for v in q],
            "encoder_speeds": list(self.encoder_velocity),
            "last_action": list(self.last_action),
        }
        # mjlab uses term-major history, oldest first, backfilled on first read.
        for key, value in values.items():
            value = [finite(v) for v in value]
            if not self.history[key]:
                self.history[key].extend([value] * HISTORY)
            else:
                self.history[key].append(value)
        return [v for key in TERMS for frame in self.history[key] for v in frame]

    def command(self, action):
        validate_calibration(self.config["calibration"], self.allow_estimated)
        self.read()  # stale sensors must not renew a motion command
        if len(action) != 2:
            raise ValueError("Policy must output exactly two actions: left, right")
        c = self.config["calibration"]
        commands, applied = [], []
        for i, a in enumerate(action):
            speed = finite(a) * ACTION_SCALE
            pct = round(max(-self.config["max_percent"], min(self.config["max_percent"],
                        100 * speed / c["rad_s_at_100_percent"][i])))
            applied.append(pct / 100 * c["rad_s_at_100_percent"][i] / ACTION_SCALE)
            commands.append(pct * c["motor_sign"][i])
        try:
            # A finite device-side lease stops motion if the host loop stalls.
            with self.device.batch(blocking=False):
                for motor, pct in zip((self.le.MOTOR_LEFT, self.le.MOTOR_RIGHT), commands):
                    self.device.motor_run_for_time(
                        self.config["command_ms"], motor=motor, speed=abs(pct), blocking=False,
                        direction=(self.le.MOTOR_MOVE_DIRECTION_CLOCKWISE if pct >= 0
                                   else self.le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE))
            self.last_action = applied
        except BaseException:
            self.stop()
            raise

    def stop(self):
        try:
            if self.device.connected:
                self.device.motor_stop(motor=self.le.MOTOR_BOTH, blocking=True)
        finally:
            self.last_action = [0.0, 0.0]

    def close(self):
        try:
            self.stop()
        finally:
            self.device.disconnect()

    def __exit__(self, *_):
        self.close()
