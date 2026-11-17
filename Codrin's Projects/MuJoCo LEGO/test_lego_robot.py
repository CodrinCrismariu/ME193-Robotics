"""Hardware-independent checks: python -m unittest -v"""
from contextlib import nullcontext
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from lego_robot import LegoRobot, validate_calibration


class Imu:
    def __init__(self):
        self.header = object()  # SDK RPC header is not JSON serializable
        self.yaw = 0
        for axis in "XYZ":
            setattr(self, "gyroscope" + axis, 0)
            setattr(self, "accelerometer" + axis, 1000 if axis == "Z" else 0)


class Motor:
    def __init__(self, bit, pos=0):
        self.motorBitMask, self.position = bit, pos


class Device:
    def __init__(self):
        self.connected = True
        self.calls = []

    def batch(self, **kw):
        return nullcontext()

    def motor_run_for_time(self, *args, **kw):
        self.calls.append((args, kw))

    def motor_stop(self, **kw):
        self.calls.append(("stop", kw))

    def disconnect(self):
        self.connected = False


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(Path(__file__).with_name("hardware.json").read_text())
        self.cfg["max_percent"] = 20
        self.cfg["calibration"].update(verified=True, gyro_rad_s_per_raw=math.pi / 180,
                                      accel_g_per_raw=0.001, heading_rad_per_raw=math.pi / 180,
                                      body_axes=[1, 2, 3], motor_sign=[1, 1], encoder_zero_deg=[0, 0],
                                      rad_s_at_100_percent=[30, 30])
        self.now = 100.0
        self.sdk = SimpleNamespace(DoubleMotor=Device, ImuDeviceNotification=Imu,
                                   MotorNotification=Motor, device_notification_parser=lambda n: n,
                                   MOTOR_BITS_LEFT=1, MOTOR_BITS_RIGHT=2, MOTOR_LEFT=0,
                                   MOTOR_RIGHT=1, MOTOR_BOTH=2,
                                   MOTOR_MOVE_DIRECTION_CLOCKWISE=1,
                                   MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE=-1)
        self.robot = LegoRobot(self.cfg, self.sdk, clock=lambda: self.now)
        self.robot._notification([Imu(), Motor(1), Motor(2)])

    def test_observation_units_and_term_history(self):
        obs = self.robot.observation()
        self.assertEqual(len(obs), 160)
        self.assertEqual(obs[30:60], [0, 0, 1] * 10)
        self.assertEqual(obs[60:80], [0, 1] * 10)
        self.now += 0.1
        imu = Imu()
        imu.gyroscopeX, imu.yaw = 180, 90
        self.robot._notification([imu, Motor(1, 90), Motor(2, -90)])
        obs = self.robot.observation()
        self.assertAlmostEqual(obs[27], math.pi)  # newest gyro x
        self.assertEqual(obs[:27], [0] * 27)
        self.assertAlmostEqual(obs[78], 1)  # newest heading sin
        self.assertAlmostEqual(obs[116], 1)  # newest left sin
        self.assertAlmostEqual(obs[117], -1)  # newest right sin
        self.assertAlmostEqual(obs[138], math.pi / 0.2)  # newest left speed

    def test_stale_encoder_even_with_fresh_imu(self):
        self.now += 0.21
        self.robot._notification([Imu()])
        with self.assertRaisesRegex(RuntimeError, "left"):
            self.robot.command([0.5, 0.5])
        self.assertEqual(self.robot.device.calls, [])

    def test_raw_telemetry_is_json_serializable(self):
        json.dumps(self.robot.read())

    def test_finite_lease_limit_direction_and_last_action(self):
        self.cfg["calibration"]["motor_sign"] = [1, -1]
        self.robot.command([1, 1])
        left, right = self.robot.device.calls
        self.assertEqual(left[0], (200,))
        self.assertEqual(left[1]["speed"], 20)
        self.assertEqual(right[1]["direction"], -1)
        self.assertAlmostEqual(self.robot.last_action[0], 6 / 21)

    def test_bad_actions_never_move(self):
        for action in ([1], [1, float("nan")], [float("inf"), 1]):
            with self.assertRaises(ValueError):
                self.robot.command(action)
        self.assertEqual(self.robot.device.calls, [])

    def test_requires_calibration(self):
        self.cfg["calibration"]["verified"] = False
        with self.assertRaises(ValueError):
            self.robot.observation()

    def test_estimate_requires_explicit_opt_in(self):
        c = self.cfg['calibration']
        c.update(verified=False, status='automatic_estimate')
        with self.assertRaises(ValueError):
            validate_calibration(c)
        validate_calibration(c, allow_estimated=True)

    def test_invalid_axis_mapping(self):
        self.cfg["calibration"]["body_axes"] = [-1, 2, 3]
        with self.assertRaises(ValueError):
            validate_calibration(self.cfg["calibration"])

    def test_disconnect_on_exception(self):
        try:
            raise RuntimeError("policy failed")
        except RuntimeError:
            self.robot.__exit__(RuntimeError, None, None)
        self.assertFalse(self.robot.device.connected)
        self.assertEqual(self.robot.device.calls[0][0], "stop")


if __name__ == "__main__":
    unittest.main()
