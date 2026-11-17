"""mjlab task: teach the double-motor beam robot to drive in a straight line down +x.

The policy sees only what the real robot can sense (its motion sensor and encoders, with their noise, bias and
Bluetooth delay); the critic, used only in training, sees everything.

The robot (onshape/beam_robot.xml, built from your Onshape export by build_beam_robot.py) is a double motor with a beam
on each output shaft. It has no wheels: spinning the beams makes it tip over its own shaft again and again, so it moves
by tumbling. The two beams are different (one centred on its shaft, one longer and fixed at its end), so spinning both
the same way drifts sideways. The policy must learn how fast to spin each beam to go straight.

Importing this file registers the task "Beam-Drive-Straight" (see train.py / play.py / make_video.py next to it).
Everything you will want to tune is in the constants block and the reward weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  apply_body_impulse,
  dr,
  is_terminated,
  joint_vel_rel,
  last_action,
  projected_gravity,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.envs.mdp.actions.actions import JointVelocityAction
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import BuiltinSensorCfg, ObjRef
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ---- constants you may want to change ----------------------------------------------------------------
# the robot built by build_beam_robot.py: next to this file when run by runIt.py, one folder up in the repo
_HERE = Path(__file__).resolve().parent
ROBOT_XML = next(p for p in (_HERE / "onshape" / "beam_robot.xml", _HERE.parent / "onshape" / "beam_robot.xml") if p.exists())
LEFT, RIGHT = "left_beam", "right_beam"          # joint names in beam_robot.xml
TARGET_SPEED = 0.2          # m/s forward; spinning both beams at 6 rad/s by hand tumbles it along at about 0.12 m/s
# "Straight" means: along +x, the target direction. The robot starts up to START_YAW off it (put down crooked, or knocked)
# and must turn back. It knows how far off it is from its motion sensor's heading, which on the real robot is zeroed
# while it faces the target direction (set it down pointing the way it should go, start the program, then it may be
# moved or pushed). That heading drifts (SENSOR_NOISE below), as a real one does.
START_YAW = math.radians(45)   # rad; it starts pointing up to this far off the target direction, either way
MAX_OFF_LINE = 0.5          # m; the episode ends if the robot wanders this far from y = 0
EPISODE_S = 10.0

# What the policy sees: only the Double Motor's own sensors (spec sheet: 3-axis accelerometer, 3-axis gyro, an absolute
# encoder on each output), as imperfect as real ones. LEGO does not name the motion sensor chip; these numbers are for a
# typical 6-axis MEMS part of this class (ST LSM6DS3 datasheet: accelerometer zero-g offset +-40 mg, gyro zero-rate
# level up to +-10 deg/s before calibration). LEGO firmware recalibrates the gyro whenever the robot sits still, so what
# is left is much smaller. LEGO motor encoders count 360 steps per turn (1 degree, +-1 degree) and report speed in whole
# degrees per second. "bias" is a fixed error drawn for each episode; "noise" is new on every reading.
SENSOR_NOISE = {
  "gyro_noise": 0.01,           # rad/s (0.6 deg/s)
  "gyro_bias": 0.01,            # rad/s, +- (what the at-rest calibration leaves)
  "accel_noise": 0.01,          # g
  "accel_bias": 0.04,           # g, +- (the datasheet's zero-g offset)
  "accel_range": 8.0,           # g; readings beyond this are cut off (the impacts of tumbling can exceed it)
  "encoder_step": math.radians(1.0),   # rad; angles come in whole degrees
  "encoder_offset": math.radians(1.0), # rad, +- (the encoder's accuracy)
  "speed_step": math.radians(1.0),     # rad/s; speeds come in whole degrees per second
  "speed_noise": 0.2,           # rad/s (speed from counting 1-degree steps over a few ms is jumpy)
  "heading_drift": 0.01,        # rad/s, +-: the heading is the gyro added up over time, so its bias makes it drift
                                # (0.01 rad/s = 6 degrees over a 10 s episode)
  "heading_walk": 0.005,        # rad/sqrt(s): random wander of that sum (from the gyro noise)
}
SENSOR_HISTORY = 10             # the policy sees its last 10 readings (200 ms, about one full delay), so it can tell
                                # how things change

# The motor: LEGO Education Double Motor 45651 (spec sheet, measured at 4.2 V): no load 322-346 rpm, 192-220 rpm at a
# 50 mN*m load, stall torque 148-173 mN*m.
MAX_BEAM_SPEED = 21.0       # rad/s per policy action unit; keep the deployed ONNX adapter's action scale
# Match hardware.json's percentage-to-speed conversion, including saturation and whole-percent commands.
# These are provisional 100%-command estimates from short 20% probes, NOT measured no-load maxima.
# LEGO specifies 322-346 rpm unloaded (33.72-36.23 rad/s), and 192-220 rpm at 0.05 N*m.
MOTOR_COMMAND_LIMITS = {LEFT: 16.494, RIGHT: 14.076}   # rad/s; left ~157.5 rpm, right ~134.4 rpm
MOTOR_KD = 0.05             # how hard the motor's speed control pushes toward the speed it was told (N*m per rad/s)
MOTOR_TORQUE_LIMIT = 0.16   # N*m, the middle of the stall torque range

# Delay. The policy runs on a laptop or tablet that talks to the motor over Bluetooth Low Energy 4.2 (as the LEGO Coding
# Canvas app and its Python API do). LEGO publishes no latency; BLE sends data only once per "connection interval"
# (7.5 ms at the fastest the standard allows, 15-30 ms is what laptops and tablets usually pick), each way, plus app
# time: 14-90 ms in all. Real setups are often slower than that (the Coding Canvas app runs in a browser through Web
# Bluetooth; the operating system and the app add their own buffering and scheduling; users report noticeable lag), so
# training assumes up to about 200 ms from a reading to the motor acting on the answer.
COMMAND_DELAY_MS = (30, 120)  # policy -> motor
SENSOR_DELAY_MS = (20, 80)    # motor sensors -> policy (in steps of the 20 ms control period)

# Domain randomization: each robot in training gets its own values in these ranges, so the policy cannot rely on exact ones.
RANDOMIZE = {
  "motor_kd": (0.6, 1.5),            # x MOTOR_KD
  "motor_torque": (0.925, 1.08),     # x MOTOR_TORQUE_LIMIT: 0.148-0.173 N*m, the spec sheet's stall torque range
  "motor_mass": (0.8, 1.2),          # x the motor's mass (0.15 kg in build_beam_robot.py)
  "motor_com_shift": 0.005,          # m; the motor's centre of mass moves up to this far (battery, wiring)
  "beam_mass": (0.9, 1.1),           # x each beam's mass
  "beam_friction": (0.4, 1.2),       # beam ends on the floor (smooth table .. rubbery mat)
  "case_friction": (0.2, 0.8),       # motor case on the floor
}

# Disturbances: random shoves on the motor body during the episode (it weighs about 1.6 N), with a twist that turns it.
PUSH_FORCE = 0.8              # N, up to this much along each axis
PUSH_TWIST = 0.002            # N*m, up to this much about each axis (knocks its heading off, for it to correct)
PUSH_DURATION_S = (0.05, 0.2)
PUSH_EVERY_S = (1.0, 3.0)     # pause between shoves
# -------------------------------------------------------------------------------------------------------

_CONTROL_DT_MS = 20           # the policy acts every 20 ms (decimation 10 x 2 ms physics step, below)
_PHYSICS_DT_MS = 2


class LegoVelocityAction(JointVelocityAction):
  """Apply the same percentage rounding and per-output limits as the real BLE adapter."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._hardware_limits = torch.tensor(
      [MOTOR_COMMAND_LIMITS[name] for name in self.target_names], device=self.device
    )

  def process_actions(self, actions: torch.Tensor):
    super().process_actions(actions)
    percent = torch.round(100 * self._processed_actions / self._hardware_limits).clamp(-100, 100)
    self._processed_actions.copy_(percent * self._hardware_limits / 100)
    # The real adapter feeds the applied (saturated/rounded) command back to the policy.
    self._raw_actions.copy_(self._processed_actions / MAX_BEAM_SPEED)


@dataclass(kw_only=True)
class LegoVelocityActionCfg(JointVelocityActionCfg):
  def build(self, env):
    return LegoVelocityAction(self, env)


def _get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
  chassis = spec.body("chassis")
  chassis.add_freejoint()                  # let the robot move around
  # Where the motion sensor sits: the middle of the motor case (its exact place inside is not published).
  case = spec.geom("motor_case")
  chassis.add_site(name="imu", pos=list(case.pos))
  return spec


def _robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    # chassis origin is on the shaft, 15.2 mm above the bottom of the motor case; the beams start flat
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.017), joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        # stiffness 0 = no position target, so the motor only chases the speed we ask for
        # The gearbox inertia (armature) comes from beam_robot.xml.
        BuiltinPdActuatorCfg(
          target_names_expr=(LEFT, RIGHT), stiffness=0.0, damping=MOTOR_KD, effort_limit=MOTOR_TORQUE_LIMIT,
          # Bluetooth delay: each command arrives this many physics steps late, a new delay drawn every control step
          # (kept the same half of the time: the radio's delay drifts rather than jumping about).
          delay_min_lag=COMMAND_DELAY_MS[0] // _PHYSICS_DT_MS,
          delay_max_lag=COMMAND_DELAY_MS[1] // _PHYSICS_DT_MS,
          delay_update_period=_CONTROL_DT_MS // _PHYSICS_DT_MS,
          delay_hold_prob=0.5,
        ),
      ),
    ),
  )


# ---- observations and rewards ----------------------------------------------------------------------------
# The robot tumbles end over end about its shaft, so its body x and z axes keep turning over. Its shaft axis (body y)
# does not: where that points is the robot's heading. Driving straight down +x means the shaft stays along world y.


def _shaft_axis(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The shaft direction (body y axis) in world coordinates. Shape [num_envs, 3]."""
  w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
  return torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], dim=-1)


def _heading(env: ManagerBasedRlEnv) -> torch.Tensor:
  """How far the robot is turned from +x, in radians (0 = straight ahead, positive = turned left)."""
  axis = _shaft_axis(env)
  return torch.atan2(-axis[:, 0], axis[:, 1])


def _off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sideways distance from this env's start line, in metres."""
  return env.scene["robot"].data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]


def heading_and_offset(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(heading), cos(heading), off-line distance, how far the shaft tilts up]. Shape [num_envs, 4]."""
  h = _heading(env)
  return torch.stack([torch.sin(h), torch.cos(h), _off_line(env), _shaft_axis(env)[:, 2]], dim=-1)


def world_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Velocity of the shaft centre and spin of the body, in world coordinates. Shape [num_envs, 6]."""
  data = env.scene["robot"].data
  return torch.cat([data.root_link_lin_vel_w, data.root_link_ang_vel_w], dim=-1)


def beam_angles(env: ManagerBasedRlEnv) -> torch.Tensor:
  """sin and cos of each beam's angle (the angle itself grows without end as the beams spin). Shape [num_envs, 4]."""
  q = env.scene["robot"].data.joint_pos
  return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)


# ---- what the real robot can sense (the policy's inputs) ----------------------------------------------------
# The gyro and accelerometer are MuJoCo sensors at the "imu" site (scene sensors below); their noise and bias are added
# by the observation settings. Here: the encoders and the motion sensor's heading.


def _sensor_state(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  """Per-robot sensor errors that last a whole episode (drawn again by reset_sensor_errors)."""
  if not hasattr(env, "_beam_sensor_state"):
    n = env.num_envs
    env._beam_sensor_state = {
      "heading_error": torch.zeros(n, device=env.device),    # how far the heading has drifted so far
      "heading_drift": torch.zeros(n, device=env.device),    # its drift rate (the gyro's leftover bias)
      "encoder_offset": torch.zeros(n, 2, device=env.device),
      "average_speed": torch.zeros(n, device=env.device),    # for the reward (not a sensor): speed over ~0.5 s
    }
  return env._beam_sensor_state


def reset_sensor_errors(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """New episode = the program restarts: heading back to 0, new gyro drift, new encoder offsets."""
  s, n = _sensor_state(env), SENSOR_NOISE
  ids = slice(None) if env_ids is None else env_ids
  k = env.num_envs if env_ids is None else len(env_ids)
  s["heading_error"][ids] = 0.0
  s["heading_drift"][ids] = (torch.rand(k, device=env.device) * 2 - 1) * n["heading_drift"]
  s["encoder_offset"][ids] = (torch.rand(k, 2, device=env.device) * 2 - 1) * n["encoder_offset"]
  s["average_speed"][ids] = 0.0


def drift_heading(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Every control step the motion sensor's heading drifts a little further (bias + random wander)."""
  s, n = _sensor_state(env), SENSOR_NOISE
  dt = env.step_dt
  s["heading_error"] += s["heading_drift"] * dt + torch.randn_like(s["heading_error"]) * n["heading_walk"] * math.sqrt(dt)


def imu_heading(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The motion sensor's heading (the gyro summed up since the start, so it drifts): [sin, cos]. Shape [num_envs, 2]."""
  h = _heading(env) + _sensor_state(env)["heading_error"]
  return torch.stack([torch.sin(h), torch.cos(h)], dim=-1)


def imu_accel(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Accelerometer in g, in the motor's own frame (1 g "up" when it sits still). Shape [num_envs, 3]."""
  return env.scene["robot/imu_accel"].data / 9.81


def imu_gyro(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Gyro in rad/s, in the motor's own frame. Shape [num_envs, 3]."""
  return env.scene["robot/imu_gyro"].data


def encoder_angles(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Both output encoders: whole degrees, off by up to 1 degree. [sin, sin, cos, cos]. Shape [num_envs, 4]."""
  step = SENSOR_NOISE["encoder_step"]
  q = env.scene["robot"].data.joint_pos + _sensor_state(env)["encoder_offset"]
  q = torch.round(q / step) * step
  return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)


def encoder_speeds(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Both outputs' speeds in whole degrees per second (noise added by the observation settings). Shape [num_envs, 2]."""
  step = SENSOR_NOISE["speed_step"]
  return torch.round(env.scene["robot"].data.joint_vel / step) * step


def average_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The ~0.5 s average forward speed the speed reward uses (critic only). Shape [num_envs, 1]."""
  return _sensor_state(env)["average_speed"].unsqueeze(-1)


SPEED_AVERAGE_S = 0.5   # s; the speed reward looks at the speed averaged over about this long


def forward_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for going along +x: the speed averaged over ~0.5 s, as a share of TARGET_SPEED (0 standing still, 1 at or
  above the target, negative backwards).

  Averaged, because a tumbling robot's speed swings from 0 to well above its average within every roll; scoring each
  instant would punish exactly that, and push the policy to barely move. Straight (not peaked) up to the target, so
  every bit of extra average speed pays the same.
  """
  vx = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
  avg = _sensor_state(env)["average_speed"]
  avg += (vx - avg) * min(1.0, env.step_dt / SPEED_AVERAGE_S)
  return torch.clamp(avg / TARGET_SPEED, min=-1.0, max=1.0)


def stay_on_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  """-1 at 20 cm off the line. Gentle: the robot cannot sense where it is, only which way it points (face_forward)."""
  return -torch.square(_off_line(env) / 0.2)


def face_forward(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(_heading(env))


def keep_shaft_level(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalty for tilting the shaft up: turning too hard levers the robot up onto one end, where it is stuck."""
  return -torch.square(_shaft_axis(env)[:, 2])


def too_far_off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _off_line(env).abs() > MAX_OFF_LINE


def standing_on_end(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The shaft points more up than sideways: the robot has tipped onto one end and cannot drive any more."""
  return _shaft_axis(env)[:, 2].abs() > 0.8


# ---- environment ----------------------------------------------------------------------------------------


def beam_drive_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  n = SENSOR_NOISE

  def noisy(noise: float, bias: float = 0.0):
    """Gaussian noise on every reading, plus (if bias > 0) an error fixed for the episode, drawn from +-bias."""
    if bias <= 0.0:
      return GaussianNoiseCfg(std=noise)
    return NoiseModelWithAdditiveBiasCfg(
      noise_cfg=GaussianNoiseCfg(std=noise),
      bias_noise_cfg=UniformNoiseCfg(n_min=-bias, n_max=bias, operation="abs"),
    )

  # The policy (actor): only the robot's own sensors, with their errors, and late by 0-40 ms over Bluetooth (each robot
  # its own delay). It knows its own last command exactly.
  lag = dict(delay_min_lag=SENSOR_DELAY_MS[0] // _CONTROL_DT_MS, delay_max_lag=SENSOR_DELAY_MS[1] // _CONTROL_DT_MS,
             delay_hold_prob=0.5)
  actor_terms = {
    "gyro": ObservationTermCfg(func=imu_gyro, noise=noisy(n["gyro_noise"], n["gyro_bias"]), **lag),
    "accel": ObservationTermCfg(func=imu_accel, noise=noisy(n["accel_noise"], n["accel_bias"]),
                                clip=(-n["accel_range"], n["accel_range"]), **lag),
    "heading": ObservationTermCfg(func=imu_heading, **lag),           # drift added in imu_heading
    "encoder_angles": ObservationTermCfg(func=encoder_angles, **lag),  # whole degrees, offset per episode
    "encoder_speeds": ObservationTermCfg(func=encoder_speeds, noise=noisy(n["speed_noise"]), **lag),
    "last_action": ObservationTermCfg(func=last_action, params={"action_name": "beams"}),
  }
  # The critic, used only during training to judge how well things are going, may see everything, exactly and at once:
  # where the robot really is, how fast it really moves, which way it really points.
  critic_terms = {
    "velocity": ObservationTermCfg(func=world_velocity),
    "gravity": ObservationTermCfg(func=projected_gravity),
    "heading_offset": ObservationTermCfg(func=heading_and_offset),
    "beam_angles": ObservationTermCfg(func=beam_angles),
    "beam_vel": ObservationTermCfg(func=joint_vel_rel),
    "last_action": ObservationTermCfg(func=last_action, params={"action_name": "beams"}),
    "average_speed": ObservationTermCfg(func=average_speed),   # what the speed reward is computed from
  }
  observations = {
    # noise on in training and in play: the film should show the policy as it would run on the real robot.
    # history: the policy sees its last SENSOR_HISTORY readings, not just the newest. From one noisy, late snapshot it
    # cannot tell how fast it moves or where it is in its tumble; from how the readings change, it can.
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True, history_length=SENSOR_HISTORY),
    "critic": ObservationGroupCfg(critic_terms),
  }

  actions: dict[str, ActionTermCfg] = {
    "beams": LegoVelocityActionCfg(
      entity_name="robot", actuator_names=(LEFT, RIGHT), scale=MAX_BEAM_SPEED, use_default_offset=True,
      clip={name: (-limit, limit) for name, limit in MOTOR_COMMAND_LIMITS.items()},
    ),
  }

  def robot(**names) -> SceneEntityCfg:
    return SceneEntityCfg("robot", **names)

  r = RANDOMIZE
  events = {
    "reset_pose": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {"yaw": (-START_YAW, START_YAW)}, "velocity_range": {}},   # on the start line, crooked
    ),
    # ---- the robot's sensor errors: new each episode; the heading drifts a little every step ----
    "sensor_errors": EventTermCfg(func=reset_sensor_errors, mode="reset"),
    "heading_drift": EventTermCfg(func=drift_heading, mode="step"),
    # ---- domain randomization. Masses once per robot at the start; the rest again at every episode. ----
    "motor_mass": EventTermCfg(
      func=dr.pseudo_inertia,     # scales mass and inertia together (mass x e^(2 alpha)) and moves the centre of mass
      mode="startup",
      params={
        "asset_cfg": robot(body_names=("chassis",)),
        "alpha_range": (math.log(r["motor_mass"][0]) / 2, math.log(r["motor_mass"][1]) / 2),
        "t_range": (-r["motor_com_shift"], r["motor_com_shift"]),
      },
    ),
    "beam_mass": EventTermCfg(
      func=dr.pseudo_inertia,
      mode="startup",
      params={
        "asset_cfg": robot(body_names=(LEFT, RIGHT)),
        "alpha_range": (math.log(r["beam_mass"][0]) / 2, math.log(r["beam_mass"][1]) / 2),
      },
    ),
    "motor_gains": EventTermCfg(
      func=dr.pd_gains,
      mode="reset",
      params={"asset_cfg": robot(), "kp_range": (1.0, 1.0), "kd_range": r["motor_kd"], "operation": "scale"},
    ),
    "motor_torque": EventTermCfg(
      func=dr.effort_limits,
      mode="reset",
      params={"asset_cfg": robot(), "effort_limit_range": r["motor_torque"], "operation": "scale"},
    ),
    "beam_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="reset",
      params={"asset_cfg": robot(geom_names=(f"{LEFT}.*", f"{RIGHT}.*")), "ranges": r["beam_friction"], "operation": "abs",
              "shared_random": True},   # both beams on the same floor
    ),
    "case_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="reset",
      params={"asset_cfg": robot(geom_names=("motor_case",)), "ranges": r["case_friction"], "operation": "abs"},
    ),
    # ---- disturbances: shoves on the motor body, a little above its centre so they can also tip it ----
    "push": EventTermCfg(
      func=apply_body_impulse,
      mode="step",
      params={
        "asset_cfg": robot(body_names=("chassis",)),
        "force_range": (-PUSH_FORCE, PUSH_FORCE),
        "torque_range": (-PUSH_TWIST, PUSH_TWIST),
        "duration_s": PUSH_DURATION_S,
        "cooldown_s": PUSH_EVERY_S,
        "body_point_offset": (0.0, 0.0, 0.01),
      },
    ),
  }

  rewards = {
    # The main reward, and the only positive one: forward progress. Standing still earns nothing (and still pays the
    # penalties below), so it is never safest to stay put; once moving, every step is worth something, so tipping over
    # costs. (Tried and dropped: a flat reward for every step it is alive. The policy learned to sit still and collect it.)
    "forward_speed": RewardTermCfg(func=forward_speed, weight=2.0),
    "stay_on_line": RewardTermCfg(func=stay_on_line, weight=0.25),
    "face_forward": RewardTermCfg(func=face_forward, weight=0.5),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.02),
    "shaft_level": RewardTermCfg(func=keep_shaft_level, weight=2.0),
    # Ending up on an end costs 0.5 at once, plus the progress it no longer makes (a perfect episode earns 20; every
    # reward is weight x value x 0.02 s). Much more (we tried 2 at once) and the policy learns that not moving is the
    # safest way not to fall.
    "fell": RewardTermCfg(func=is_terminated, weight=-25.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "off_line": TerminationTermCfg(func=too_far_off_line),
    "on_end": TerminationTermCfg(func=standing_on_end),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": _robot_cfg()},
      # the Double Motor's motion sensor, at the "imu" site in the motor case
      sensors=(
        BuiltinSensorCfg(name="imu_gyro", sensor_type="gyro", obj=ObjRef(type="site", name="imu", entity="robot")),
        BuiltinSensorCfg(name="imu_accel", sensor_type="accelerometer",
                         obj=ObjRef(type="site", name="imu", entity="robot")),
      ),
      num_envs=1,
      env_spacing=3.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,   # follow the robot (its shaft centre)
      entity_name="robot",
      distance=0.8,
      elevation=-25.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.002)),   # small step: light beams, hard tumbling contacts
    decimation=10,                                           # the policy acts 50 times a second
    episode_length_s=1e10 if play else EPISODE_S,
  )


def beam_drive_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(128, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="beam_drive_straight",
    logger="tensorboard",           # the default, wandb, needs an account and a login
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=2000,
  )


register_mjlab_task(
  task_id="Beam-Drive-Straight",
  env_cfg=beam_drive_env_cfg(),
  play_env_cfg=beam_drive_env_cfg(play=True),
  rl_cfg=beam_drive_ppo_runner_cfg(),
)
