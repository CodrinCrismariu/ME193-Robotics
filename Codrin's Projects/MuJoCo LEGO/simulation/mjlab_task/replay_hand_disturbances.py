"""Replay video-timed, approximate hand wrenches without resetting or scripting the robot's motion.

Usage: python replay_hand_disturbances.py model_299.pt hand_schedule.json
Stages alongside beam_task.py and onshape/. The schedule times use the source video's clock.
"""
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import mediapy as media
import torch

import beam_task  # registers task and the physical adapter's command limits
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def main():
  checkpoint, schedule_path = map(Path, sys.argv[1:])
  schedule = json.loads(schedule_path.read_text())
  configure_torch_backends()
  torch.manual_seed(schedule["seed"])
  cfg = load_env_cfg("Beam-Drive-Straight", play=True)
  cfg.seed = schedule["seed"]
  cfg.scene.num_envs = 1
  cfg.viewer.width = cfg.viewer.height = 720
  cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = .35, -65., 180.
  # Nominal physical parameters; retain imperfect sensors, but isolate the scheduled hand contacts.
  cfg.events = {name: event for name, event in cfg.events.items()
                if name in ("reset_pose", "sensor_errors", "heading_drift")}
  cfg.events["reset_pose"].params["pose_range"] = {"yaw": (0., 0.)}
  cfg.terminations = {}  # Never teleport/reset a failed rollout for the comparison.
  agent = load_rl_cfg("Beam-Drive-Straight")
  base = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
  env = RslRlVecEnvWrapper(base, clip_actions=agent.clip_actions)
  try:
    runner = (load_runner_cls("Beam-Drive-Straight") or MjlabOnPolicyRunner)(env, asdict(agent), device="cuda:0")
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    robot = base.scene["robot"]
    body_ids = robot.find_bodies("chassis")[0]
    dt = cfg.sim.mujoco.timestep * cfg.decimation
    fps = round(1 / dt)
    rows = []
    obs = env.get_observations()
    with torch.inference_mode(), media.VideoWriter("simulation.mp4", shape=(720, 720), fps=fps) as writer:
      for step in range(round(schedule["duration_s"] / dt)):
        t = step * dt
        writer.add_image(base.render())
        force = torch.zeros((1, 1, 3), device="cuda:0")
        torque = torch.zeros_like(force)
        active = []
        for event in schedule["wrenches"]:
          if event["start_s"] <= t < event["end_s"]:
            phase = (t - event["start_s"]) / (event["end_s"] - event["start_s"])
            strength = math.sin(math.pi * phase) ** 2
            force += torch.tensor(event["force_world_N"], device="cuda:0") * strength
            torque += torch.tensor(event["torque_world_Nm"], device="cuda:0") * strength
            active.append(event["contact"])
        robot.write_external_wrench_to_sim(force, torque, body_ids=body_ids)
        action = policy(obs) if t >= schedule["policy_start_s"] else torch.zeros((1, 2), device="cuda:0")
        obs = env.step(action)[0]
        pos = robot.data.root_link_pos_w[0].tolist()
        vel = robot.data.joint_vel[0].tolist()
        command = base.action_manager.get_term("beams")._processed_actions[0].tolist()
        rows.append([t, *pos, *vel, *command, *force.flatten().tolist(), *torque.flatten().tolist(), " ".join(active)])
        if step % fps == 0:
          print(f"t={t:5.2f}  x={pos[0]:+.3f} y={pos[1]:+.3f}  contacts={active}", flush=True)
    with Path("telemetry.csv").open("w") as f:
      writer = csv.writer(f)
      writer.writerow(["time_s", "x_m", "y_m", "z_m", "left_velocity_rad_s", "right_velocity_rad_s",
                       "left_command_rad_s", "right_command_rad_s", "fx_N", "fy_N", "fz_N",
                       "tx_Nm", "ty_Nm", "tz_Nm", "contacts"])
      writer.writerows(rows)
    print("Recorded 20 seconds without scripted poses or automatic resets.")
  finally:
    env.close()


if __name__ == "__main__":
  main()
