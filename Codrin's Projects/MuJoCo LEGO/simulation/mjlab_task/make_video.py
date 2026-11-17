"""Record a video of a trained policy. Made for the cluster, which has no screen.

    python3 runIt.py --gpu --time 00:10:00 --mem 16G \
      --with 09_onshape_robot/mjlab_task/drive_task.py --with 09_onshape_robot/onshape \
      --with runs/<your run>/logs/rsl_rl/drive_straight/<date folder>/model_499.pt \
      09_onshape_robot/mjlab_task/make_video.py model_499.pt

For the beam robot, send beam_task.py instead of drive_task.py and name the task after the checkpoint:
      ... --with 09_onshape_robot/mjlab_task/beam_task.py ... make_video.py model_499.pt Beam-Drive-Straight

runIt.py brings the .mp4 home to ./runs/ and opens it. (mjlab's own play.py also records a video, but then opens a
web viewer that never exits on a machine without a screen.)
"""
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import torch

if importlib.util.find_spec("drive_task"):
  import drive_task  # noqa: F401  (registers "Drive-Straight")
if importlib.util.find_spec("beam_task"):
  import beam_task  # noqa: F401  (registers "Beam-Drive-Straight")
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

SECONDS = 10

if len(sys.argv) not in (2, 3):
  sys.exit("usage: make_video.py model_499.pt [Drive-Straight | Beam-Drive-Straight]")
checkpoint = Path(sys.argv[1])
TASK = sys.argv[2] if len(sys.argv) == 3 else "Drive-Straight"

configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
env_cfg.viewer.width, env_cfg.viewer.height = 640, 480
agent_cfg = load_rl_cfg(TASK)
steps_per_second = round(1 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation))
steps = SECONDS * steps_per_second

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
env = VideoRecorder(env, video_folder=".", step_trigger=lambda step: step == 0, video_length=steps, disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)


def heading(robot) -> float:
  """Degrees turned away from +x. The beam robot tumbles, so for it this is where its shaft points, not its body."""
  w, x, y, z = robot.data.root_link_quat_w[0].tolist()
  if TASK.startswith("Beam"):
    ax, ay = 2 * (x * y - w * z), 1 - 2 * (x * x + z * z)     # shaft (body y axis) in the world
    return torch.atan2(torch.tensor(-ax), torch.tensor(ay)).item() * 57.296
  return torch.atan2(torch.tensor(2 * (w * z + x * y)), torch.tensor(1 - 2 * (y * y + z * z))).item() * 57.296


robot = env.unwrapped.scene["robot"]
start = robot.data.root_link_pos_w[0, :2].clone()
print("  second   x (m)   y (m)   heading (deg)   speed (m/s)")
obs = env.get_observations()
with torch.no_grad():
  for step in range(steps):
    obs = env.step(policy(obs))[0]
    if step % steps_per_second == steps_per_second - 1:   # once a second
      x, y = (robot.data.root_link_pos_w[0, :2] - start).tolist()
      vx = robot.data.root_link_lin_vel_w[0, 0].item()
      print(f"  {(step + 1) / steps_per_second:5.0f}  {x:7.2f} {y:7.2f}  {heading(robot):11.1f}  {vx:12.2f}")
env.close()
print("wrote the video (an .mp4 next to this script)")
